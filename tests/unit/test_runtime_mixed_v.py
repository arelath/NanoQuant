from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nanoquant.runtime import (
    MIXED_V_FORMAT_VERSION,
    MIXED_V_RECORD_BITS,
    FactorizedReferenceBackend,
    LogicalLayerState,
    MixedVArtifactError,
    MixedVLayerState,
    PackedLayerState,
    PackedReferenceBackend,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    convert_logical_to_packed,
    mixed_v_from_packed_words,
    mixed_v_payload_word_count,
    open_mixed_v_artifact,
    pack_mixed_v_records,
    pack_sign_matrix,
    unpack_mixed_v_records,
    write_logical_artifact,
    write_mixed_v_artifact,
)


def test_mixed_v_records_use_minimal_exact_19_bit_stream() -> None:
    indices = (torch.arange(34, dtype=torch.int32).reshape(2, 17) * 29) % 1024
    pair_ids = (torch.arange(34, dtype=torch.int32).reshape(2, 17) * 13) % 496

    payload = pack_mixed_v_records(indices, pair_ids)
    restored_indices, restored_pairs = unpack_mixed_v_records(payload, (2, 17))

    assert MIXED_V_RECORD_BITS == 19
    assert payload.shape == (mixed_v_payload_word_count(34),)
    assert payload.numel() == 21
    assert torch.equal(restored_indices, indices)
    assert torch.equal(restored_pairs, pair_ids)
    assert mixed_v_payload_word_count(1088 * 216) == 139_536


def test_mixed_v_records_reject_invalid_pair_ids_and_padding() -> None:
    invalid_pair = torch.tensor([496 << 10], dtype=torch.int32)
    with pytest.raises(ValueError, match="496 valid pairs"):
        unpack_mixed_v_records(invalid_pair, (1, 1))

    padded = pack_mixed_v_records(
        torch.zeros((1, 1), dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int32),
    )
    padded[0] = -(1 << 31)
    with pytest.raises(ValueError, match="non-zero padding"):
        unpack_mixed_v_records(padded, (1, 1))


def _mixed_state(name: str = "blocks.0.linear") -> tuple[MixedVLayerState, PackedLayerState]:
    spec = QuantizedLinearSpec(
        name,
        "nanoquant-v1",
        35,
        3,
        7,
        "float32",
        "float32",
    )
    free_rows = 2
    words_per_row = 2
    codebook = torch.zeros(1024, dtype=torch.int32)
    codebook[0] = 0x12345678
    codebook[1] = 3
    indices = torch.tensor([[0, 1]] * (spec.rank - free_rows), dtype=torch.int32)
    pairs = torch.zeros_like(indices)
    free_right = pack_sign_matrix(
        torch.where(
            torch.arange(free_rows * spec.in_features).reshape(free_rows, spec.in_features)
            % 3
            == 0,
            -1.0,
            1.0,
        )
    )
    payload = pack_mixed_v_records(indices, pairs)
    # Pair 0 is (bit 0, bit 1); codebook[1] therefore decodes to an all-zero tail word.
    coded = torch.tensor(
        [[0x12345678 ^ 3, 0]] * (spec.rank - free_rows),
        dtype=torch.int32,
    )
    right_words = torch.cat((free_right, coded), dim=0)
    left_words = pack_sign_matrix(
        torch.where(
            torch.arange(spec.out_features * spec.rank).reshape(3, 7) % 2 != 0,
            1.0,
            -1.0,
        )
    )
    packed = PackedLayerState(
        spec,
        "llama.cpp-i32-lsb-v1",
        left_words,
        right_words,
        torch.linspace(0.5, 1.5, spec.in_features),
        torch.linspace(0.75, 1.25, spec.rank),
        torch.linspace(1.0, 1.5, spec.out_features),
    )
    mixed = MixedVLayerState(
        spec,
        MIXED_V_FORMAT_VERSION,
        free_rows,
        left_words,
        free_right,
        payload,
        codebook,
        packed.scale_pre,
        packed.scale_mid,
        packed.scale_post,
    )
    assert words_per_row == mixed.free_right_words.shape[1]
    return mixed, packed


def test_mixed_v_state_predecodes_exactly_and_reference_backend_executes_it() -> None:
    mixed, packed = _mixed_state()

    assert torch.equal(mixed.to_packed().right_words, packed.right_words)
    value = torch.linspace(-0.5, 0.5, mixed.spec.in_features).reshape(1, -1)
    backend = PackedReferenceBackend()
    actual = backend.linear(value, backend.prepare(mixed, "cpu"))
    expected = backend.linear(value, backend.prepare(packed, "cpu"))

    assert torch.equal(actual, expected)


def test_mixed_v_state_rejects_a_correction_that_sets_sign_padding() -> None:
    mixed, _packed = _mixed_state()
    indices, pairs = unpack_mixed_v_records(
        mixed.coded_payload,
        (mixed.coded_rows, mixed.free_right_words.shape[1]),
    )
    pairs[:, 1] = 30  # Lexicographic pair (0, 31), invalid for a 35-feature tail word.

    with pytest.raises(ValueError, match="non-zero padding bit"):
        MixedVLayerState(
            mixed.spec,
            mixed.format,
            mixed.free_rows,
            mixed.left_words,
            mixed.free_right_words,
            pack_mixed_v_records(indices, pairs),
            mixed.codebook_words,
            mixed.scale_pre,
            mixed.scale_mid,
            mixed.scale_post,
        )


def test_mixed_v_builder_rejects_records_for_a_different_right_factor() -> None:
    _mixed, packed = _mixed_state()
    codebook = torch.zeros(1024, dtype=torch.int32)
    indices = torch.zeros((5, 2), dtype=torch.int32)
    pairs = torch.zeros_like(indices)

    with pytest.raises(ValueError, match="do not reconstruct"):
        mixed_v_from_packed_words(
            packed,
            free_rows=2,
            codebook_words=codebook,
            codebook_indices=indices,
            correction_pair_ids=pairs,
        )


def _base_state(name: str, rank: int) -> LogicalLayerState:
    spec = QuantizedLinearSpec(name, "nanoquant-v1", 35, 3, rank, "float32", "float32")
    return LogicalLayerState(
        spec,
        torch.where(torch.arange(3 * rank).reshape(3, rank) % 2 != 0, 1.0, -1.0),
        torch.where(torch.arange(rank * 35).reshape(rank, 35) % 3 != 0, 1.0, -1.0),
        torch.ones(35),
        torch.ones(rank),
        torch.ones(3),
    )


def _base_artifact(tmp_path: Path):
    logical = write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/model", "revision", "fixture", "config", "tokenizer"),
        {
            0: (_base_state("blocks.0.linear", 5),),
            1: (_base_state("blocks.1.linear", 5),),
        },
    )
    return convert_logical_to_packed(logical.root, tmp_path / "packed")


def test_mixed_v_overlay_persists_replacements_and_falls_back_to_base(tmp_path: Path) -> None:
    base = _base_artifact(tmp_path)
    mixed, expected = _mixed_state()

    overlay = write_mixed_v_artifact(tmp_path / "mixed", base, {0: (mixed,)})
    compact = overlay.load_runtime_layer("blocks.0.linear")
    fallback = overlay.load_runtime_layer("blocks.1.linear")

    assert isinstance(compact, MixedVLayerState)
    assert isinstance(fallback, PackedLayerState)
    assert overlay.replacement_names == frozenset(("blocks.0.linear",))
    assert [spec.rank for spec in overlay.layer_specs()] == [7, 5]
    assert torch.equal(overlay.load_packed_layer("blocks.0.linear").right_words, expected.right_words)
    assert overlay.manifest.layer_count == 1
    assert overlay.manifest.artifact_bytes == overlay.manifest.blocks[0].bytes


def test_mixed_v_overlay_rejects_wrong_base_and_future_schema(tmp_path: Path) -> None:
    base = _base_artifact(tmp_path)
    mixed, _expected = _mixed_state()
    overlay = write_mixed_v_artifact(tmp_path / "mixed", base, {0: (mixed,)})
    descriptor = overlay.root / "nanoquant-mixed-v-overlay.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["base_packed_descriptor_sha256"] = "0" * 64
    descriptor.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MixedVArtifactError, match="different packed base"):
        open_mixed_v_artifact(overlay.root, base, verify_hashes=False)

    payload["base_packed_descriptor_sha256"] = overlay.manifest.base_packed_descriptor_sha256
    payload["schema_version"] = 2
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MixedVArtifactError, match="unsupported mixed-V artifact schema"):
        open_mixed_v_artifact(overlay.root, base, verify_hashes=False)


def test_mixed_v_overlay_output_is_executable_against_logical_replacement(tmp_path: Path) -> None:
    base = _base_artifact(tmp_path)
    mixed, _packed = _mixed_state()
    overlay = write_mixed_v_artifact(tmp_path / "mixed", base, {0: (mixed,)})
    value = torch.linspace(-1, 1, 35).reshape(1, 35)
    packed_backend = PackedReferenceBackend()
    logical_backend = FactorizedReferenceBackend()

    actual = packed_backend.linear(
        value,
        packed_backend.prepare(overlay.load_runtime_layer("blocks.0.linear"), "cpu"),
    )
    expected = logical_backend.linear(
        value,
        logical_backend.prepare(mixed.to_logical(), "cpu"),
    )

    assert torch.equal(actual, expected)
