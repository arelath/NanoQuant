from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nanoquant.runtime import (
    PRODUCT_CODEBOOK_FORMAT_VERSION,
    LogicalLayerState,
    ProductCodebookArtifactError,
    ProductCodebookLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    convert_logical_to_packed,
    open_product_codebook_artifact,
    pack_product_codebook_indices,
    pack_product_half_table,
    pack_sign_matrix,
    unpack_product_codebook_indices,
    write_logical_artifact,
    write_product_codebook_artifact,
)


def _signs(rows: int, columns: int, modulus: int) -> torch.Tensor:
    flat = torch.arange(rows * columns).reshape(rows, columns)
    return torch.where(flat % modulus == 0, -1.0, 1.0).contiguous()


def _table_for_words(words: list[int]) -> torch.Tensor:
    result = torch.ones((256, 16))
    for index, word in enumerate(words):
        bits = (torch.tensor(word, dtype=torch.int64) >> torch.arange(16)) & 1
        result[index] = (1 - 2 * bits).to(torch.float32)
    return result.contiguous()


def test_product_codebook_indices_use_an_exact_16_bit_stream() -> None:
    indices = ((torch.arange(15).reshape(3, 5) * 7919) % 65536).to(torch.int32)

    payload = pack_product_codebook_indices(indices)

    assert payload.shape == (8,)
    assert torch.equal(unpack_product_codebook_indices(payload, (3, 5)), indices)
    payload[-1] |= torch.tensor(1 << 16, dtype=torch.int32)
    with pytest.raises(ValueError, match="non-zero padding"):
        unpack_product_codebook_indices(payload, (3, 5))


def _compact_state(*, transposed: bool = False) -> ProductCodebookLayerState:
    spec = QuantizedLinearSpec(
        "blocks.0.linear",
        "nanoquant-v1",
        35,
        3,
        7,
        "float32",
        "float32",
    )
    factor_out = spec.in_features if transposed else spec.out_features
    factor_in = spec.out_features if transposed else spec.in_features
    free_rows = 2
    left = _signs(factor_out, spec.rank, 3)
    free_right = _signs(free_rows, factor_in, 2)
    coded_words = [0x12345678, 0x00000005] if factor_in > 32 else [0x00000005]
    index_row = torch.tensor(
        [0 if index == 0 else index | (index << 8) for index in range(len(coded_words))],
        dtype=torch.int32,
    )
    indices = index_row[None, :].repeat(spec.rank - free_rows, 1)
    first = _table_for_words([word & 0xFFFF for word in coded_words])
    second = _table_for_words([(word >> 16) & 0xFFFF for word in coded_words])
    return ProductCodebookLayerState(
        spec,
        PRODUCT_CODEBOOK_FORMAT_VERSION,
        transposed,
        free_rows,
        pack_sign_matrix(left),
        pack_sign_matrix(free_right),
        pack_product_codebook_indices(indices),
        pack_product_half_table(first),
        pack_product_half_table(second),
        torch.linspace(0.5, 1.5, factor_in),
        torch.linspace(0.75, 1.25, spec.rank),
        torch.linspace(1.0, 1.5, factor_out),
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_product_codebook_state_predecodes_factorization_orientation(transposed: bool) -> None:
    compact = _compact_state(transposed=transposed)
    packed = compact.to_packed()

    assert packed.spec == compact.spec
    logical = packed.to_logical()
    assert logical.left_binary.shape == (3, 7)
    assert logical.right_binary.shape == (7, 35)
    assert compact.compact_logical_bits() == (
        compact.factor_left_words.numel() * 32
        + compact.factor_right_free_words.numel() * 32
        + compact.record_count * 16
        + 8192
        + 16
        + (compact.factor_in_features + compact.spec.rank + compact.factor_out_features) * 16
    )


def _base_artifact(tmp_path: Path):  # type: ignore[no-untyped-def]
    spec = QuantizedLinearSpec(
        "blocks.0.linear", "nanoquant-v1", 35, 3, 5, "float32", "float32"
    )
    logical = LogicalLayerState(
        spec,
        _signs(3, 5, 2),
        _signs(5, 35, 3),
        torch.ones(35),
        torch.ones(5),
        torch.ones(3),
    )
    artifact = write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/model", "revision", "fixture", "config", "tokenizer"),
        {0: (logical,)},
    )
    return convert_logical_to_packed(artifact.root, tmp_path / "packed")


def test_product_codebook_artifact_is_base_bound_and_executable(tmp_path: Path) -> None:
    base = _base_artifact(tmp_path)
    compact = _compact_state()
    overlay = write_product_codebook_artifact(
        tmp_path / "product",
        base,
        {0: (compact,)},
        allocation_sha256="a" * 64,
        allocation_total_bits=123456,
        effective_bpw=0.999,
        correction_source_sha256="b" * 64,
        replay={"maximum_rmse": 0.0},
    )

    restored = overlay.load_compact_layer(compact.spec.name)
    assert torch.equal(restored.to_packed().right_words, compact.to_packed().right_words)
    assert overlay.manifest.compact_mlp_bits == compact.compact_logical_bits()

    descriptor = overlay.root / "nanoquant-product-codebook-overlay.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["base_packed_descriptor_sha256"] = "0" * 64
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProductCodebookArtifactError, match="another base"):
        open_product_codebook_artifact(overlay.root, base, verify_hashes=False)
