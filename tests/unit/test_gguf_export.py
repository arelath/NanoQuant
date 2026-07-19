from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

import nanoquant.infrastructure.gguf_export as gguf_export
from nanoquant.infrastructure.gguf_export import export_llamacpp_gguf
from nanoquant.infrastructure.io_utils import hash_file as real_hash_file
from nanoquant.infrastructure.mmproj_export import MmprojExportResult
from nanoquant.runtime import (
    PACKED_REFERENCE_CONVERTER_SHA256,
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    convert_logical_to_packed,
    write_logical_artifact,
)


def _packed(tmp_path: Path) -> Path:
    spec = QuantizedLinearSpec(
        "blocks.0.self_attn.q_proj",
        "nanoquant-v1",
        32,
        2,
        32,
        "float32",
        "float32",
    )
    state = LogicalLayerState(
        spec,
        torch.ones((2, 32)),
        torch.ones((32, 32)),
        torch.ones(32),
        torch.ones(32),
        torch.ones(2),
    )
    logical = write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/model", "revision", "gemma3", "config", "tokenizer"),
        {0: (state,)},
    )
    return convert_logical_to_packed(logical.root, tmp_path / "packed").root


def test_gguf_export_is_converter_pinned_and_resumable(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    packed = _packed(tmp_path)
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"vision_config": {"hidden_size": 16}}),
        encoding="utf-8",
    )
    reference = tmp_path / "llama.cpp"
    reference.mkdir()
    converter = tmp_path / "vendored" / "convert_nanoquant_to_gguf.py"
    converter.parent.mkdir()
    converter.write_text(
        """from pathlib import Path
import argparse
p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--nanoquant-checkpoint')
p.add_argument('--outfile')
p.add_argument('--outtype')
p.add_argument('--no-lazy', action='store_true')
a = p.parse_args()
Path(a.outfile).write_bytes(b'GGUF-fixture')
""",
        encoding="utf-8",
    )
    quantizer = reference / "build" / "bin" / "Release" / "llama-quantize.exe"
    quantizer.parent.mkdir(parents=True)
    quantizer.write_bytes(b"fixture quantizer")

    def pinned_hash(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == converter.resolve():
            return PACKED_REFERENCE_CONVERTER_SHA256
        return real_hash_file(resolved)

    monkeypatch.setattr(gguf_export, "hash_file", pinned_hash)
    commands: list[tuple[str, ...]] = []
    command_environments: list[dict[str, str] | None] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(command)
        commands.append(command)
        command_environments.append(kwargs.get("env"))
        if command[1] == str(converter):
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF-converted")
        else:
            Path(command[-2]).write_bytes(b"GGUF-q8_0")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gguf_export.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gguf_export,
        "_inspect_gguf_tensor_contract",
        lambda *_args: ("q8_0", 3, ("bf16",)),
    )
    mmproj_calls: list[Path] = []

    def export_mmproj(_source, mmproj_output, _reference, **_kwargs):  # type: ignore[no-untyped-def]
        mmproj_path = Path(mmproj_output).resolve()
        mmproj_calls.append(mmproj_path)
        return MmprojExportResult(
            mmproj_path,
            reference / "convert_hf_to_gguf.py",
            456,
            "sha256:mmproj",
            7,
            ("bf16", "f32"),
            len(mmproj_calls) > 1,
        )

    monkeypatch.setattr(gguf_export, "export_mmproj_bfloat16", export_mmproj)
    output = tmp_path / "output" / "model.gguf"
    checkpoint = tmp_path / "output" / "checkpoint"

    first = export_llamacpp_gguf(
        packed,
        source,
        checkpoint,
        output,
        reference,
        converter_path=converter,
    )
    second = export_llamacpp_gguf(
        packed,
        source,
        checkpoint,
        output,
        reference,
        converter_path=converter,
    )

    assert output.read_bytes() == b"GGUF-q8_0"
    assert not first.reused
    assert second.reused
    assert first.mmproj is not None and not first.mmproj.reused
    assert second.mmproj is not None and second.mmproj.reused
    assert mmproj_calls == [output.parent / "mmproj-BF16.gguf"] * 2
    receipt = json.loads(output.with_suffix(".gguf.export.json").read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 3
    assert receipt["gguf_sha256"] == real_hash_file(output)
    assert receipt["converter_sha256"] == PACKED_REFERENCE_CONVERTER_SHA256
    assert receipt["token_embedding_type"] == "q8_0"
    assert receipt["nanoquant_scale_type"] == "bf16"
    assert receipt["nanoquant_scale_tensor_count"] == 3
    assert receipt["quantizer_sha256"] == real_hash_file(quantizer)
    assert len(commands) == 2
    converter_environment = command_environments[0]
    assert converter_environment is not None
    assert converter_environment["NO_LOCAL_GGUF"] == "1"
    assert str(reference) in converter_environment["PYTHONPATH"]
    assert str(reference / "gguf-py") in converter_environment["PYTHONPATH"]
    assert commands[1][1:3] == ("--token-embedding-type", "Q8_0")
    assert commands[1][-1] == "F16"


def test_gguf_export_contract_rejects_widened_or_missing_scales() -> None:
    with pytest.raises(ValueError, match="must all be BF16"):
        gguf_export._require_bfloat16_nanoquant_scales(3, ("f32",), 3)
    with pytest.raises(ValueError, match="tensor count differs"):
        gguf_export._require_bfloat16_nanoquant_scales(0, (), 3)


def test_gguf_export_rejects_unsupported_embedding_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported token embedding"):
        export_llamacpp_gguf(
            tmp_path / "packed",
            tmp_path / "snapshot",
            tmp_path / "checkpoint",
            tmp_path / "model.gguf",
            tmp_path / "llama.cpp",
            token_embedding_type="bf16",
        )


def test_gguf_export_rejects_unpinned_converter(tmp_path: Path) -> None:
    packed = _packed(tmp_path)
    source = tmp_path / "snapshot"
    source.mkdir()
    reference = tmp_path / "llama.cpp"
    reference.mkdir()
    (reference / "convert_nanoquant_to_gguf.py").write_text("# wrong\n", encoding="utf-8")

    try:
        export_llamacpp_gguf(
            packed,
            source,
            tmp_path / "checkpoint",
            tmp_path / "model.gguf",
            reference,
        )
    except ValueError as exc:
        assert "converter hash differs" in str(exc)
    else:
        raise AssertionError("unpinned converter was accepted")
