from __future__ import annotations

import json
from pathlib import Path

import torch

import nanoquant.infrastructure.gguf_export as gguf_export
from nanoquant.infrastructure.gguf_export import export_llamacpp_gguf
from nanoquant.infrastructure.io_utils import hash_file as real_hash_file
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
    reference = tmp_path / "llama.cpp"
    reference.mkdir()
    converter = reference / "convert_nanoquant_to_gguf.py"
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

    def pinned_hash(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == converter.resolve():
            return PACKED_REFERENCE_CONVERTER_SHA256
        return real_hash_file(resolved)

    monkeypatch.setattr(gguf_export, "hash_file", pinned_hash)
    output = tmp_path / "output" / "model.gguf"
    checkpoint = tmp_path / "output" / "checkpoint"

    first = export_llamacpp_gguf(packed, source, checkpoint, output, reference)
    second = export_llamacpp_gguf(packed, source, checkpoint, output, reference)

    assert output.read_bytes() == b"GGUF-fixture"
    assert not first.reused
    assert second.reused
    receipt = json.loads(output.with_suffix(".gguf.export.json").read_text(encoding="utf-8"))
    assert receipt["gguf_sha256"] == real_hash_file(output)
    assert receipt["converter_sha256"] == PACKED_REFERENCE_CONVERTER_SHA256


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
