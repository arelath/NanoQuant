from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import nanoquant.infrastructure.mmproj_export as mmproj_export
from nanoquant.infrastructure.io_utils import hash_file as real_hash_file
from nanoquant.infrastructure.mmproj_export import (
    MMPROJ_CONVERTER_SHA256,
    export_mmproj_bfloat16,
    source_has_vision_stack,
)


def _snapshot(tmp_path: Path, *, vision: bool) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True)
    config = {"architectures": ["Gemma3ForConditionalGeneration"]}
    if vision:
        config["vision_config"] = {"hidden_size": 16, "num_hidden_layers": 1}
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


def test_source_vision_stack_detection_is_config_driven(tmp_path: Path) -> None:
    assert source_has_vision_stack(_snapshot(tmp_path / "vision", vision=True))
    assert not source_has_vision_stack(_snapshot(tmp_path / "text", vision=False))


def test_mmproj_export_is_bfloat16_validated_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(tmp_path, vision=True)
    reference = tmp_path / "llama.cpp"
    reference.mkdir()
    (reference / "gguf-py").mkdir()
    converter = reference / "convert_hf_to_gguf.py"
    converter.write_text("# fixture\n", encoding="utf-8")
    output = tmp_path / "output" / "mmproj-BF16.gguf"

    def pinned_hash(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == converter.resolve():
            return MMPROJ_CONVERTER_SHA256
        return real_hash_file(resolved)

    commands: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        command = tuple(command)
        commands.append(command)
        if command[1] == str(converter):
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF-mmproj")
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "general_type": "mmproj",
                    "file_type": 32,
                    "tensor_count": 7,
                    "tensor_types": ["bf16", "f32"],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(mmproj_export, "hash_canonical_text_file", pinned_hash)
    monkeypatch.setattr(mmproj_export.subprocess, "run", fake_run)

    first = export_mmproj_bfloat16(snapshot, output, reference)
    second = export_mmproj_bfloat16(snapshot, output, reference)

    assert not first.reused
    assert second.reused
    assert first.output == output.resolve()
    assert first.tensor_count == 7
    assert first.tensor_types == ("bf16", "f32")
    assert [command for command in commands if command[1] == str(converter)] == [commands[0]]
    assert commands[0][commands[0].index("--outtype") + 1] == "bf16"
    receipt = json.loads(output.with_suffix(".gguf.export.json").read_text(encoding="utf-8"))
    assert receipt["file_type"] == "mostly_bfloat16"
    assert receipt["output_sha256"] == real_hash_file(output)


def test_mmproj_export_rejects_text_only_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not declare a vision stack"):
        export_mmproj_bfloat16(
            _snapshot(tmp_path, vision=False),
            tmp_path / "mmproj-BF16.gguf",
            tmp_path / "llama.cpp",
        )
