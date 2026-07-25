from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest
import torch

import nanoquant.llamacpp_quality as llamacpp_quality
from nanoquant.llamacpp_quality import (
    LlamaCppQualityRequest,
    execute_llamacpp_quality_evaluation,
)
from nanoquant.quality_evaluation import PreparedQualityInputs, QualityEvaluationRequest


def test_llamacpp_quality_uses_first_raw_token_as_context_without_bos(
    tmp_path: Path,
) -> None:
    prepared = PreparedQualityInputs(
        torch.tensor(((10, 11, 12, 13), (14, 15, 16, 17)), dtype=torch.long),
        "qwen-wikitext-fingerprint",
        None,
        0,
        "sha256:" + "a" * 64,
        (),
    )
    quality_request = QualityEvaluationRequest(
        tmp_path,
        "Qwen/Qwen3-0.6B",
        "revision",
        tmp_path / "run",
        device="cpu",
        task_names=("piqa",),
    )

    sequences, task_candidates, task_truncated = llamacpp_quality._quality_sequences(
        quality_request,
        prepared,
    )

    assert sequences == (
        llamacpp_quality._Sequence((10, 11, 12, 13), 1),
        llamacpp_quality._Sequence((14, 15, 16, 17), 1),
    )
    assert task_candidates == ()
    assert task_truncated == ()


def test_llamacpp_quality_is_protocol_matched_and_identity_resumable(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "llama.cpp"
    (root / ".git").mkdir(parents=True)
    runner = root / "build" / "nanoquant-quality" / "nanoquant-llamacpp-quality"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(b"runner")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF fixture")
    output = tmp_path / "gguf-quality.json"
    prepared = PreparedQualityInputs(
        torch.tensor(((1, 2, 3), (1, 4, 5)), dtype=torch.long),
        "wikitext-fingerprint",
        1,
        0,
        "sha256:" + "a" * 64,
        (),
    )
    quality_request = QualityEvaluationRequest(
        tmp_path,
        "fixture/model",
        "revision",
        tmp_path / "run",
        device="cpu",
        task_names=("piqa",),
    )
    base_result = {
        "label": "base",
        "wikitext": {
            "total_negative_log_likelihood": 2.0,
            "mean_negative_log_likelihood": 0.5,
            "perplexity": math.exp(0.5),
            "token_count": 4,
            "window_count": 2,
            "sample_count": 2,
        },
        "tasks": [],
    }
    monkeypatch.setattr(
        llamacpp_quality,
        "_git_capture",
        lambda _root: {
            "repository": "https://github.com/arelath/llama.cpp.git",
            "commit": "1" * 40,
            "branch": "nanoquants",
            "dirty": False,
        },
    )
    monkeypatch.setattr(
        llamacpp_quality,
        "_runtime_files",
        lambda _root, _runner: (runner,),
    )
    calls = []

    def run(_request, _runner, input_path, output_path):  # type: ignore[no-untyped-def]
        calls.append(input_path.read_bytes())
        with output_path.open("wb") as destination:
            destination.write(b"NQQO0001")
            destination.write(struct.pack("<I", 2))
            destination.write(struct.pack("<dI", 1.0, 2))
            destination.write(struct.pack("<dI", 3.0, 2))
        return llamacpp_quality._RunnerResourceMetrics(
            400,
            50,
            800,
            "fixture-child-process",
        )

    monkeypatch.setattr(llamacpp_quality, "_run", run)
    request = LlamaCppQualityRequest(
        gguf,
        output,
        root,
        device="cpu",
        runner=runner,
        gpu_layers=0,
        parallel=2,
    )

    result = execute_llamacpp_quality_evaluation(
        request,
        quality_request,
        prepared,
        base_result,
        {"wikitext_token_hash": "sha256:tokens", "task_names": ()},
    )

    assert result["passed"] is True
    assert result["results"]["gguf"]["wikitext"]["token_count"] == 4
    assert result["results"]["gguf"]["wikitext"]["perplexity"] == math.e
    assert result["results"]["gguf"]["peak_device_bytes"] == 400
    assert result["results"]["gguf"]["peak_device_shared_bytes"] == 50
    assert result["results"]["gguf"]["peak_host_bytes"] == 800
    assert result["results"]["gguf"]["memory_measurement"] == "fixture-child-process"
    assert result["comparison"]["wikitext"]["ratio"] == pytest.approx(math.exp(0.5))
    assert result["identity"]["llama_cpp_commit"] == "1" * 40
    assert result["runtime"]["git"]["repository"] == (
        "https://github.com/arelath/llama.cpp.git"
    )
    assert len(calls) == 1
    assert calls[0].startswith(b"NQQL0001")

    reused = execute_llamacpp_quality_evaluation(
        request,
        quality_request,
        prepared,
        base_result,
        {"wikitext_token_hash": "sha256:tokens", "task_names": ()},
    )
    assert reused["reused"] is True
    assert len(calls) == 1


def test_llamacpp_quality_runner_source_uses_target_only_logits() -> None:
    source = Path("tools/llamacpp/quality_runner/main.cpp").read_text(encoding="utf-8")

    assert "batch.logits[batch_index] = scored ? 1 : 0;" in source
    assert "sequence.tokens[position + 1]" in source
    assert "llama_memory_clear" in source


def test_llamacpp_quality_markdown_reports_packed_runtime_memory() -> None:
    payload = {
        "gguf": {
            "path": "model.gguf",
            "bytes": 250,
            "sha256": "a" * 64,
        },
        "runtime": {"git": {"commit": "b" * 40}},
        "wall_seconds": 4.0,
        "results": {
            "gguf": {
                "peak_device_bytes": 400,
                "peak_device_shared_bytes": 50,
                "peak_host_bytes": 800,
                "memory_measurement": "fixture-child-process",
            }
        },
        "comparison": {
            "wikitext": {
                "base_perplexity": 10.0,
                "frozen_perplexity": 12.0,
            },
            "tasks": [],
        },
    }

    rendered = llamacpp_quality.render_llamacpp_quality_markdown(payload)

    assert "Packed GGUF runtime resource" in rendered
    assert "| Dedicated GPU memory | 400 |" in rendered
    assert "| Shared GPU memory | 50 |" in rendered
    assert "| Host working set | 800 |" in rendered
