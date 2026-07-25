from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from nanoquant import quality_evaluation


def test_quality_padding_falls_back_to_eos_for_tokenizers_without_pad_tokens() -> None:
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=128009)

    assert quality_evaluation._quality_pad_token_id(tokenizer) == 128009


def test_quality_padding_rejects_tokenizers_without_pad_or_eos_tokens() -> None:
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=None)

    with pytest.raises(ValueError, match="neither a valid pad nor EOS"):
        quality_evaluation._quality_pad_token_id(tokenizer)


def test_wikitext_tokenization_is_bounded_to_the_evaluated_prefix(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeDataset(dict[str, list[str]]):
        _fingerprint = "fixture-fingerprint"

    class FakeTokenizer:
        bos_token_id = 2

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            assert text == "first\n\nsecond"
            calls.append(dict(kwargs))
            return SimpleNamespace(input_ids=torch.tensor([[10, 11, 12, 13, 14, 15]]))

    datasets = ModuleType("datasets")
    datasets.Dataset = object  # type: ignore[attr-defined]
    datasets.DownloadConfig = lambda **_kwargs: object()  # type: ignore[attr-defined]
    datasets.config = SimpleNamespace(HF_DATASETS_CACHE=tmp_path)  # type: ignore[attr-defined]
    datasets.load_dataset = lambda *_args, **_kwargs: FakeDataset(text=["first", "second"])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(
        quality_evaluation.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )

    tokens, fingerprint, bos_token_id = quality_evaluation._wikitext_tokens(
        tmp_path,
        samples=2,
        sequence_length=4,
        local_files_only=False,
    )

    assert calls == [{"return_tensors": "pt", "truncation": True, "max_length": 6}]
    assert torch.equal(tokens, torch.tensor([[2, 10, 11, 12], [2, 13, 14, 15]]))
    assert fingerprint == "fixture-fingerprint"
    assert bos_token_id == 2


def test_wikitext_uses_first_raw_token_as_context_without_bos(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeDataset(dict[str, list[str]]):
        _fingerprint = "qwen-fixture"

    class FakeTokenizer:
        bos_token_id = None

        def __call__(self, _text: str, **kwargs: object) -> SimpleNamespace:
            calls.append(dict(kwargs))
            return SimpleNamespace(
                input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
            )

    datasets = ModuleType("datasets")
    datasets.Dataset = object  # type: ignore[attr-defined]
    datasets.DownloadConfig = lambda **_kwargs: object()  # type: ignore[attr-defined]
    datasets.config = SimpleNamespace(HF_DATASETS_CACHE=tmp_path)  # type: ignore[attr-defined]
    datasets.load_dataset = lambda *_args, **_kwargs: FakeDataset(text=["qwen"])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(
        quality_evaluation.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )

    tokens, fingerprint, bos_token_id = quality_evaluation._wikitext_tokens(
        tmp_path,
        samples=2,
        sequence_length=4,
        local_files_only=False,
    )

    assert calls == [{"return_tensors": "pt", "truncation": True, "max_length": 8}]
    assert torch.equal(tokens, torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17]]))
    assert fingerprint == "qwen-fixture"
    assert bos_token_id is None


def test_base_only_quality_does_not_load_a_pytorch_candidate(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    request = quality_evaluation.QualityEvaluationRequest(
        tmp_path / "snapshot",
        "fixture/model",
        "revision",
        tmp_path / "run",
        device="cpu",
        wikitext_samples=1,
        wikitext_sequence_length=2,
        task_names=("piqa",),
    )
    prepared = quality_evaluation.PreparedQualityInputs(
        torch.tensor(((1, 2),), dtype=torch.long),
        "fixture-fingerprint",
        1,
        0,
        "sha256:" + "a" * 64,
        (),
    )
    source_model = SimpleNamespace()
    source_model.to = lambda _device: source_model
    monkeypatch.setattr(
        quality_evaluation,
        "acquire_device_lease",
        lambda _device: nullcontext(),
    )
    monkeypatch.setattr(
        quality_evaluation.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen"),
    )
    monkeypatch.setattr(
        quality_evaluation,
        "_checkpoint_dtype",
        lambda _snapshot: torch.float32,
    )
    monkeypatch.setattr(
        quality_evaluation,
        "load_causal_language_model",
        lambda *_args, **_kwargs: source_model,
    )
    monkeypatch.setattr(
        quality_evaluation,
        "_evaluate_model",
        lambda *_args, **_kwargs: {
            "label": "base",
            "wikitext": {"perplexity": 2.0},
            "tasks": [],
            "elapsed_seconds": 1.0,
            "peak_device_bytes": 0,
            "peak_host_bytes": 0,
        },
    )
    monkeypatch.setattr(quality_evaluation, "_release_device_memory", lambda: None)
    monkeypatch.setattr(
        quality_evaluation,
        "load_frozen_run",
        lambda *_args, **_kwargs: pytest.fail("frozen candidate must not be loaded"),
    )
    monkeypatch.setattr(
        quality_evaluation,
        "load_packed_model",
        lambda *_args, **_kwargs: pytest.fail("packed candidate must not be loaded"),
    )

    result = quality_evaluation.execute_quality_evaluation(
        request,
        prepared=prepared,
        evaluate_candidate=False,
    )

    assert result["passed"] is True
    assert result["candidate"] is None
    assert tuple(result["results"]) == ("base",)
    assert result["comparison"] is None


def test_reasoning_comparison_rejects_hidden_thinking_regression() -> None:
    base = {
        "wikitext": {"perplexity": 10.0},
        "tasks": [],
        "reasoning": [
            {"mode": "thinking", "mean_negative_log_likelihood": 2.0},
            {"mode": "non_thinking", "mean_negative_log_likelihood": 1.0},
        ],
    }
    candidate = {
        "wikitext": {"perplexity": 10.0},
        "tasks": [],
        "reasoning": [
            {"mode": "thinking", "mean_negative_log_likelihood": 2.6},
            {"mode": "non_thinking", "mean_negative_log_likelihood": 1.05},
        ],
    }

    comparison = quality_evaluation.compare_quality_results(base, candidate, 1.10)

    assert comparison["reasoning_cross_mode"]["passed"] is False
    assert {item["mode"] for item in comparison["reasoning"]} == {
        "thinking",
        "non_thinking",
    }
