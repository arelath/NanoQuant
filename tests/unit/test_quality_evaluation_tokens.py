from __future__ import annotations

import sys
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
