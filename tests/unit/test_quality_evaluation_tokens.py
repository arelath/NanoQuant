from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

from nanoquant import quality_evaluation


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
    datasets.DownloadConfig = lambda **_kwargs: object()  # type: ignore[attr-defined]
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
        local_files_only=True,
    )

    assert calls == [{"return_tensors": "pt", "truncation": True, "max_length": 6}]
    assert torch.equal(tokens, torch.tensor([[2, 10, 11, 12], [2, 13, 14, 15]]))
    assert fingerprint == "fixture-fingerprint"
    assert bos_token_id == 2
