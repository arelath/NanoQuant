from __future__ import annotations

from pathlib import Path

import pytest

from nanoquant.application.task_evaluation import pinned_legacy_multiple_choice_tasks
from nanoquant.infrastructure import hf_task_evaluation
from nanoquant.infrastructure.hf_task_evaluation import (
    HFCausalPairTokenizer,
    hash_hf_tokenizer_snapshot,
    prepare_pinned_hf_multiple_choice_inputs,
)


class RecordingTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    is_fast = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.calls.append((text, add_special_tokens))
        prefix = [self.bos_token_id] if add_special_tokens else []
        return [*prefix, *(ord(character) for character in text)]


def test_causal_pair_tokenizer_matches_lm_eval_space_movement_and_bos_split() -> None:
    tokenizer = RecordingTokenizer()
    encode = HFCausalPairTokenizer(tokenizer, add_special_tokens=True)

    context, continuation = encode("Question:  ", "answer")

    assert tokenizer.calls == [("Question:  answer", True), ("Question:", True)]
    assert context == (2, *(ord(character) for character in "Question:"))
    assert continuation == tuple(ord(character) for character in "  answer")


def test_causal_pair_tokenizer_rejects_empty_inputs() -> None:
    encode = HFCausalPairTokenizer(RecordingTokenizer())

    with pytest.raises(ValueError, match="non-empty context"):
        encode("", "answer")
    with pytest.raises(ValueError, match="empty continuation"):
        encode("context", "")


def test_tokenizer_snapshot_hash_covers_every_behavior_file(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text('{"version": 1}', encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text('{"add_bos_token": true}', encoding="utf-8")
    first = hash_hf_tokenizer_snapshot(tmp_path)

    (tmp_path / "tokenizer_config.json").write_text('{"add_bos_token": false}', encoding="utf-8")
    second = hash_hf_tokenizer_snapshot(tmp_path)

    assert first.startswith("sha256:") and first != second
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="recognized"):
        hash_hf_tokenizer_snapshot(tmp_path / "empty")


def test_hf_preparation_pins_pair_behavior_in_cache_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    task = pinned_legacy_multiple_choice_tasks()[0]
    documents = ({"goal": "Choose?", "sol1": "first", "sol2": "second", "label": 0},)
    monkeypatch.setattr(
        hf_task_evaluation,
        "load_pinned_multiple_choice_documents",
        lambda *_args, **_kwargs: documents,
    )
    tokenizer = RecordingTokenizer()

    prepared = prepare_pinned_hf_multiple_choice_inputs(
        task,
        tokenizer,
        tokenizer_name="google/gemma-3-1b-it",
        tokenizer_revision="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        tokenizer_content_hash="sha256:" + "a" * 64,
        maximum_samples=1,
    )

    parameters = dict(prepared.cache_identity.tokenizer_parameters)
    assert parameters == {
        "add_special_tokens": True,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "is_fast": True,
        "pair_encoding": "lm-eval-causal-pair-v1",
        "tokenizer_class": "RecordingTokenizer",
    }
    assert prepared.examples[0].contexts[0][0] == 2
    assert prepared.examples[0].continuations[0][0] == ord(" ")
