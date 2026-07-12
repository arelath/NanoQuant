from dataclasses import replace

import pytest

from nanoquant.application.dataset import DatasetRecord, prepare_dataset
from nanoquant.config.schema import DatasetConfig, DatasetSourceConfig


class Provider:
    def records(self, source: DatasetSourceConfig) -> tuple[DatasetRecord, ...]:
        if source.name == "chat":
            return tuple(
                DatasetRecord(str(index), messages=({"role": "user", "content": f"question {index}"},))
                for index in range(10)
            )
        return tuple(DatasetRecord(str(index), text=f"document {index}") for index in range(10))


class Tokenizer:
    name_or_path = "fixture/tokenizer"
    bos_token_id = 2
    eos_token_id = [1, 3]
    pad_token_id = 0
    chat_template = "{{ role }}: {{ content }}"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = [ord(character) % 17 + 4 for character in text]
        return ([self.bos_token_id] if add_special_tokens else []) + values + ([1] if add_special_tokens else [])

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert not tokenize and not add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _config() -> DatasetConfig:
    return DatasetConfig(
        sources=(
            DatasetSourceConfig("text", revision="text-rev", weight=0.75),
            DatasetSourceConfig("chat", revision="chat-rev", weight=0.25),
        ),
        selection_seed=9,
    )


def test_dataset_mixture_selection_tokenization_and_fingerprint_are_deterministic() -> None:
    first = prepare_dataset(
        _config(), 8, 12, Provider(), Tokenizer(), tokenizer_source="fixture/tokenizer", tokenizer_revision="tok-rev"
    )
    second = prepare_dataset(
        _config(), 8, 12, Provider(), Tokenizer(), tokenizer_source="fixture/tokenizer", tokenizer_revision="tok-rev"
    )
    assert first.fingerprint == second.fingerprint
    assert first.selected_sample_ids == second.selected_sample_ids
    assert sum(sample.startswith("text:") for sample in first.selected_sample_ids) == 6
    assert sum(sample.startswith("chat:") for sample in first.selected_sample_ids) == 2
    assert first.input_ids.shape == (8, 12) and first.attention_mask.shape == (8, 12)
    assert first.valid_token_count == int(first.attention_mask.sum())
    assert first.tokenizer.revision == "tok-rev"
    assert first.tokenizer.chat_template_hash is not None
    assert first.tokenizer.eos_token_ids == (1, 3)


def test_dataset_seed_and_selected_content_change_identity() -> None:
    first = prepare_dataset(
        _config(), 4, 8, Provider(), Tokenizer(), tokenizer_source="fixture/tokenizer", tokenizer_revision="tok-rev"
    )
    changed = prepare_dataset(
        replace(_config(), selection_seed=10),
        4,
        8,
        Provider(),
        Tokenizer(),
        tokenizer_source="fixture/tokenizer",
        tokenizer_revision="tok-rev",
    )
    assert first.fingerprint != changed.fingerprint


def test_dataset_requires_pinned_revisions_unique_ids_and_capacity() -> None:
    unpinned = DatasetConfig(sources=(DatasetSourceConfig("text"),))
    with pytest.raises(ValueError, match="not pinned"):
        prepare_dataset(unpinned, 1, 8, Provider(), Tokenizer(), tokenizer_source="x", tokenizer_revision="rev")
    with pytest.raises(ValueError, match="requires"):
        prepare_dataset(_config(), 30, 8, Provider(), Tokenizer(), tokenizer_source="x", tokenizer_revision="rev")
