import random
from types import SimpleNamespace

import torch

from nanoquant.infrastructure.hf_calibration_dataset import _pack_chat_records, _slice_wikitext


class Tokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
        del messages, kwargs
        return list(range(2, 14))

    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.arange(max(40, len(text))).reshape(1, -1))


def test_chat_packing_and_wikitext_slicing_are_exact_length_and_deterministic() -> None:
    records = ({"messages": [{"role": "user", "content": str(index)}]} for index in range(20))
    chat = _pack_chat_records(records, Tokenizer(), count=3, sequence_length=10)
    first = _slice_wikitext("x" * 100, Tokenizer(), 4, 8, random.Random(1))
    second = _slice_wikitext("x" * 100, Tokenizer(), 4, 8, random.Random(1))

    assert len(chat) == 3 and all(len(row) == 10 for row in chat)
    assert all(len(row) == 8 for row in first)
    assert first == second
