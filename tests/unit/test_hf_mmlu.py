from types import SimpleNamespace

import torch
from torch import nn

from nanoquant.infrastructure.hf_mmlu import PreparedMmlu, evaluate_mmlu


class _AnswerModel(nn.Module):
    def forward(self, *, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
        assert use_cache is False and logits_to_keep == 1
        assert torch.equal(position_ids[:, -1], attention_mask.sum(dim=1) - 1)
        logits = torch.zeros((input_ids.shape[0], 1, 10))
        logits[torch.arange(input_ids.shape[0]), 0, input_ids[:, -1]] = 1
        return SimpleNamespace(logits=logits)


def test_mmlu_scores_one_answer_per_question_and_restores_order() -> None:
    prepared = PreparedMmlu(
        ((1, 1, 7), (6,), (1, 5)), ("math", "math", "history"), (2, 0, 0), (5, 6, 7, 8), {},
    )
    batches = []
    result = evaluate_mmlu(_AnswerModel(), prepared, device="cpu", pad_token_id=0,
                           after_batch=lambda: batches.append(True))
    assert batches == [True]
    assert result["primary_value"] == 2 / 3
    assert [example["predicted"] for example in result["examples"]] == [2, 1, 0]
    assert result["subjects"] == {"history": {"count": 1, "correct": 1}, "math": {"count": 2, "correct": 1}}
