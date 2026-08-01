import torch

from nanoquant.application.distillation import TopKDistillationConfig, TopKTeacherBatch
from tools.probe_topk_tail_distillation import _normalizer_protocol, _parser, _target_normalizers


def test_objective_parser_supports_a_protocol_matched_conditional_control() -> None:
    args = _parser().parse_args(
        [
            "--run-output",
            "run",
            "--snapshot",
            "snapshot",
            "--output-directory",
            "output",
            "--objective",
            "conditional_topk",
        ]
    )

    assert args.objective == "conditional_topk"


def test_teacher_normalizer_protocol_is_bound_to_tokens_and_temperature() -> None:
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    base = _normalizer_protocol(tokens, TopKDistillationConfig(), "revision")
    changed_tokens = _normalizer_protocol(tokens + 1, TopKDistillationConfig(), "revision")
    changed_temperature = _normalizer_protocol(
        tokens,
        TopKDistillationConfig(temperature=2.0),
        "revision",
    )

    assert base["sample_count"] == 2
    assert base["sequence_length"] == 3
    assert base["token_hash"] != changed_tokens["token_hash"]
    assert base["temperature"] != changed_temperature["temperature"]


def test_target_normalizers_follow_cached_batch_sample_and_token_indices() -> None:
    target = TopKTeacherBatch(
        sample_indices=(2, 0),
        token_indices=torch.tensor([1, 6]),
        top_values=torch.zeros((2, 2)),
        top_indices=torch.zeros((2, 2), dtype=torch.int32),
    )
    cached = ((torch.tensor([21.0, 2.0]),),)

    selected = _target_normalizers(cached, 0, 0, target, device="cpu")

    assert torch.equal(selected, torch.tensor([21.0, 2.0]))
