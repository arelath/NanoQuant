from pathlib import Path

import pytest
import torch

from tools.probe_topk_tail_mass import _means, _parse_arm, _tail_mass_sums


def test_tail_mass_metric_detects_error_hidden_by_conditional_topk() -> None:
    teacher = torch.tensor([[[3.0, 2.0, 0.0, -1.0]]])
    # The selected top-two logit difference is unchanged, but both selected
    # logits lose mass to the unselected vocabulary.
    student = torch.tensor([[[1.0, 0.0, 0.0, -1.0]]])
    result = _means(
        _tail_mass_sums(
            teacher,
            student,
            torch.tensor([[0]]),
            top_k=2,
        )
    )

    assert result["conditional_topk_kl"] == pytest.approx(0.0, abs=1e-7)
    assert float(result["topk_plus_tail_kl"]) > 0
    assert float(result["full_kl"]) > 0
    assert float(result["student_teacher_topk_mass"]) < float(result["teacher_topk_mass"])


def test_tail_mass_arm_parser_supports_pre_post_and_overlay() -> None:
    assert _parse_arm("pre=prekd") == ("pre", "prekd", None)
    assert _parse_arm("post=postkd") == ("post", "postkd", None)
    assert _parse_arm("fixed=postkd:evidence/fixed") == (
        "fixed",
        "postkd",
        Path("evidence/fixed"),
    )
    with pytest.raises(Exception, match="arm must use"):
        _parse_arm("broken=prekd:overlay")
