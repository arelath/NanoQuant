from pathlib import Path

from nanoquant.application.kl_budget import KlSequenceResult
from tools.probe_factorized_component_overlays_kl import _arm_result, _parse_arm


def test_arm_result_uses_token_weighted_sequence_metrics() -> None:
    result = _arm_result(
        "candidate",
        (
            KlSequenceResult(2.0, 0.5, 2),
            KlSequenceResult(4.0, 1.0, 1),
        ),
    )
    assert result.negative_log_likelihood == 8.0 / 3.0
    assert result.kl_nats_per_token == 2.0 / 3.0
    assert result.token_count == 3


def test_component_arm_can_name_the_unmodified_global_tuning_baseline() -> None:
    assert _parse_arm("postkd") == ("postkd", None)
    assert _parse_arm("folded=overlay") == ("folded", Path("overlay"))
