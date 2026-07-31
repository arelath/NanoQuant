from nanoquant.application.kl_budget import KlSequenceResult
from tools.probe_factorized_component_overlays_kl import _arm_result


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
