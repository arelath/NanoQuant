import pytest

from nanoquant.application.loss_snapshots import BlockLossRecorder, normalized_activation_error
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import BlockLossMetrics


def test_normalized_activation_error_matches_weighted_reconstruction_convention() -> None:
    assert normalized_activation_error(2.0, 8.0) == 0.25
    assert normalized_activation_error(2.0, 0.0) is None


@pytest.mark.parametrize(
    "record",
    (
        lambda recorder: recorder.record_target_weighted_mean_square(float("nan")),
        lambda recorder: recorder.record_source_reference(float("inf")),
        lambda recorder: recorder.record_block_entry(float("nan")),
        lambda recorder: recorder.record_post_block_refit(float("-inf")),
        lambda recorder: recorder.record_final_frozen_pre_kd(float("nan")),
    ),
)
def test_block_loss_recorder_rejects_nonfinite_boundaries(record) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FloatingPointError, match="non-finite"):
        record(BlockLossRecorder())


def test_block_loss_metrics_decode_legacy_payload_without_normalized_fields() -> None:
    recorder = BlockLossRecorder()
    recorder.record_source_reference(0.0)
    recorder.record_block_entry(2.0)
    recorder.record_final_frozen_pre_kd(1.0)
    payload = to_dict(recorder.finalize())
    payload.pop("target_weighted_mean_square")
    payload.pop("block_entry_normalized_error")
    payload.pop("final_frozen_normalized_error")

    decoded = from_dict(BlockLossMetrics, payload)

    assert decoded.target_weighted_mean_square is None
    assert decoded.block_entry_normalized_error is None
    assert decoded.final_frozen_normalized_error is None
