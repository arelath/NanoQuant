from nanoquant.application.loss_snapshots import BlockLossRecorder, normalized_activation_error
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import BlockLossMetrics


def test_normalized_activation_error_matches_weighted_reconstruction_convention() -> None:
    assert normalized_activation_error(2.0, 8.0) == 0.25
    assert normalized_activation_error(2.0, 0.0) is None


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
