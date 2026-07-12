from pathlib import Path

import torch

from nanoquant.application.calibration import (
    CausalOnlineCalibrationState,
    CausalOnlineLayerSnapshot,
    OnlineAccumulatorSnapshot,
)
from nanoquant.infrastructure.calibration_checkpoint import (
    load_causal_calibration_state,
    save_causal_calibration_state,
)


def test_causal_calibration_checkpoint_round_trips(tmp_path: Path) -> None:
    inputs = OnlineAccumulatorSnapshot(torch.tensor([1.0, 2.0]), torch.tensor(3.0), 4, 1.0, 1.0, 0.999)
    outputs = OnlineAccumulatorSnapshot(torch.tensor([5.0]), torch.tensor(6.0), 4, 1e6, 1e-6, 0.999)
    state = CausalOnlineCalibrationState((CausalOnlineLayerSnapshot("block.0.proj", inputs, outputs),), 4)
    save_causal_calibration_state(tmp_path, state)

    loaded = load_causal_calibration_state(tmp_path)
    assert loaded.sample_count == 4
    assert loaded.layers[0].path == state.layers[0].path
    assert torch.equal(loaded.layers[0].inputs.total, inputs.total)
    assert torch.equal(loaded.layers[0].inputs.global_max, inputs.global_max)
    assert torch.equal(loaded.layers[0].outputs.total, outputs.total)
    assert loaded.layers[0].outputs.batch_count == outputs.batch_count
    assert loaded.layers[0].outputs.pre_scale == outputs.pre_scale
