from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nanoquant.application.calibration import (
    CausalOnlineCalibrationState,
    CausalOnlineLayerSnapshot,
    OnlineAccumulatorSnapshot,
    materialize_causal_online_state,
)
from nanoquant.infrastructure.calibration_checkpoint import save_causal_calibration_state
from tools.compare_fisher_state import _reference_key, _tensor_metrics, compare_fisher_state


def test_reference_key_maps_resident_paths_to_retained_names() -> None:
    path = "block.3.mlp.gate_proj"

    assert _reference_key(path, "input") == "i.model.layers.3.mlp.gate_proj"
    assert _reference_key(path, "output") == "o.model.layers.3.mlp.gate_proj"


def test_reference_key_rejects_unknown_layer_paths() -> None:
    with pytest.raises(ValueError, match="unsupported Fisher layer path"):
        _reference_key("model.layers.3.mlp.gate_proj", "input")


def test_tensor_metrics_reports_exact_relative_definitions() -> None:
    metrics = _tensor_metrics(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 2.0]))

    assert metrics["element_count"] == 2
    assert metrics["exact"] is False
    assert metrics["mean_absolute_error"] == pytest.approx(0.5)
    assert metrics["max_absolute_error"] == pytest.approx(1.0)
    assert metrics["mean_relative_error"] == pytest.approx(0.25)
    assert metrics["max_relative_error"] == pytest.approx(0.5)
    assert metrics["l1_relative_error"] == pytest.approx(1.0 / 3.0)


def test_tensor_metrics_rejects_shape_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        _tensor_metrics(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="finite tensors"):
        _tensor_metrics(torch.tensor([float("nan")]), torch.ones(1))


def test_comparison_tool_paths_are_importable_from_repository_root() -> None:
    assert Path(__file__).parents[2].joinpath("tools", "compare_fisher_state.py").is_file()


def test_compare_fisher_state_reports_exact_materialized_reference(tmp_path: Path) -> None:
    inputs = OnlineAccumulatorSnapshot(torch.tensor([2.0, 8.0]), torch.tensor(4.0), 2, 1.0, 1.0, 0.999)
    outputs = OnlineAccumulatorSnapshot(torch.tensor([18.0]), torch.tensor(5.0), 2, 1e6, 1e-6, 0.999)
    state = CausalOnlineCalibrationState(
        (CausalOnlineLayerSnapshot("block.0.proj", inputs, outputs),),
        processed_samples=2,
    )
    state_path = tmp_path / "state"
    save_causal_calibration_state(state_path, state)
    materialized = materialize_causal_online_state(state, shrinkage=0.6)[0]
    reference_path = tmp_path / "reference.safetensors"
    save_file(
        {
            "i.model.layers.0.proj": materialized.input_importance,
            "o.model.layers.0.proj": materialized.output_importance,
        },
        reference_path,
    )

    comparison = compare_fisher_state(state_path, reference_path, shrinkage=0.6)

    assert comparison["state_schema_version"] == 2
    assert comparison["state_algorithm_version"] == state.algorithm_version
    assert comparison["sample_count"] == 2
    assert comparison["input"]["exact_layer_count"] == 1
    assert comparison["output"]["exact_layer_count"] == 1
    assert comparison["unexpected_reference_keys"] == []
