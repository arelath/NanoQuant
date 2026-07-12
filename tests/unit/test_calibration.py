import pytest
import torch
from torch import nn

from nanoquant.application.calibration import UnsupportedCalibrationMode, calibrate_block
from nanoquant.domain.calibration_math import activation_square_mean, robust_tau, shrink_importance


class CalibrationBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 4, bias=False)
        self.second = nn.Linear(4, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(value)))


def _runner(block: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return block(value)


def test_activation_math_known_values_and_shrinkage() -> None:
    value = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    assert robust_tau(value, percentile=0.5) == 5
    assert torch.equal(activation_square_mean(value), torch.tensor([4.5, 10.0]))
    assert torch.equal(shrink_importance(torch.tensor([1.0, 3.0]), 0.5), torch.tensor([1.5, 2.5]))


def test_online_forward_and_two_phase_calibration_are_typed_finite_and_remove_hooks() -> None:
    batches = (
        torch.randn(2, 3, generator=torch.Generator().manual_seed(1)),
        torch.randn(2, 3, generator=torch.Generator().manual_seed(2)),
    )
    for method in ("online_fisher", "two_phase_fisher", "forward_only"):
        block = CalibrationBlock()
        results = calibrate_block(block, batches, ("first", "second"), _runner, method=method, shrinkage=0.2)
        assert [result.path for result in results] == ["first", "second"]
        assert all(torch.isfinite(result.input_importance).all() for result in results)
        assert all(torch.isfinite(result.output_importance).all() for result in results)
        if method == "forward_only":
            assert all(
                torch.equal(result.output_importance, torch.ones_like(result.output_importance)) for result in results
            )
        assert all(not module._forward_hooks and not module._backward_hooks for module in block.modules())


def test_two_phase_calibration_is_equal_for_equivalent_batch_partitions() -> None:
    batch = torch.randn(4, 3, generator=torch.Generator().manual_seed(4))
    first = CalibrationBlock()
    second = CalibrationBlock()
    second.load_state_dict(first.state_dict())
    combined = calibrate_block(first, (batch,), ("first", "second"), _runner, method="two_phase_fisher")
    partitioned = calibrate_block(
        second, (batch[:2], batch[2:]), ("first", "second"), _runner, method="two_phase_fisher"
    )
    for left, right in zip(combined, partitioned, strict=True):
        assert torch.allclose(left.input_importance, right.input_importance, rtol=1e-5, atol=1e-6)


def test_two_phase_is_deterministic() -> None:
    batches = (torch.randn(3, 3, generator=torch.Generator().manual_seed(8)),)
    first = CalibrationBlock()
    second = CalibrationBlock()
    second.load_state_dict(first.state_dict())
    left = calibrate_block(first, batches, ("first", "second"), _runner, method="two_phase_fisher")
    right = calibrate_block(second, batches, ("first", "second"), _runner, method="two_phase_fisher")
    for lhs, rhs in zip(left, right, strict=True):
        assert torch.equal(lhs.input_importance, rhs.input_importance)
        assert torch.equal(lhs.output_importance, rhs.output_importance)


def test_dbf_is_explicitly_rejected_as_research_only() -> None:
    with pytest.raises(UnsupportedCalibrationMode, match="CAL004"):
        calibrate_block(CalibrationBlock(), (torch.ones(1, 3),), ("first",), _runner, method="dbf")
