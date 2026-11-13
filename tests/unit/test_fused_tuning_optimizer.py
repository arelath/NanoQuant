import copy

import pytest
import torch
from torch import nn

from nanoquant.application.fused_tuning_optimizer import fp32_scale_masters, restore_fused_state
from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.application.parity_adamw import ParityAdamWState
from nanoquant.application.tuning import TuningRequest, tune_factorized
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.tuning_checkpoint import (
    TuningCheckpointIdentity,
    active_tuning_checkpoint,
    save_tuning_checkpoint,
)


def _model():
    torch.manual_seed(12)
    return nn.Sequential(TrainableFactorizedLinear(
        torch.randn(3, 2), torch.randn(2, 4), torch.ones(4), torch.ones(2), torch.ones(3),
        residual_stages=(TrainableFactorizedLinear(
            torch.randn(3, 2), torch.randn(2, 4), torch.ones(4), torch.ones(2), torch.ones(3),
        ),),
    )).bfloat16()


def test_fused_tuning_resumes_exactly_with_fp32_scales_and_retains_bf16_storage(tmp_path, device="cpu"):
    original = _model()
    original.to(device)
    inputs = torch.randn(8, 2, 4).bfloat16()
    targets = torch.randn(8, 2, 3).bfloat16()
    request = TuningRequest(
        inputs, targets, 3, 2, 1e-3, seed=4, restore_best_state=False, epoch_loss_mode="legacy_training",
        optimizer_backend="fused_adamw_scales_fp32",
    )
    uninterrupted, interrupted, resumed = (copy.deepcopy(original) for _ in range(3))
    forward = lambda module, values: module(values)  # noqa: E731
    expected = tune_factorized(uninterrupted, "0", request, forward)
    saved = []
    identity = TuningCheckpointIdentity("fused", "model", "plan", 0, "0", "factorized")

    def checkpoint(state):
        saved.append(state)
        save_tuning_checkpoint(tmp_path, state, identity)
        raise InterruptedError("epoch boundary")

    with pytest.raises(InterruptedError, match="epoch boundary"):
        tune_factorized(interrupted, "0", request, forward, checkpoint_sink=checkpoint)
    loaded = active_tuning_checkpoint(tmp_path, identity)
    assert loaded is not None
    state = loaded.state
    assert state.completed_epochs == 1
    assert state.steps_completed == 4
    assert all(item.kahan_compensation is None for item in state.optimizer_states)
    assert all(value.dtype == torch.float32 for name, value in state.parameter_values if "scale_" in name)
    actual = tune_factorized(resumed, "0", request, forward, resume=state)
    assert actual.final.loss == expected.final.loss
    for actual_parameter, expected_parameter in zip(resumed.parameters(), uninterrupted.parameters(), strict=True):
        assert actual_parameter.dtype == torch.bfloat16
        assert torch.equal(actual_parameter, expected_parameter)
    assert all(value.dtype == torch.bfloat16 for value in interrupted.parameters())


def test_fused_optimizer_rejects_parity_state():
    parameter = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], fused=True)
    state = ParityAdamWState("p", torch.tensor(1), torch.zeros(2), torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="parity optimizer state"):
        restore_fused_state(optimizer, [("p", parameter)], [state], 1)


@pytest.mark.cuda
def test_fused_tuning_cuda_checkpoint_resume(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with acquire_device_lease("cuda"):
        test_fused_tuning_resumes_exactly_with_fp32_scales_and_retains_bf16_storage(tmp_path, device="cuda")


def test_scale_precision_restored_on_failure_without_replacing_parameters():
    model = _model()
    before = dict(model.named_parameters())
    with pytest.raises(RuntimeError):
        with fp32_scale_masters(model):
            assert all(value.dtype == torch.float32 for name, value in model.named_parameters() if "scale_" in name)
            raise RuntimeError("failed")
    assert all(value.dtype == torch.bfloat16 for value in model.parameters())
    assert all(value is before[name] for name, value in model.named_parameters())
