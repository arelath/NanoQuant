from copy import deepcopy

import pytest
import torch
from torch import nn

from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.application.tuning import (
    TuningRequest,
    _release_cuda_cache_under_pressure,
    post_block_refit,
    tune_factorized,
    tune_non_factorized,
)


class Hybrid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2, bias=False)
        self.quant = TrainableFactorizedLinear(
            torch.tensor([[1.0], [-1.0]]),
            torch.tensor([[1.0, -1.0, 1.0]]),
            torch.ones(3),
            torch.ones(1),
            torch.ones(2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.quant(value)


def _forward(model: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return model(value)


class _DeviceProperties:
    total_memory = 100


@pytest.mark.parametrize(("reserved", "expected_calls"), [(79, 0), (80, 1)])
def test_cuda_cache_release_is_gated_on_reserved_memory_pressure(
    monkeypatch: pytest.MonkeyPatch, reserved: int, expected_calls: int
) -> None:
    calls = 0

    def empty_cache() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: reserved)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: _DeviceProperties())
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    _release_cuda_cache_under_pressure(torch.device("cuda"))

    assert calls == expected_calls


def test_nonfactorized_tuning_is_independent_and_restores_best_state() -> None:
    model = Hybrid()
    inputs = torch.randn(16, 3, generator=torch.Generator().manual_seed(1))
    quant_before = {name: value.detach().clone() for name, value in model.quant.named_parameters()}
    targets = model.quant(inputs).detach() + inputs[:, :2] * 0.5
    metrics = tune_non_factorized(model, TuningRequest(inputs, targets, 20, 4, 0.05), _forward)
    assert metrics.best.loss < metrics.before.loss
    assert metrics.final.loss == metrics.best.loss
    assert all(torch.equal(value, quant_before[name]) for name, value in model.quant.named_parameters())


def test_factorized_tuning_changes_only_selected_module_and_restores_best() -> None:
    model = Hybrid()
    inputs = torch.randn(16, 3, generator=torch.Generator().manual_seed(2))
    base_before = model.base.weight.detach().clone()
    targets = model.base(inputs).detach() + 0.25 * torch.randn(16, 2, generator=torch.Generator().manual_seed(3))
    metrics = tune_factorized(model, "quant", TuningRequest(inputs, targets, 15, 4, 0.02), _forward)
    assert metrics.final.loss == metrics.best.loss <= metrics.before.loss
    assert torch.equal(model.base.weight, base_before)


def test_post_block_refit_updates_scales_without_latent_changes() -> None:
    model = Hybrid()
    inputs = torch.randn(12, 3, generator=torch.Generator().manual_seed(4))
    latent_before = (model.quant.left_latent.detach().clone(), model.quant.right_latent.detach().clone())
    targets = torch.zeros(12, 2)
    metrics = post_block_refit(model, TuningRequest(inputs, targets, 10, 3, 0.02), _forward)
    assert metrics.final.loss <= metrics.before.loss
    assert torch.equal(model.quant.left_latent, latent_before[0])
    assert torch.equal(model.quant.right_latent, latent_before[1])


def test_microbatch_accumulation_preserves_optimizer_batch_update() -> None:
    full_batch_model = Hybrid()
    microbatch_model = deepcopy(full_batch_model)
    inputs = torch.randn(8, 3, generator=torch.Generator().manual_seed(5))
    targets = torch.randn(8, 2, generator=torch.Generator().manual_seed(6))
    full = TuningRequest(inputs, targets, 2, 4, 0.01, seed=7)
    micro = TuningRequest(inputs, targets, 2, 4, 0.01, seed=7, microbatch_size=2)

    full_metrics = tune_factorized(full_batch_model, "quant", full, _forward)
    micro_metrics = tune_factorized(microbatch_model, "quant", micro, _forward)

    assert micro_metrics.final.loss == pytest.approx(full_metrics.final.loss, rel=1e-6, abs=1e-7)
    for full_parameter, micro_parameter in zip(
        full_batch_model.quant.parameters(), microbatch_model.quant.parameters(), strict=True
    ):
        assert torch.allclose(full_parameter, micro_parameter, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned tuning staging requires a GPU")
def test_pinned_shuffle_staging_matches_pageable_tuning_bitwise() -> None:
    pageable_model = Hybrid().cuda().to(torch.bfloat16)
    pinned_model = deepcopy(pageable_model)
    inputs = torch.randn(8, 3, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(17))
    targets = torch.randn(8, 2, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(18))
    request = TuningRequest(inputs, targets, 2, 4, 0.01, seed=19, microbatch_size=2)
    pinned_request = TuningRequest(
        inputs.pin_memory(),
        targets.pin_memory(),
        request.epochs,
        request.batch_size,
        request.learning_rate,
        seed=request.seed,
        microbatch_size=request.microbatch_size,
    )

    pageable_metrics = tune_factorized(pageable_model, "quant", request, _forward)
    pinned_metrics = tune_factorized(pinned_model, "quant", pinned_request, _forward)

    assert pinned_metrics == pageable_metrics
    for pageable_parameter, pinned_parameter in zip(
        pageable_model.parameters(), pinned_model.parameters(), strict=True
    ):
        assert torch.equal(pageable_parameter, pinned_parameter)
    del pageable_model, pinned_model
    torch.cuda.empty_cache()
