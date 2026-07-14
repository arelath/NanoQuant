from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from torch import nn

import nanoquant.application.tuning as tuning_module
from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.application.tuning import (
    TuningRequest,
    _PinnedBatchStager,
    _release_cuda_cache_under_pressure,
    post_block_refit,
    tune_factorized,
    tune_non_factorized,
)
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.infrastructure.profiling import Profiler


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


class _FakeEvent:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def synchronize(self) -> None:
        self.calls.append(f"{self.name}.synchronize")

    def record(self, _stream: object) -> None:
        self.calls.append(f"{self.name}.record")


class _FakeHostBuffer:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def __getitem__(self, _item: object) -> "_FakeHostBuffer":
        return self


class _FakeDeviceBuffer(_FakeHostBuffer):
    def copy_(self, source: _FakeHostBuffer, *, non_blocking: bool) -> None:
        assert non_blocking
        self.calls.append(f"{self.name}.copy({source.name})")


class _FakeStream:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def wait_event(self, event: _FakeEvent) -> None:
        self.calls.append(f"copy.wait({event.name})")


class _FakeSource:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeTrainingStager:
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets
        self.consumed: list[int] = []
        self.closed = False

    def batches(self, indexes: tuple[torch.Tensor, ...]):  # type: ignore[no-untyped-def]
        for position, selected in enumerate(indexes):
            yield tuning_module._StagedTrainingBatch(
                self.inputs[selected],
                self.targets[selected],
                position % 2,
            )

    def mark_consumed(self, slot: int) -> None:
        self.consumed.append(slot)

    def close(self) -> None:
        self.closed = True


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


def test_pinned_batch_stager_reuses_fixed_device_slot_after_compute_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    stager = object.__new__(_PinnedBatchStager)
    stager.inputs = _FakeSource("input")  # type: ignore[assignment]
    stager.targets = _FakeSource("target")  # type: ignore[assignment]
    stager.input_buffers = (_FakeHostBuffer("input_host", calls),)  # type: ignore[assignment]
    stager.target_buffers = (_FakeHostBuffer("target_host", calls),)  # type: ignore[assignment]
    stager.device_input_buffers = (_FakeDeviceBuffer("input_device", calls),)  # type: ignore[assignment]
    stager.device_target_buffers = (_FakeDeviceBuffer("target_device", calls),)  # type: ignore[assignment]
    stager.device = torch.device("cuda")
    stager.copy_stream = _FakeStream(calls)  # type: ignore[assignment]
    stager.ready_events = (_FakeEvent("ready", calls),)  # type: ignore[assignment]
    stager.consumed_events = (_FakeEvent("consumed", calls),)  # type: ignore[assignment]
    stager.ready_recorded = [True]
    stager.consumed_recorded = [True]
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        torch,
        "index_select",
        lambda source, _dim, _indexes, *, out: calls.append(f"{source.name}.index_select") or out,
    )

    stager._schedule(torch.tensor([0, 1]), 0)

    assert calls == [
        "ready.synchronize",
        "input.index_select",
        "target.index_select",
        "copy.wait(consumed)",
        "input_device.copy(input_host)",
        "target_device.copy(target_host)",
        "ready.record",
    ]
    assert stager.ready_recorded == [True]
    assert stager.consumed_recorded == [False]


def test_tuning_marks_each_staged_batch_consumed_and_closes_stager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Hybrid()
    inputs = torch.randn(8, 3, generator=torch.Generator().manual_seed(40))
    targets = torch.randn(8, 2, generator=torch.Generator().manual_seed(41))
    stager = _FakeTrainingStager(inputs, targets)
    monkeypatch.setattr(tuning_module, "_pinned_batch_stager", lambda *_args: stager)

    tune_factorized(
        model,
        "quant",
        TuningRequest(inputs, targets, 1, 4, 0.01, seed=42, microbatch_size=2),
        _forward,
    )

    assert stager.consumed == [0, 1, 0, 1]
    assert stager.closed is True


def test_tuning_synchronizes_gradient_handoff_before_each_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Hybrid()
    inputs = torch.randn(8, 3, generator=torch.Generator().manual_seed(43))
    targets = torch.randn(8, 2, generator=torch.Generator().manual_seed(44))
    synchronized: list[torch.device] = []
    monkeypatch.setattr(
        tuning_module,
        "_synchronize_gradient_handoff",
        synchronized.append,
    )

    tune_factorized(
        model,
        "quant",
        TuningRequest(inputs, targets, 1, 4, 0.01, seed=45, microbatch_size=2),
        _forward,
    )

    assert synchronized == [torch.device("cpu"), torch.device("cpu")]


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


def test_factorized_tuning_can_keep_final_epoch_state(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Hybrid()
    inputs = torch.randn(8, 3, generator=torch.Generator().manual_seed(70))
    targets = torch.randn(8, 2, generator=torch.Generator().manual_seed(71))
    observed: list[torch.Tensor] = []
    losses = iter((10.0, 1.0, 5.0, 5.0))

    def evaluate(*_args: object, **_kwargs: object) -> float:
        observed.append(model.quant.left_latent.detach().clone())
        return next(losses)

    monkeypatch.setattr(tuning_module, "_evaluate_loss", evaluate)
    metrics = tune_factorized(
        model,
        "quant",
        TuningRequest(inputs, targets, 2, 4, 0.02, restore_best_state=False),
        _forward,
    )

    assert metrics.best.loss == 1.0
    assert metrics.final.loss == 5.0
    assert torch.equal(observed[-1], observed[2])
    assert not torch.equal(observed[-1], observed[1])


def test_tuning_epoch_observer_receives_full_evaluation_trajectory() -> None:
    model = Hybrid()
    inputs = torch.randn(8, 3, generator=torch.Generator().manual_seed(20))
    targets = torch.randn(8, 2, generator=torch.Generator().manual_seed(21))
    trajectory: list[tuple[int, float]] = []

    metrics = tune_factorized(
        model,
        "quant",
        TuningRequest(
            inputs,
            targets,
            3,
            4,
            0.02,
            epoch_observer=lambda epoch, loss: trajectory.append((epoch, loss)),
        ),
        _forward,
    )

    assert [epoch for epoch, _loss in trajectory] == [0, 1, 2, 3]
    assert trajectory[0][1] == metrics.before.loss


def test_factorized_tuning_epoch_resume_is_bitwise_equivalent() -> None:
    initial = Hybrid()
    control = deepcopy(initial)
    interrupted = deepcopy(initial)
    inputs = torch.randn(12, 3, generator=torch.Generator().manual_seed(50))
    targets = torch.randn(12, 2, generator=torch.Generator().manual_seed(51))
    request = TuningRequest(inputs, targets, 5, 4, 0.02, seed=52, microbatch_size=2)
    checkpoints = []

    control_metrics = tune_factorized(control, "quant", request, _forward)

    def checkpoint_sink(state):  # type: ignore[no-untyped-def]
        checkpoints.append(state)
        if state.completed_epochs == 2:
            raise InterruptedError("injected epoch interruption")

    with pytest.raises(InterruptedError, match="epoch interruption"):
        tune_factorized(
            interrupted,
            "quant",
            request,
            _forward,
            checkpoint_sink=checkpoint_sink,
        )

    assert checkpoints[-1].completed_epochs == 2
    restarted = deepcopy(initial)
    resumed_metrics = tune_factorized(
        restarted,
        "quant",
        request,
        _forward,
        resume=checkpoints[-1],
    )

    assert resumed_metrics == control_metrics
    for control_parameter, resumed_parameter in zip(
        control.parameters(), restarted.parameters(), strict=True
    ):
        assert torch.equal(resumed_parameter, control_parameter)


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


def test_micro_profiling_preserves_tuning_result_and_records_hot_loop_phases() -> None:
    control_model = Hybrid()
    profiled_model = deepcopy(control_model)
    inputs = torch.randn(8, 3, generator=torch.Generator().manual_seed(30))
    targets = torch.randn(8, 2, generator=torch.Generator().manual_seed(31))
    request = TuningRequest(inputs, targets, 2, 4, 0.01, seed=32, microbatch_size=2)

    control_metrics = tune_factorized(control_model, "quant", request, _forward)
    profiler = Profiler(
        ProfilingConfig(level=ProfilingLevel.MICRO, emit_span_events=False),
        run_id="tuning-micro",
    )
    profiled_metrics = tune_factorized(profiled_model, "quant", request, _forward, profiler)

    assert profiled_metrics == control_metrics
    for control_parameter, profiled_parameter in zip(
        control_model.parameters(), profiled_model.parameters(), strict=True
    ):
        assert torch.equal(profiled_parameter, control_parameter)
    payload = profiler.snapshot()
    phase_paths = {str(phase["path"]) for phase in payload["phases"]}  # type: ignore[index]
    assert {
        "initial_evaluation",
        "epoch/batch_stage",
        "epoch/forward",
        "epoch/loss",
        "epoch/backward",
        "epoch/optimizer_step",
        "epoch/epoch_evaluation/synchronize",
        "final_evaluation",
    } <= phase_paths
    counters = {str(counter["name"]): counter for counter in payload["counters"]}  # type: ignore[index]
    assert counters["tuning.tokens"]["total"] == 16
    assert counters["tuning.steps"]["total"] == 4
    assert counters["tuning.best_state_clones"]["total"] >= 1


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
