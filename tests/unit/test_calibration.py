from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.application.calibration import (
    CAUSAL_CALIBRATION_ALGORITHM_VERSION,
    CausalOnlineCalibrationState,
    UnsupportedCalibrationMode,
    calibrate_block,
    calibrate_causal_model,
    causal_language_model_loss,
    materialize_causal_online_state,
    memory_efficient_causal_language_model_loss,
)
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.calibration_math import MeanAccumulator, activation_square_mean, robust_tau, shrink_importance
from nanoquant.infrastructure.profiling import Profiler


def test_mean_accumulator_weights_activation_rows_not_batches() -> None:
    accumulator = MeanAccumulator(2)
    accumulator.update(torch.tensor([[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]]))
    accumulator.update(torch.tensor([[[9.0, 11.0]]]))

    assert torch.equal(accumulator.finalize(), torch.tensor([4.5, 6.5]))


class CalibrationBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 4, bias=False)
        self.second = nn.Linear(4, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(value)))


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.hidden = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 7, bias=False)
        self._input_hook: torch.utils.hooks.RemovableHandle | None = None

    def enable_input_require_grads(self) -> None:
        self._input_hook = self.embedding.register_forward_hook(
            lambda _module, _inputs, output: output.requires_grad_(True)
        )

    def disable_input_require_grads(self) -> None:
        if self._input_hook is not None:
            self._input_hook.remove()
            self._input_hook = None

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        del use_cache
        hidden = torch.tanh(self.hidden(self.embedding(input_ids)))
        return SimpleNamespace(logits=self.lm_head(hidden))


class TinyTextStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.hidden = nn.Linear(4, 4, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        del use_cache
        return SimpleNamespace(last_hidden_state=torch.tanh(self.hidden(self.embedding(input_ids))))


class TinySplitCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyTextStack()
        self.lm_head = nn.Linear(4, 7, bias=False)
        self._input_hook: torch.utils.hooks.RemovableHandle | None = None

    def enable_input_require_grads(self) -> None:
        self._input_hook = self.model.embedding.register_forward_hook(
            lambda _module, _inputs, output: output.requires_grad_(True)
        )

    def disable_input_require_grads(self) -> None:
        if self._input_hook is not None:
            self._input_hook.remove()
            self._input_hook = None

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> SimpleNamespace:
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids, use_cache).last_hidden_state))


def _runner(block: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return block(value)


def test_activation_math_known_values_and_shrinkage() -> None:
    value = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    assert robust_tau(value, percentile=0.5) == 5
    assert torch.equal(activation_square_mean(value), torch.tensor([4.5, 10.0]))
    assert torch.equal(shrink_importance(torch.tensor([1.0, 3.0]), 0.5), torch.tensor([1.5, 2.5]))


def test_activation_square_mean_matches_legacy_512_token_partial_accumulation() -> None:
    values = torch.randn(1025, 7, generator=torch.Generator().manual_seed(19))
    expected = torch.zeros(7)
    for start in range(0, values.shape[0], 512):
        expected.add_(values[start : start + 512].float().square().sum(dim=0))
    expected.div_(values.shape[0])

    assert torch.equal(activation_square_mean(values), expected)


def test_online_forward_and_two_phase_calibration_are_typed_finite_and_remove_hooks() -> None:
    batches = (
        torch.randn(2, 3, generator=torch.Generator().manual_seed(1)),
        torch.randn(2, 3, generator=torch.Generator().manual_seed(2)),
    )
    for method in ("online_fisher", "two_phase_fisher", "forward_only"):
        block = CalibrationBlock()
        progress = []
        results = calibrate_block(
            block,
            batches,
            ("first", "second"),
            _runner,
            method=method,
            shrinkage=0.2,
            progress_callback=lambda completed, total, sink=progress: sink.append(
                (completed, total)
            ),
        )
        total = len(batches) * (2 if method == "two_phase_fisher" else 1)
        assert progress == [(completed, total) for completed in range(1, total + 1)]
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


def test_causal_loss_applies_exact_next_token_shift() -> None:
    logits = torch.zeros(1, 3, 3)
    logits[0, 0, 1] = 4
    logits[0, 1, 2] = 4
    tokens = torch.tensor([[0, 1, 2]])
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 3), tokens[:, 1:].reshape(-1))
    assert torch.equal(causal_language_model_loss(logits, tokens), expected)


def test_memory_efficient_causal_loss_disables_compiled_vocabulary_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]] = []
    expected = torch.tensor(2.5)

    def fake_linear_cross_entropy(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        tokens: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append((hidden, weight, tokens, kwargs))
        return expected

    monkeypatch.setattr("nanoquant.application.calibration.linear_cross_entropy", fake_linear_cross_entropy)
    hidden = torch.randn(1, 4, 3)
    weight = torch.randn(7, 3)
    tokens = torch.tensor([[0, 1, 2, 3]])

    actual = memory_efficient_causal_language_model_loss(hidden, weight, tokens)

    assert actual is expected
    assert calls == [(hidden, weight, tokens, {"shift": True, "filter_eps": None})]


@pytest.mark.parametrize("method", ["online_fisher", "two_phase_fisher"])
def test_full_causal_calibration_collects_output_fisher_and_restores_model(method: str) -> None:
    model = TinyCausalModel().eval()
    original_requires_grad = tuple(parameter.requires_grad for parameter in model.parameters())
    progress = []
    results = calibrate_causal_model(
        model,
        (torch.tensor([[0, 1, 2, 3]]), torch.tensor([[3, 2, 1, 0]])),
        (("hidden", model.hidden),),
        method=method,
        shrinkage=0.2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )
    total = 4 if method == "two_phase_fisher" else 2
    assert progress == [(completed, total) for completed in range(1, total + 1)]
    assert len(results) == 1
    assert results[0].sample_count == 2
    assert results[0].method == method
    assert torch.isfinite(results[0].input_importance).all()
    assert torch.isfinite(results[0].output_importance).all()
    assert torch.count_nonzero(results[0].output_importance) > 0
    assert model.training is False
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == original_requires_grad
    assert all(not module._forward_hooks and not module._backward_hooks for module in model.modules())


def test_chunked_hidden_gradient_matches_direct_causal_loss() -> None:
    direct = TinyCausalModel()
    chunked = TinySplitCausalModel()
    chunked.model.embedding.load_state_dict(direct.embedding.state_dict())
    chunked.model.hidden.load_state_dict(direct.hidden.state_dict())
    chunked.lm_head.load_state_dict(direct.lm_head.state_dict())
    batches = (torch.tensor([[0, 1, 2, 3, 4, 5]]),)
    direct_stats = calibrate_causal_model(direct, batches, (("hidden", direct.hidden),))
    chunked_stats = calibrate_causal_model(chunked, batches, (("hidden", chunked.model.hidden),))
    assert torch.allclose(direct_stats[0].input_importance, chunked_stats[0].input_importance)
    assert torch.allclose(direct_stats[0].output_importance, chunked_stats[0].output_importance, atol=1e-10)


def test_resumed_online_causal_accumulators_match_uninterrupted_result() -> None:
    batches = tuple(torch.tensor([[index % 7, (index + 1) % 7, (index + 2) % 7]]) for index in range(4))
    uninterrupted_model = TinyCausalModel()
    resumed_model = TinyCausalModel()
    resumed_model.load_state_dict(uninterrupted_model.state_dict())
    uninterrupted = calibrate_causal_model(
        uninterrupted_model,
        batches,
        (("hidden", uninterrupted_model.hidden),),
        shrinkage=0.2,
    )
    states = []
    calibrate_causal_model(
        resumed_model,
        batches[:2],
        (("hidden", resumed_model.hidden),),
        state_sink=states.append,
    )
    calibrate_causal_model(
        resumed_model,
        batches[2:],
        (("hidden", resumed_model.hidden),),
        initial_state=states[-1],
        state_sink=states.append,
    )
    resumed = materialize_causal_online_state(states[-1], shrinkage=0.2)
    assert states[-1].sample_count == 4
    assert torch.equal(uninterrupted[0].input_importance, resumed[0].input_importance)
    assert torch.equal(uninterrupted[0].output_importance, resumed[0].output_importance)
    assert torch.equal(uninterrupted[0].input_mean, resumed[0].input_mean)


def test_online_causal_calibration_rejects_incompatible_numerical_state() -> None:
    model = TinyCausalModel()
    incompatible = CausalOnlineCalibrationState((), 0, CAUSAL_CALIBRATION_ALGORITHM_VERSION - 1)
    with pytest.raises(ValueError, match="incompatible numerical algorithm"):
        calibrate_causal_model(
            model,
            (torch.tensor([[0, 1, 2]]),),
            (("hidden", model.hidden),),
            initial_state=incompatible,
        )


def test_causal_calibration_micro_profile_preserves_results_and_records_accumulation() -> None:
    control_model = TinyCausalModel()
    profiled_model = deepcopy(control_model)
    batches = (torch.tensor([[0, 1, 2, 3]]), torch.tensor([[3, 2, 1, 0]]))
    control = calibrate_causal_model(
        control_model,
        batches,
        (("hidden", control_model.hidden),),
        shrinkage=0.2,
    )
    profiler = Profiler(
        ProfilingConfig(level=ProfilingLevel.MICRO, emit_span_events=False),
        run_id="calibration-micro",
    )
    profiled = calibrate_causal_model(
        profiled_model,
        batches,
        (("hidden", profiled_model.hidden),),
        shrinkage=0.2,
        recorder=profiler,
    )

    assert len(profiled) == len(control)
    for actual, expected in zip(profiled, control, strict=True):
        assert actual.path == expected.path
        assert actual.sample_count == expected.sample_count
        assert actual.method == expected.method
        assert torch.equal(actual.input_importance, expected.input_importance)
        assert torch.equal(actual.output_importance, expected.output_importance)
    payload = profiler.snapshot()
    phase_paths = {str(phase["path"]) for phase in payload["phases"]}  # type: ignore[index]
    assert {
        "forward",
        "forward/accumulate",
        "loss",
        "backward",
        "backward/accumulate",
        "shrinkage",
    } <= phase_paths
    counters = {str(counter["name"]): counter for counter in payload["counters"]}  # type: ignore[index]
    assert counters["calibration.batches"]["total"] == 2
    assert counters["calibration.samples"]["total"] == 2
    assert counters["calibration.accumulator_updates"]["total"] == 4


def test_two_phase_block_calibration_profiles_both_passes_without_changing_results() -> None:
    batches = (torch.randn(2, 3, generator=torch.Generator().manual_seed(23)),)
    control_block = CalibrationBlock()
    profiled_block = deepcopy(control_block)
    control = calibrate_block(
        control_block,
        batches,
        ("first", "second"),
        _runner,
        method="two_phase_fisher",
    )
    profiler = Profiler(
        ProfilingConfig(level=ProfilingLevel.MICRO, emit_span_events=False),
        run_id="block-calibration-micro",
    )
    profiled = calibrate_block(
        profiled_block,
        batches,
        ("first", "second"),
        _runner,
        method="two_phase_fisher",
        recorder=profiler,
    )

    for actual, expected in zip(profiled, control, strict=True):
        assert torch.equal(actual.input_importance, expected.input_importance)
        assert torch.equal(actual.output_importance, expected.output_importance)
    payload = profiler.snapshot()
    counters = {str(counter["name"]): counter for counter in payload["counters"]}  # type: ignore[index]
    assert counters["calibration.batches"]["total"] == 2
    assert counters["calibration.samples"]["total"] == 4
    assert counters["calibration.accumulator_updates"]["total"] == 8
