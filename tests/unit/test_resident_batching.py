from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.resident_quantization import (
    _block_loss,
    _place_completed_decoder_block,
    _release_uncompleted_decoder_blocks,
    _run_block_batched,
    _run_quality_logits_batched,
    _streamed_quality_metrics,
)


class _BlockAdapter:
    def run_block(self, block: nn.Module, value: torch.Tensor, **_metadata: object) -> torch.Tensor:
        return block(value)


class _QualityModel(nn.Module):
    def __init__(self, logits_by_token: torch.Tensor) -> None:
        super().__init__()
        self.logits_by_token = nn.Parameter(logits_by_token)
        self.batch_sizes: list[int] = []

    def forward(self, input_ids: torch.Tensor, *, use_cache: bool) -> SimpleNamespace:
        assert use_cache is False
        self.batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(logits=self.logits_by_token[input_ids])


class _QualityAdapter:
    def run_full_forward(self, model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        return model(input_ids=tokens, use_cache=False).logits  # type: ignore[no-any-return, operator]


def test_quality_evaluation_streams_sequences_to_pageable_cpu_and_matches_full_metrics() -> None:
    sample_count = 3
    sequence_length = 300
    vocabulary_size = 11
    tokens = torch.arange(sample_count * sequence_length).reshape(sample_count, sequence_length) % vocabulary_size
    base_logits = torch.randn(
        vocabulary_size,
        vocabulary_size,
        generator=torch.Generator().manual_seed(47),
    )
    perturbation = torch.linspace(-0.2, 0.3, vocabulary_size).outer(
        torch.linspace(0.1, 0.9, vocabulary_size)
    )
    reference_model = _QualityModel(base_logits)
    compressed_model = _QualityModel(base_logits + perturbation)
    adapter = _QualityAdapter()

    reference_logits = _run_quality_logits_batched(adapter, reference_model, tokens, "cpu")  # type: ignore[arg-type]

    assert reference_model.batch_sizes == [1] * sample_count
    assert reference_logits.device.type == "cpu"
    assert not reference_logits.is_pinned()
    with torch.no_grad():
        compressed_logits = compressed_model(input_ids=tokens, use_cache=False).logits
    expected = (
        float(
            torch.nn.functional.cross_entropy(
                reference_logits[:, :-1].reshape(-1, vocabulary_size),
                tokens[:, 1:].reshape(-1),
            )
        ),
        float(
            torch.nn.functional.cross_entropy(
                compressed_logits[:, :-1].reshape(-1, vocabulary_size),
                tokens[:, 1:].reshape(-1),
            )
        ),
        float((compressed_logits - reference_logits).square().mean()),
        float((compressed_logits.argmax(-1) == reference_logits.argmax(-1)).float().mean()),
    )
    compressed_model.batch_sizes.clear()

    actual = _streamed_quality_metrics(  # type: ignore[arg-type]
        adapter, compressed_model, tokens, reference_logits
    )

    assert compressed_model.batch_sizes == [1] * sample_count
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_uncompleted_dense_decoder_blocks_are_released_from_model_shell() -> None:
    completed = nn.Linear(2, 2)
    layers = nn.ModuleList((nn.Linear(2, 2), completed, nn.Linear(2, 2)))

    released = _release_uncompleted_decoder_blocks(layers, {1})

    assert released == 2
    assert isinstance(layers[0], nn.Identity)
    assert layers[1] is completed
    assert isinstance(layers[2], nn.Identity)


def test_completed_decoder_block_is_not_retained_when_full_model_restore_is_disabled() -> None:
    layers = nn.ModuleList((nn.Identity(), nn.Identity()))
    completed = nn.Linear(2, 2)

    retained = _place_completed_decoder_block(layers, 0, completed, retain=False)

    assert not retained
    assert isinstance(layers[0], nn.Identity)


def test_completed_decoder_block_is_retained_for_inline_full_model_forward() -> None:
    layers = nn.ModuleList((nn.Identity(), nn.Identity()))
    completed = nn.Linear(2, 2)

    retained = _place_completed_decoder_block(layers, 0, completed, retain=True)

    assert retained
    assert layers[0] is completed


def test_block_forward_does_not_retain_autograd_graphs() -> None:
    inputs = torch.randn(5, 4, requires_grad=True)
    block = nn.Linear(4, 3, bias=False)

    actual = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2)

    assert not actual.requires_grad
    assert actual.grad_fn is None


def test_block_forward_micro_profile_preserves_output_and_attributes_batches() -> None:
    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(44))
    block = nn.Linear(4, 3, bias=False)
    expected = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2)
    profiler = Profiler(ProfilingConfig(level=ProfilingLevel.MICRO), run_id="block-forward")

    actual = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2, recorder=profiler)

    assert torch.equal(actual, expected)
    payload = profiler.snapshot()
    phases = {phase["path"]: phase for phase in payload["phases"]}
    assert {"batch_stage", "forward", "store"} <= phases.keys()
    assert phases["batch_stage"]["count"] == phases["forward"]["count"] == phases["store"]["count"] == 3
    assert all(phase["failed_count"] == 0 for phase in phases.values())
    counters = {counter["name"]: counter for counter in payload["counters"]}
    assert counters["forward.batches"]["total"] == 3
    assert counters["forward.elements"]["total"] == actual.numel()


def test_block_loss_micro_profile_preserves_accumulation_and_attributes_work() -> None:
    inputs = torch.randn(5, 4, generator=torch.Generator().manual_seed(45))
    targets = torch.randn(5, 3, generator=torch.Generator().manual_seed(46))
    importance = torch.linspace(0.5, 1.5, 3)
    block = nn.Linear(4, 3, bias=False)
    expected = _block_loss(_BlockAdapter(), block, inputs, targets, importance, {}, 2)
    profiler = Profiler(ProfilingConfig(level=ProfilingLevel.MICRO), run_id="block-loss")

    actual = _block_loss(_BlockAdapter(), block, inputs, targets, importance, {}, 2, profiler)

    assert actual == expected
    payload = profiler.snapshot()
    phases = {phase["path"]: phase for phase in payload["phases"]}
    assert {"batch_stage", "forward", "loss", "synchronize"} <= phases.keys()
    assert phases["batch_stage"]["count"] == phases["forward"]["count"] == phases["loss"]["count"] == 3
    assert phases["synchronize"]["count"] == 1
    assert all(phase["failed_count"] == 0 for phase in phases.values())
    counters = {counter["name"]: counter for counter in payload["counters"]}
    assert counters["forward.batches"]["total"] == 3
    assert counters["forward.elements"]["total"] == targets.numel()


def test_block_loss_streamed_fp32_accumulation_matches_full_tensor_formula() -> None:
    inputs = torch.randn(2, 300, 4, generator=torch.Generator().manual_seed(52))
    targets = torch.randn(2, 300, 3, generator=torch.Generator().manual_seed(53))
    importance = torch.linspace(0.5, 1.5, 3)
    block = nn.Linear(4, 3, bias=False)
    prediction = block(inputs).detach()
    expected = float(((prediction.float() - targets.float()).square() * importance).sum() / targets.numel())

    actual = _block_loss(_BlockAdapter(), block, inputs, targets, importance, {}, 2)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA resident transfer requires a GPU")
def test_cuda_block_forward_produces_bitwise_equal_pageable_host_activations() -> None:
    inputs = torch.randn(5, 4, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(41))
    block = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        actual = _run_block_batched(_BlockAdapter(), block, inputs, {}, 2, "cpu")
        expected = torch.cat(
            [block(inputs[start : start + 2].cuda()).cpu() for start in range(0, inputs.shape[0], 2)]
        )

    assert not actual.is_pinned()
    assert torch.equal(actual, expected)
    del block
    torch.cuda.empty_cache()
    torch._C._accelerator_emptyHostCache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned CUDA transfer requires a GPU")
def test_prefetched_block_loss_matches_pageable_accumulation_bitwise() -> None:
    inputs = torch.randn(7, 4, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(42))
    targets = torch.randn(7, 3, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(43))
    importance = torch.linspace(0.5, 1.5, 3)
    block = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16, device="cuda")

    pageable = _block_loss(_BlockAdapter(), block, inputs, targets, importance, {}, 2)
    prefetched = _block_loss(
        _BlockAdapter(), block, inputs.pin_memory(), targets.pin_memory(), importance, {}, 2
    )

    assert prefetched == pageable
    del block
    torch.cuda.empty_cache()
    torch._C._accelerator_emptyHostCache()
