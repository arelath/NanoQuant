from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import nanoquant.resident_quantization as resident
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ADMMConfig,
    ExecutorKind,
    LayerRankBudgetConfig,
    ObservabilityConfig,
    ProfilingConfig,
    ProfilingLevel,
)
from nanoquant.resident_quantization import ResidentQuantizationRequest


def test_resident_algorithm_version_invalidates_commit_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"),
        Path("output"),
        "fixture/model",
        "revision",
        ((1, 2, 3),),
        device="cpu",
    )
    original = resident._resident_config_hash(request)

    monkeypatch.setattr(
        resident,
        "RESIDENT_ALGORITHM_VERSION",
        resident.RESIDENT_ALGORITHM_VERSION + 1,
    )

    assert resident._resident_config_hash(request) != original


def test_torch_runtime_version_invalidates_commit_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"),
        Path("output"),
        "fixture/model",
        "revision",
        ((1, 2, 3),),
        device="cpu",
    )
    original = resident._resident_config_hash(request)

    monkeypatch.setattr(torch, "__version__", "different-runtime")

    assert resident._resident_config_hash(request) != original


def test_legacy_tuning_seed_mode_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(request, legacy_tuning_seed_reset=True)
    )


def test_tuning_epoch_loss_mode_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(request, restore_best_tuning_state=False, tuning_epoch_loss_mode="legacy_training")
    )


def test_admm_orientation_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(request, admm=ADMMConfig(outer_iterations=1, inner_iterations=1, transpose_wide=True))
    )


def test_rank_retry_policy_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(request, rank_retry=replace(request.rank_retry, maximum_attempts=2))
    )


def test_maximum_rank_policy_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(request, maximum_rank_layer_patterns=("self_attn.v_proj",))
    )


def test_layer_budget_multiplier_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(
            request,
            layer_budget_multipliers=(LayerRankBudgetConfig("self_attn.q_proj", 1.25),),
        )
    )


def test_executor_placement_does_not_invalidate_semantic_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) == resident._resident_config_hash(
        replace(
            request,
            executor=ExecutorKind.CPU_OFFLOAD,
            restore_completed_blocks=False,
            evaluate_inline_quality=False,
        )
    )
    assert resident._model_placement_device(
        replace(request, device="cuda:0", executor=ExecutorKind.CPU_OFFLOAD)
    ) == "cpu"


def test_activation_gpu_cache_does_not_invalidate_semantic_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) == resident._resident_config_hash(
        replace(
            request,
            activation_gpu_cache=ActivationGpuCacheMode.BOTH,
            activation_gpu_reserve_bytes=1234,
        )
    )
    assert resident._activation_cache_fits(20, 100, 80)
    assert not resident._activation_cache_fits(21, 100, 80)


def test_activation_gpu_cache_auto_falls_back_but_explicit_policy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.zeros(4)
    request = ResidentQuantizationRequest(
        Path("snapshot"),
        Path("output"),
        "fixture/model",
        "revision",
        ((1, 2, 3),),
        device="cuda:0",
        activation_gpu_reserve_bytes=8,
    )
    events = Mock()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (15, 100))

    assert resident._cache_activation_tensor(
        value,
        request,
        events,
        role="compressed_inputs",
        required=False,
    ) is value
    assert events.emit.call_args.args[:3] == (
        "resource",
        "info",
        "activation_gpu_cache.skipped",
    )
    with pytest.raises(RuntimeError, match="requires 16 bytes plus 8 reserved bytes"):
        resident._cache_activation_tensor(
            value,
            request,
            events,
            role="compressed_inputs",
            required=True,
        )


def test_profiling_does_not_invalidate_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) == resident._resident_config_hash(
        replace(
            request,
            profiling=ProfilingConfig(level=ProfilingLevel.MICRO, cuda_timing=True),
        )
    )


def test_observability_does_not_invalidate_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) == resident._resident_config_hash(
        replace(
            request,
            observability=ObservabilityConfig(event_level="debug"),
        )
    )


def test_legacy_cuda_numerics_enables_and_restores_tf32() -> None:
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with resident._legacy_cuda_numerics():
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
