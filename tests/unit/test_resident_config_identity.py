import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import nanoquant.resident_quantization as resident
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ADMMConfig,
    BinaryFactorSearchConfig,
    ExecutorKind,
    LayerRankBudgetConfig,
    ObjectiveConfig,
    ObjectiveKind,
    ObservabilityConfig,
    PostRefitCovarianceRefinementConfig,
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


def test_resident_algorithm_version_does_not_invalidate_calibration_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"),
        Path("output"),
        "fixture/model",
        "revision",
        ((1, 2, 3),),
        device="cpu",
        calibration_method="online_fisher",
        calibration_shrinkage=0.6,
    )
    original = resident._calibration_config_hash(request)

    monkeypatch.setattr(
        resident,
        "RESIDENT_ALGORITHM_VERSION",
        resident.RESIDENT_ALGORITHM_VERSION + 1,
    )

    assert resident._calibration_config_hash(request) == original
    assert resident._calibration_config_hash(replace(request, calibration_shrinkage=0.5)) != original


def test_legacy_preprocessing_pointer_recovers_only_matching_calibration_protocol(
    tmp_path: Path,
) -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"),
        tmp_path / "run",
        "fixture/model",
        "revision",
        ((1, 2, 3),),
        device="cpu",
        calibration_method="online_fisher",
        calibration_shrinkage=0.6,
    )
    state = request.output / "state"
    state.mkdir(parents=True)
    (state / "preprocessing.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "resident_config_hash": "sha256:historical",
                "calibration": {
                    "artifact_type": "calibration-stats",
                    "artifact_id": "sha256-calibration",
                    "schema_version": 1,
                },
                "objectives": {
                    "artifact_type": "objective-specs",
                    "artifact_id": "sha256-objectives",
                    "schema_version": 1,
                },
                "plan": {
                    "artifact_type": "quantization-plan",
                    "artifact_id": "sha256-plan",
                    "schema_version": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (request.output / "manifest.json").write_text(
        json.dumps(
            {
                "resolved_config": resident._resident_manifest_config(
                    request,
                    "resident-quantization",
                )
            }
        ),
        encoding="utf-8",
    )

    references, source = resident._resolve_calibration_references(request)

    assert references is not None
    assert tuple(reference.artifact_id for reference in references) == (
        "sha256-calibration",
        "sha256-objectives",
    )
    assert source == "legacy_preprocessing_calibration"
    assert resident._resolve_calibration_references(
        replace(request, calibration_shrinkage=0.5)
    ) == (None, "computed")


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


def test_binary_factor_search_policy_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(
            request,
            binary_factor_search=BinaryFactorSearchConfig(enabled=True),
        )
    )


def test_covariance_refinement_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(
            request,
            covariance_refinement=ObjectiveConfig(kind=ObjectiveKind.DENSE_HESSIAN),
        )
    )


def test_post_refit_covariance_refinement_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(
            request,
            post_refit_covariance_refinement=PostRefitCovarianceRefinementConfig(
                enabled=True,
                block_indices=(0,),
                shared_input_groups=("self_attn.attn_qkv",),
            ),
        )
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


def test_overcomplete_rank_ceiling_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(request, overcomplete_rank_ceiling_fraction=1.5)
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
    assert resident._activation_cache_reserve_bytes(20, 8, automatic=True) == 20
    assert resident._activation_cache_reserve_bytes(4, 8, automatic=True) == 8
    assert resident._activation_cache_reserve_bytes(20, 8, automatic=False) == 8


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
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (31, 100))

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
    assert events.emit.call_args.kwargs["configured_reserve_bytes"] == 8
    assert events.emit.call_args.kwargs["reserve_bytes"] == 16
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (23, 100))
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
