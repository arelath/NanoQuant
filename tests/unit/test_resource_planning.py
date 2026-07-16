import pytest

from nanoquant.domain.resources import ResourceComponents, ResourceMargins
from nanoquant.domain.stages import HostInventory
from nanoquant.infrastructure.resource_planning import (
    InsufficientResourcesError,
    ResourcePlanningRequest,
    build_resource_plan,
)


def _components() -> ResourceComponents:
    mib = 1024**2
    return ResourceComponents(
        source_checkpoint_bytes=10 * mib,
        packed_output_bytes=2 * mib,
        active_block_bytes=1 * mib,
        factor_workspace_bytes=1 * mib,
        hessian_bytes=0,
        activation_bytes=8 * mib,
        tuning_state_bytes=0,
        committed_artifact_bytes=2 * mib,
    )


def test_resource_plan_selects_mmap_when_activations_do_not_fit_host_or_gpu() -> None:
    mib = 1024**2
    plan = build_resource_plan(
        ResourcePlanningRequest(_components(), margins=ResourceMargins(0, 0, 0)),
        HostInventory(
            cpu_bytes_available=4 * mib, gpu_bytes_available=2 * mib, temporary_disk_bytes_available=40 * mib
        ),
    )

    assert plan.executor == "streaming"
    assert plan.activation_tier == "mmap"
    assert plan.peak_gpu_bytes == 2 * mib
    assert plan.peak_host_bytes == 2 * mib
    assert plan.temporary_disk_bytes == 22 * mib


def test_resource_preflight_refuses_before_work_when_margin_adjusted_minimum_is_missing() -> None:
    mib = 1024**2
    with pytest.raises(InsufficientResourcesError, match=r"RES001.*temporary disk requires"):
        build_resource_plan(
            ResourcePlanningRequest(_components()),
            HostInventory(
                cpu_bytes_available=64 * mib,
                gpu_bytes_available=64 * mib,
                temporary_disk_bytes_available=8 * mib,
            ),
        )


def test_explicit_cuda_activation_tier_is_included_in_peak() -> None:
    mib = 1024**2
    plan = build_resource_plan(
        ResourcePlanningRequest(
            _components(),
            requested_executor="resident",
            requested_activation_tier="cuda",
            margins=ResourceMargins(0, 0, 0),
        ),
        HostInventory(64 * mib, 64 * mib, 64 * mib),
    )
    assert plan.peak_gpu_bytes == 20 * mib


def test_auto_uses_cpu_offload_between_resident_and_streaming() -> None:
    mib = 1024**2
    plan = build_resource_plan(
        ResourcePlanningRequest(_components(), margins=ResourceMargins(0, 0, 0)),
        HostInventory(64 * mib, 4 * mib, 64 * mib),
    )

    assert plan.executor == "cpu_offload"
    assert plan.peak_gpu_bytes == 3 * mib
    assert plan.peak_host_bytes == 20 * mib


def test_auto_skips_cpu_offload_when_source_shell_does_not_fit_host() -> None:
    mib = 1024**2
    plan = build_resource_plan(
        ResourcePlanningRequest(_components(), margins=ResourceMargins(0, 0, 0)),
        HostInventory(10 * mib, 4 * mib, 64 * mib),
    )

    assert plan.executor == "streaming"
    assert plan.peak_gpu_bytes == 2 * mib
