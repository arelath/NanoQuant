"""Host inventory, placement selection, and preflight resource refusal."""

from __future__ import annotations

import ctypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.domain.resources import ResourceComponents, ResourceMargins, ResourcePlan
from nanoquant.domain.stages import HostInventory


class InsufficientResourcesError(RuntimeError):
    code = "RES001"


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _available_host_memory() -> int:
    if os.name != "nt":
        sysconf = cast(Any, os).sysconf
        page_size = sysconf("SC_PAGE_SIZE")
        available_pages = sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def inspect_host(temporary_directory: str | Path, device: str = "cuda:0") -> HostInventory:
    gpu_bytes = 0
    if device.startswith("cuda") and torch.cuda.is_available():
        with torch.cuda.device(device):
            gpu_bytes, _total = torch.cuda.mem_get_info()
    disk_bytes = shutil.disk_usage(Path(temporary_directory).resolve()).free
    return HostInventory(_available_host_memory(), int(gpu_bytes), disk_bytes)


@dataclass(frozen=True, slots=True)
class ResourcePlanningRequest:
    components: ResourceComponents
    requested_executor: str = "auto"
    requested_activation_tier: str = "auto"
    margins: ResourceMargins = ResourceMargins()


def _limit(available: int, margin: float) -> int:
    return int(available * (1 - margin))


def build_resource_plan(request: ResourcePlanningRequest, host: HostInventory) -> ResourcePlan:
    components = request.components
    gpu_limit = _limit(host.gpu_bytes_available, request.margins.gpu_fraction)
    host_limit = _limit(host.cpu_bytes_available, request.margins.host_fraction)
    disk_limit = _limit(host.temporary_disk_bytes_available, request.margins.disk_fraction)
    resident_base_gpu = (
        components.source_checkpoint_bytes
        + components.active_block_bytes
        + components.factor_workspace_bytes
        + components.hessian_bytes
        + components.tuning_state_bytes
    )
    resident_cuda_gpu = resident_base_gpu + components.activation_bytes
    streaming_gpu = (
        components.active_block_bytes
        + components.factor_workspace_bytes
        + components.hessian_bytes
        + components.tuning_state_bytes
    )
    base_host = components.active_block_bytes + components.factor_workspace_bytes + components.hessian_bytes
    executor = request.requested_executor
    if executor == "auto":
        executor = "resident" if resident_cuda_gpu <= gpu_limit else "streaming"
    if executor not in {"resident", "cpu_offload", "streaming"}:
        raise ValueError(f"unsupported executor: {executor}")
    peak_gpu = resident_base_gpu if executor == "resident" else streaming_gpu

    tier = request.requested_activation_tier
    if tier == "auto":
        if executor == "resident" and peak_gpu + components.activation_bytes <= gpu_limit:
            tier = "cuda"
        elif base_host + components.activation_bytes <= host_limit:
            tier = "pinned_ram" if components.activation_bytes <= host_limit // 4 else "ram"
        else:
            tier = "mmap"
    if tier not in {"cuda", "pinned_ram", "ram", "mmap"}:
        raise ValueError(f"unsupported activation tier: {tier}")
    if tier == "cuda":
        peak_gpu += components.activation_bytes
    peak_host = base_host + (components.activation_bytes if tier in {"pinned_ram", "ram"} else 0)
    temporary_disk = (
        components.source_checkpoint_bytes
        + components.packed_output_bytes
        + components.committed_artifact_bytes
        + components.temporary_overhead_bytes
        + (components.activation_bytes if tier == "mmap" else 0)
    )
    failures = []
    if peak_gpu > gpu_limit:
        failures.append(f"GPU requires {peak_gpu} bytes but margin-adjusted limit is {gpu_limit}")
    if peak_host > host_limit:
        failures.append(f"host requires {peak_host} bytes but margin-adjusted limit is {host_limit}")
    if temporary_disk > disk_limit:
        failures.append(f"temporary disk requires {temporary_disk} bytes but margin-adjusted limit is {disk_limit}")
    if failures:
        raise InsufficientResourcesError("RES001 " + "; ".join(failures))
    return ResourcePlan(
        executor,
        tier,
        peak_gpu,
        peak_host,
        temporary_disk,
        components.source_checkpoint_bytes + components.active_block_bytes,
        components.packed_output_bytes + components.committed_artifact_bytes + components.activation_bytes,
        gpu_limit,
        host_limit,
        disk_limit,
        components,
    )
