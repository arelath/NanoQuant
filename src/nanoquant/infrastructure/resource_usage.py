"""Process resource measurements without an optional monitoring dependency."""

from __future__ import annotations

import ctypes
import importlib
import mmap
import os
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nanoquant.domain.resources import peak_device_memory_bytes as peak_device_memory_bytes


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


@dataclass(frozen=True, slots=True)
class ProcessMemorySnapshot:
    """Current and lifetime-high-water process memory counters."""

    working_set_bytes: int
    peak_working_set_bytes: int
    private_bytes: int
    peak_private_bytes: int


@dataclass(frozen=True, slots=True)
class GpuProcessMemorySnapshot:
    """Current and lifetime-high-water WDDM process-memory counters."""

    dedicated_bytes: int
    shared_bytes: int
    peak_dedicated_bytes: int
    peak_shared_bytes: int


class _PdhValueUnion(ctypes.Union):
    _fields_ = [
        ("long_value", ctypes.c_long),
        ("double_value", ctypes.c_double),
        ("large_value", ctypes.c_longlong),
    ]


class _PdhFormattedCounterValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("status", ctypes.c_ulong), ("value", _PdhValueUnion)]


class _PdhFormattedCounterValueItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("value", _PdhFormattedCounterValue)]


_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2


def _pdh_status(value: int) -> int:
    return int(ctypes.c_uint32(value).value)


class _WindowsGpuProcessMemorySampler:
    def __init__(self, process_id: int) -> None:
        self._process_prefix = f"pid_{process_id}_"
        self._pdh = cast(Any, ctypes.WinDLL("pdh", use_last_error=True))
        self._query = ctypes.c_void_p()
        self._dedicated = ctypes.c_void_p()
        self._shared = ctypes.c_void_p()
        self._configure_api()
        self._check(self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)), "PdhOpenQueryW")
        self._add_counter(r"\GPU Process Memory(*)\Dedicated Usage", self._dedicated)
        self._add_counter(r"\GPU Process Memory(*)\Shared Usage", self._shared)
        self._peak_dedicated = 0
        self._peak_shared = 0

    def _configure_api(self) -> None:
        self._pdh.PdhOpenQueryW.argtypes = (ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p))
        self._pdh.PdhOpenQueryW.restype = ctypes.c_long
        self._pdh.PdhAddEnglishCounterW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
        self._pdh.PdhCollectQueryData.argtypes = (ctypes.c_void_p,)
        self._pdh.PdhCollectQueryData.restype = ctypes.c_long
        self._pdh.PdhGetFormattedCounterArrayW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_void_p,
        )
        self._pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long

    @staticmethod
    def _check(status: int, operation: str) -> None:
        normalized = _pdh_status(status)
        if normalized != 0:
            raise OSError(normalized, f"{operation} failed")

    def _add_counter(self, path: str, destination: ctypes.c_void_p) -> None:
        self._check(
            self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(destination)),
            "PdhAddEnglishCounterW",
        )

    def _counter_total(self, counter: ctypes.c_void_p) -> int:
        buffer_size = ctypes.c_ulong(0)
        item_count = ctypes.c_ulong(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter,
            _PDH_FMT_LARGE,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            None,
        )
        normalized = _pdh_status(status)
        if normalized not in (0, _PDH_MORE_DATA):
            raise OSError(normalized, "PdhGetFormattedCounterArrayW size query failed")
        if buffer_size.value == 0:
            return 0
        buffer = ctypes.create_string_buffer(buffer_size.value)
        self._check(
            self._pdh.PdhGetFormattedCounterArrayW(
                counter,
                _PDH_FMT_LARGE,
                ctypes.byref(buffer_size),
                ctypes.byref(item_count),
                buffer,
            ),
            "PdhGetFormattedCounterArrayW",
        )
        items = ctypes.cast(buffer, ctypes.POINTER(_PdhFormattedCounterValueItem))
        return sum(
            max(0, int(items[index].value.large_value))
            for index in range(item_count.value)
            if (items[index].name or "").lower().startswith(self._process_prefix)
        )

    def sample(self) -> GpuProcessMemorySnapshot:
        self._check(self._pdh.PdhCollectQueryData(self._query), "PdhCollectQueryData")
        dedicated = self._counter_total(self._dedicated)
        shared = self._counter_total(self._shared)
        self._peak_dedicated = max(self._peak_dedicated, dedicated)
        self._peak_shared = max(self._peak_shared, shared)
        return GpuProcessMemorySnapshot(dedicated, shared, self._peak_dedicated, self._peak_shared)


_gpu_process_memory_lock = threading.Lock()
_gpu_process_memory_sampler: _WindowsGpuProcessMemorySampler | None = None
_gpu_process_memory_unavailable = False


def gpu_process_memory_snapshot() -> GpuProcessMemorySnapshot | None:
    """Return current-process WDDM dedicated/shared usage when available."""

    global _gpu_process_memory_sampler, _gpu_process_memory_unavailable
    if os.name != "nt" or _gpu_process_memory_unavailable:
        return None
    with _gpu_process_memory_lock:
        try:
            if _gpu_process_memory_sampler is None:
                _gpu_process_memory_sampler = _WindowsGpuProcessMemorySampler(os.getpid())
            return _gpu_process_memory_sampler.sample()
        except (OSError, AttributeError):
            _gpu_process_memory_unavailable = True
            return None


def process_memory_snapshot() -> ProcessMemorySnapshot:
    """Return non-synchronizing process memory counters when the platform exposes them."""
    if os.name == "nt":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return ProcessMemorySnapshot(
            int(counters.WorkingSetSize),
            int(counters.PeakWorkingSetSize),
            int(counters.PagefileUsage),
            int(counters.PeakPagefileUsage),
        )
    resource = cast(Any, importlib.import_module("resource"))
    raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(raw_peak if platform.system() == "Darwin" else raw_peak * 1024)
    current = peak
    if platform.system() == "Linux":
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            current = int(fields[1]) * mmap.PAGESIZE
        except (OSError, IndexError, ValueError):
            current = peak
    return ProcessMemorySnapshot(current, peak, 0, 0)


def peak_process_memory_bytes() -> int:
    """Return peak resident/working-set bytes for the current process."""
    return process_memory_snapshot().peak_working_set_bytes
