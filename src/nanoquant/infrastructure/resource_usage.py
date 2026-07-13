"""Process resource measurements without an optional monitoring dependency."""

from __future__ import annotations

import ctypes
import importlib
import mmap
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


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
