from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HostInventory:
    cpu_bytes_available: int
    gpu_bytes_available: int
    temporary_disk_bytes_available: int


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    peak_cpu_bytes: int = 0
    peak_gpu_bytes: int = 0
    temporary_disk_bytes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    findings: tuple[ValidationFinding, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)
