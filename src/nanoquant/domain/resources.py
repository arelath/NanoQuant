"""Pure resource planning types for resident and streaming execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceComponents:
    source_checkpoint_bytes: int
    packed_output_bytes: int
    active_block_bytes: int
    factor_workspace_bytes: int
    hessian_bytes: int
    activation_bytes: int
    tuning_state_bytes: int
    committed_artifact_bytes: int
    temporary_overhead_bytes: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.as_tuple()):
            raise ValueError("resource component sizes cannot be negative")

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.source_checkpoint_bytes,
            self.packed_output_bytes,
            self.active_block_bytes,
            self.factor_workspace_bytes,
            self.hessian_bytes,
            self.activation_bytes,
            self.tuning_state_bytes,
            self.committed_artifact_bytes,
            self.temporary_overhead_bytes,
        )


@dataclass(frozen=True, slots=True)
class ResourceMargins:
    gpu_fraction: float = 0.10
    host_fraction: float = 0.10
    disk_fraction: float = 0.10

    def __post_init__(self) -> None:
        if any(value < 0 or value >= 1 for value in (self.gpu_fraction, self.host_fraction, self.disk_fraction)):
            raise ValueError("resource safety margins must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    executor: str
    activation_tier: str
    peak_gpu_bytes: int
    peak_host_bytes: int
    temporary_disk_bytes: int
    bytes_read: int
    bytes_written: int
    gpu_limit_after_margin: int
    host_limit_after_margin: int
    disk_limit_after_margin: int
    components: ResourceComponents
    warnings: tuple[str, ...] = ()
