"""Deployment-only runtime backend contracts and capability matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import torch

DeviceLike: TypeAlias = str | torch.device
WorkloadKind: TypeAlias = Literal["prefill", "decode"]


@dataclass(frozen=True, slots=True)
class ProjectionMemberSpec:
    name: str
    row_start: int
    row_end: int

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith("/") or ".." in self.name.split("."):
            raise ValueError("runtime projection member name must be canonical")
        if self.row_start < 0 or self.row_end <= self.row_start:
            raise ValueError("runtime projection member slice must be non-empty")


@dataclass(frozen=True, slots=True)
class QuantizedLinearSpec:
    """Backend-independent description of one frozen NanoQuant linear."""

    name: str
    logical_format: str
    in_features: int
    out_features: int
    rank: int
    factor_dtype: str
    scale_dtype: str
    outlier_count: int = 0
    outlier_value_dtype: str | None = None
    has_outlier_scales: bool = False
    has_bias: bool = False
    members: tuple[ProjectionMemberSpec, ...] = ()
    patch_rank: int = 0
    patch_value_dtype: str | None = None
    bias_dtype: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith("/") or ".." in self.name.split("."):
            raise ValueError("runtime layer name must be a canonical dotted path")
        if self.in_features <= 0 or self.out_features <= 0 or self.rank <= 0:
            raise ValueError("runtime linear dimensions and rank must be positive")
        if self.outlier_count < 0 or self.outlier_count > self.in_features:
            raise ValueError("runtime outlier count is outside the input dimension")
        if (self.outlier_count == 0) != (self.outlier_value_dtype is None):
            raise ValueError("runtime outlier dtype must be present exactly when outliers are present")
        if self.has_outlier_scales and self.outlier_count == 0:
            raise ValueError("runtime outlier scales require outlier values")
        if self.patch_rank < 0 or self.patch_rank > min(self.in_features, self.out_features):
            raise ValueError("runtime patch rank is outside the linear dimensions")
        if (self.patch_rank == 0) != (self.patch_value_dtype is None):
            raise ValueError("runtime patch dtype must be present exactly when a patch is present")
        if not self.has_bias and self.bias_dtype is not None:
            raise ValueError("runtime bias dtype requires a bias")
        if self.members:
            if len(self.members) < 2 or len({member.name for member in self.members}) != len(self.members):
                raise ValueError("runtime projection group members must be unique")
            cursor = 0
            for member in self.members:
                if member.row_start != cursor:
                    raise ValueError("runtime projection group slices must be contiguous")
                cursor = member.row_end
            if cursor != self.out_features:
                raise ValueError("runtime projection group slices must cover the output")


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    kind: WorkloadKind
    device_type: str
    input_dtype: str
    batch_size: int
    token_count: int
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("prefill", "decode"):
            raise ValueError(f"unsupported runtime workload kind: {self.kind}")
        if not self.device_type:
            raise ValueError("runtime workload device type must be non-empty")
        if self.batch_size <= 0 or self.token_count <= 0:
            raise ValueError("runtime workload batch and token counts must be positive")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    logical_formats: tuple[str, ...]
    device_types: tuple[str, ...]
    input_dtypes: tuple[str, ...]
    factor_dtypes: tuple[str, ...]
    scale_dtypes: tuple[str, ...]
    outlier_value_dtypes: tuple[str, ...]
    workload_kinds: tuple[WorkloadKind, ...]
    supports_bias: bool
    supports_outliers: bool
    supports_deterministic: bool
    in_feature_alignment: int = 1
    out_feature_alignment: int = 1
    rank_alignment: int = 1
    maximum_batch_size: int | None = None
    maximum_token_count: int | None = None
    patch_value_dtypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.logical_formats or not self.device_types or not self.input_dtypes:
            raise ValueError("runtime backend capabilities must declare formats, devices, and input dtypes")
        if min(self.in_feature_alignment, self.out_feature_alignment, self.rank_alignment) <= 0:
            raise ValueError("runtime backend alignments must be positive")
        if self.maximum_batch_size is not None and self.maximum_batch_size <= 0:
            raise ValueError("runtime backend maximum batch size must be positive")
        if self.maximum_token_count is not None and self.maximum_token_count <= 0:
            raise ValueError("runtime backend maximum token count must be positive")


@dataclass(frozen=True, slots=True)
class SupportResult:
    supported: bool
    code: str
    reason: str

    @classmethod
    def accepted(cls) -> SupportResult:
        return cls(True, "NQ-INF-OK", "supported")

    @classmethod
    def rejected(cls, code: str, reason: str) -> SupportResult:
        if not code or not reason:
            raise ValueError("runtime backend rejection requires a code and reason")
        return cls(False, code, reason)


@runtime_checkable
class RuntimeLayerState(Protocol):
    @property
    def spec(self) -> QuantizedLinearSpec: ...


@dataclass(frozen=True, slots=True)
class PreparedLayer:
    backend_name: str
    backend_version: str
    spec: QuantizedLinearSpec
    payload: object


@runtime_checkable
class RuntimeBackend(Protocol):
    name: str
    version: str

    def capabilities(self) -> BackendCapabilities: ...

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult: ...

    def prepare(self, state: RuntimeLayerState, device: DeviceLike) -> PreparedLayer: ...

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor: ...


def evaluate_capabilities(
    capabilities: BackendCapabilities,
    op: QuantizedLinearSpec,
    workload: WorkloadSpec,
) -> SupportResult:
    """Return the first stable, actionable reason a backend cannot run an operation."""

    checks = (
        (
            op.logical_format in capabilities.logical_formats,
            "NQ-INF-FORMAT",
            f"logical format {op.logical_format!r} is not supported",
        ),
        (
            workload.device_type in capabilities.device_types,
            "NQ-INF-DEVICE",
            f"device type {workload.device_type!r} is not supported",
        ),
        (
            workload.input_dtype in capabilities.input_dtypes,
            "NQ-INF-INPUT-DTYPE",
            f"input dtype {workload.input_dtype!r} is not supported",
        ),
        (
            op.factor_dtype in capabilities.factor_dtypes,
            "NQ-INF-FACTOR-DTYPE",
            f"factor dtype {op.factor_dtype!r} is not supported",
        ),
        (
            op.scale_dtype in capabilities.scale_dtypes,
            "NQ-INF-SCALE-DTYPE",
            f"scale dtype {op.scale_dtype!r} is not supported",
        ),
        (
            workload.kind in capabilities.workload_kinds,
            "NQ-INF-WORKLOAD",
            f"workload kind {workload.kind!r} is not supported",
        ),
        (
            not op.has_bias or capabilities.supports_bias,
            "NQ-INF-BIAS",
            "bias is not supported",
        ),
        (
            op.bias_dtype is None or op.bias_dtype in capabilities.scale_dtypes,
            "NQ-INF-BIAS-DTYPE",
            f"bias dtype {op.bias_dtype!r} is not supported",
        ),
        (
            op.outlier_count == 0 or capabilities.supports_outliers,
            "NQ-INF-OUTLIERS",
            "salient outliers are not supported",
        ),
        (
            op.outlier_value_dtype is None or op.outlier_value_dtype in capabilities.outlier_value_dtypes,
            "NQ-INF-OUTLIER-DTYPE",
            f"outlier dtype {op.outlier_value_dtype!r} is not supported",
        ),
        (
            op.patch_value_dtype is None or op.patch_value_dtype in capabilities.patch_value_dtypes,
            "NQ-INF-PATCH",
            f"low-rank patch dtype {op.patch_value_dtype!r} is not supported",
        ),
        (
            not workload.deterministic or capabilities.supports_deterministic,
            "NQ-INF-DETERMINISM",
            "deterministic execution is not supported",
        ),
        (
            op.in_features % capabilities.in_feature_alignment == 0,
            "NQ-INF-IN-ALIGNMENT",
            f"input features must be aligned to {capabilities.in_feature_alignment}",
        ),
        (
            op.out_features % capabilities.out_feature_alignment == 0,
            "NQ-INF-OUT-ALIGNMENT",
            f"output features must be aligned to {capabilities.out_feature_alignment}",
        ),
        (
            op.rank % capabilities.rank_alignment == 0,
            "NQ-INF-RANK-ALIGNMENT",
            f"rank must be aligned to {capabilities.rank_alignment}",
        ),
        (
            capabilities.maximum_batch_size is None or workload.batch_size <= capabilities.maximum_batch_size,
            "NQ-INF-BATCH",
            f"batch size {workload.batch_size} exceeds the backend maximum",
        ),
        (
            capabilities.maximum_token_count is None or workload.token_count <= capabilities.maximum_token_count,
            "NQ-INF-TOKENS",
            f"token count {workload.token_count} exceeds the backend maximum",
        ),
    )
    for accepted, code, reason in checks:
        if not accepted:
            return SupportResult.rejected(code, reason)
    return SupportResult.accepted()
