"""Deployment-only NanoQuant runtime surface.

This package must remain importable without calibration, datasets, factorization,
optimizers, or experiment orchestration.
"""

from nanoquant.runtime.artifact import (
    DESCRIPTOR_SCHEMA_VERSION,
    LOGICAL_FORMAT_VERSION,
    LogicalArtifactError,
    LogicalModelManifest,
    OpenLogicalArtifact,
    RuntimeModelMetadata,
    open_logical_artifact,
    write_logical_artifact,
)
from nanoquant.runtime.backend import (
    BackendCapabilities,
    PreparedLayer,
    QuantizedLinearSpec,
    RuntimeBackend,
    SupportResult,
    WorkloadSpec,
)
from nanoquant.runtime.logical import LogicalLayerState
from nanoquant.runtime.planning import (
    BackendPlan,
    BackendPlanningError,
    LayerDispatch,
    PreparedDispatch,
    plan_backends,
    prepare_plan,
)
from nanoquant.runtime.reference import DenseReferenceBackend, FactorizedReferenceBackend

__all__ = [
    "BackendCapabilities",
    "BackendPlan",
    "BackendPlanningError",
    "DenseReferenceBackend",
    "FactorizedReferenceBackend",
    "LayerDispatch",
    "LOGICAL_FORMAT_VERSION",
    "DESCRIPTOR_SCHEMA_VERSION",
    "LogicalArtifactError",
    "LogicalLayerState",
    "LogicalModelManifest",
    "OpenLogicalArtifact",
    "PreparedDispatch",
    "PreparedLayer",
    "QuantizedLinearSpec",
    "RuntimeBackend",
    "RuntimeModelMetadata",
    "SupportResult",
    "WorkloadSpec",
    "plan_backends",
    "open_logical_artifact",
    "prepare_plan",
    "write_logical_artifact",
]
