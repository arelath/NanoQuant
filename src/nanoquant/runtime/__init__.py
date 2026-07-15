"""Deployment-only NanoQuant runtime surface.

This package must remain importable without calibration, datasets, factorization,
optimizers, or experiment orchestration.
"""

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
    "LogicalLayerState",
    "PreparedDispatch",
    "PreparedLayer",
    "QuantizedLinearSpec",
    "RuntimeBackend",
    "SupportResult",
    "WorkloadSpec",
    "plan_backends",
    "prepare_plan",
]
