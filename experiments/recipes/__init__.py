"""Canonical active experiment definitions and unnumbered templates."""

from nanoquant.compression_export_workflow import HuggingFaceUploadConfig

from ._experiment import (
    BaselineKind,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentDefinition,
    ExperimentIdentity,
    ExperimentLayout,
    validate_experiment_registry,
)
from .base_compression import (
    BASE_COMPRESSION_TEMPLATE,
    GEMMA_3_1B_PARITY_TEMPLATE,
    LARGE_MODEL_COMPRESSION_TEMPLATE,
)
from .experiment001 import EXPERIMENT_001
from .experiment002 import EXPERIMENT_002
from .experiment003 import EXPERIMENT_003
from .experiment004 import EXPERIMENT_004
from .experiment005 import EXPERIMENT_005
from .experiment006 import EXPERIMENT_006
from .experiment007 import EXPERIMENT_007
from .experiment008 import EXPERIMENT_008

ALL_EXPERIMENTS = (
    EXPERIMENT_001,
    EXPERIMENT_002,
    EXPERIMENT_003,
    EXPERIMENT_004,
    EXPERIMENT_005,
    EXPERIMENT_006,
    EXPERIMENT_007,
    EXPERIMENT_008,
)
validate_experiment_registry(ALL_EXPERIMENTS)

__all__ = [
    "ALL_EXPERIMENTS",
    "BASE_COMPRESSION_TEMPLATE",
    "BaselineKind",
    "BaselineRef",
    "CompressionExportPolicy",
    "ExperimentDefinition",
    "ExperimentIdentity",
    "ExperimentLayout",
    "GEMMA_3_1B_PARITY_TEMPLATE",
    "HuggingFaceUploadConfig",
    "LARGE_MODEL_COMPRESSION_TEMPLATE",
    "EXPERIMENT_001",
    "EXPERIMENT_002",
    "EXPERIMENT_003",
    "EXPERIMENT_004",
    "EXPERIMENT_005",
    "EXPERIMENT_006",
    "EXPERIMENT_007",
    "EXPERIMENT_008",
]
