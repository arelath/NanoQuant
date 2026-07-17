"""Canonical active experiment definitions and unnumbered templates."""

from nanoquant.compression_export_workflow import HuggingFaceUploadConfig

from ._experiment import (
    BaselineKind,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentDefinition,
    ExperimentIdentity,
    ExperimentLayout,
    ExperimentRef,
    define_compression_benchmark_experiment,
    define_compression_quality_experiment,
    define_quality_evaluation_experiment,
    define_rank_expansion_experiment,
)
from .base_compression import (
    BASE_COMPRESSION_TEMPLATE,
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    GEMMA_3_4B_MODEL_REVISION,
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    GEMMA_3_270M_MODEL_REVISION,
    LARGE_MODEL_COMPRESSION_TEMPLATE,
)

__all__ = [
    "BASE_COMPRESSION_TEMPLATE",
    "BaselineKind",
    "BaselineRef",
    "CompressionExportPolicy",
    "ExperimentDefinition",
    "ExperimentIdentity",
    "ExperimentLayout",
    "ExperimentRef",
    "GEMMA_3_270M_COMPRESSION_TEMPLATE",
    "GEMMA_3_270M_MODEL_REVISION",
    "GEMMA_3_4B_COMPRESSION_TEMPLATE",
    "GEMMA_3_4B_MODEL_REVISION",
    "HuggingFaceUploadConfig",
    "LARGE_MODEL_COMPRESSION_TEMPLATE",
    "define_compression_benchmark_experiment",
    "define_compression_quality_experiment",
    "define_quality_evaluation_experiment",
    "define_rank_expansion_experiment",
]
