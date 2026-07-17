"""Required numbered-experiment identity and derived repository layout."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

from nanoquant.compression_benchmark_workflow import CompressionBenchmarkExperiment
from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    HuggingFaceUploadConfig,
)
from nanoquant.compression_quality_workflow import CompressionQualityExperiment
from nanoquant.config.schema import IntentConfig, RunConfig
from nanoquant.quality_evaluation import QualityEvaluationRequest
from nanoquant.quality_evaluation_workflow import QualityEvaluationExperiment
from nanoquant.rank_expansion_experiment import RankExpansionExperiment

_SAFE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_NUMBERED_NAME = re.compile(r"^\d{3}-")
_LLAMA_CPP_ROOT = Path(r"D:\dev\research\llama.cpp")


class BaselineKind(str, Enum):
    """How an experiment's comparison baseline is identified."""

    EXPERIMENT = "experiment"
    EXTERNAL = "external"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class BaselineRef:
    """Explicit baseline choice serialized into the existing intent contract."""

    kind: BaselineKind
    label: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("experiment baseline label or reason is required")

    @classmethod
    def external(cls, label: str) -> BaselineRef:
        return cls(BaselineKind.EXTERNAL, label)

    @classmethod
    def experiment(cls, identity: ExperimentIdentity) -> BaselineRef:
        return cls(BaselineKind.EXPERIMENT, identity.canonical_name)

    @classmethod
    def none(cls, reason: str) -> BaselineRef:
        return cls(BaselineKind.NONE, reason)

    def intent_value(self) -> str:
        if self.kind is BaselineKind.NONE:
            return f"none:{self.label}"
        return self.label


@dataclass(frozen=True, slots=True)
class ExperimentIdentity:
    """Complete required intent for one active numbered experiment."""

    number: int
    name: str
    purpose: str
    hypothesis: str
    baseline: BaselineRef
    tags: tuple[str, ...]
    owner: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.number, bool) or self.number < 1 or self.number > 999:
            raise ValueError("experiment number must be between 1 and 999")
        if _NUMBERED_NAME.match(self.name):
            raise ValueError("experiment name must not include its numeric prefix")
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError("experiment name must be one lowercase kebab-case path component")
        if not self.purpose.strip():
            raise ValueError("experiment purpose is required")
        if not self.hypothesis.strip():
            raise ValueError("experiment hypothesis is required")
        if not self.tags or any(not tag.strip() for tag in self.tags):
            raise ValueError("at least one non-empty experiment tag is required")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("experiment tags must be unique")
        if self.owner is not None and not self.owner.strip():
            raise ValueError("experiment owner must be non-empty when provided")

    @property
    def number_token(self) -> str:
        return f"{self.number:03d}"

    @property
    def canonical_name(self) -> str:
        return f"{self.number_token}-{self.name}"

    def to_config(self) -> IntentConfig:
        return IntentConfig(
            experiment_number=self.number,
            name=self.canonical_name,
            purpose=self.purpose,
            hypothesis=self.hypothesis,
            baseline_run=self.baseline.intent_value(),
            owner=self.owner,
            tags=self.tags,
        )


@dataclass(frozen=True, slots=True)
class ExperimentLayout:
    """All repository-owned locations derived from one experiment identity."""

    identity: ExperimentIdentity

    @property
    def run_root(self) -> Path:
        return Path("evidence") / self.identity.number_token

    @property
    def run_output(self) -> Path:
        return self.run_root / self.identity.canonical_name

    @property
    def outputs_root(self) -> Path:
        return Path("outputs") / self.identity.number_token

    @property
    def results_root(self) -> Path:
        return Path("Results") / self.identity.number_token

    @property
    def summary_output(self) -> Path:
        return self.outputs_root / f"{self.identity.canonical_name}-summary.json"

    @property
    def quality_output(self) -> Path:
        return self.outputs_root / f"{self.identity.canonical_name}-quality.json"

    @property
    def quality_markdown_output(self) -> Path:
        return self.results_root / f"{self.identity.canonical_name}-quality.md"

    @property
    def benchmark_output(self) -> Path:
        return self.outputs_root / f"{self.identity.canonical_name}-benchmark.json"

    @property
    def expansion_report(self) -> Path:
        return self.outputs_root / f"{self.identity.canonical_name}-expansion.json"

    @property
    def logical_output(self) -> Path:
        return self.outputs_root / "logical"

    @property
    def packed_output(self) -> Path:
        return self.outputs_root / "packed"

    @property
    def checkpoint_output(self) -> Path:
        return self.outputs_root / "llamacpp-checkpoint"

    def gguf_output(self, release_name: str) -> Path:
        _require_safe_name(release_name, "release name")
        return self.outputs_root / f"{release_name}-nanoquant.gguf"

    def published_quality_output(self) -> Path:
        return self.results_root / self.quality_output.name


WorkflowT = TypeVar("WorkflowT", covariant=True)


@dataclass(frozen=True, slots=True)
class ExperimentDefinition(Generic[WorkflowT]):
    """Canonical active experiment: identity, materialized config, workflow, and layout."""

    identity: ExperimentIdentity
    config: RunConfig
    workflow: WorkflowT
    layout: ExperimentLayout

    def __post_init__(self) -> None:
        if self.layout.identity != self.identity:
            raise ValueError("experiment layout identity does not match definition identity")
        if self.config.intent != self.identity.to_config():
            raise ValueError("experiment config intent does not match definition identity")
        if Path(self.config.output.run_root) != self.layout.run_root:
            raise ValueError("experiment config run root does not match derived layout")


@dataclass(frozen=True, slots=True)
class CompressionExportPolicy:
    """Semantic export choices; material destinations come from the experiment layout."""

    release_name: str | None = None
    llama_cpp_root: Path = _LLAMA_CPP_ROOT
    runtime_family: str = "gemma3"
    token_embedding_type: str = "q8_0"
    huggingface: HuggingFaceUploadConfig | None = None

    def __post_init__(self) -> None:
        if self.release_name is not None:
            _require_safe_name(self.release_name, "release name")
        if not self.runtime_family.strip():
            raise ValueError("runtime family is required")


def _require_safe_name(value: str, field: str) -> None:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"experiment {field} must be lowercase kebab-case")


def _default_release_name(config: RunConfig) -> str:
    value = config.model.source.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1].lower()
    _require_safe_name(value, "model-derived release name")
    return value


def _materialize_config(template: RunConfig, identity: ExperimentIdentity, layout: ExperimentLayout) -> RunConfig:
    return replace(
        template,
        intent=identity.to_config(),
        output=replace(template.output, run_root=layout.run_root.as_posix()),
    )


def _export_recipe(
    config: RunConfig,
    layout: ExperimentLayout,
    policy: CompressionExportPolicy,
) -> CompressionExportRecipe:
    release_name = policy.release_name or _default_release_name(config)
    return CompressionExportRecipe(
        logical_output=layout.logical_output,
        packed_output=layout.packed_output,
        checkpoint_output=layout.checkpoint_output,
        gguf_output=layout.gguf_output(release_name),
        llama_cpp_root=policy.llama_cpp_root,
        runtime_family=policy.runtime_family,
        token_embedding_type=policy.token_embedding_type,
        huggingface=policy.huggingface,
    )


def define_compression_benchmark_experiment(
    identity: ExperimentIdentity,
    template: RunConfig,
    *,
    expected_blocks: int = 26,
    export: CompressionExportPolicy = CompressionExportPolicy(),
) -> ExperimentDefinition[CompressionBenchmarkExperiment]:
    layout = ExperimentLayout(identity)
    config = _materialize_config(template, identity, layout)
    workflow = CompressionBenchmarkExperiment(
        export=_export_recipe(config, layout, export),
        benchmark_output=layout.benchmark_output,
        expected_blocks=expected_blocks,
    )
    return ExperimentDefinition(identity, config, workflow, layout)


def define_quality_evaluation_experiment(
    identity: ExperimentIdentity,
    template: RunConfig,
    request: QualityEvaluationRequest,
    *,
    resolve_model_from_config: bool = False,
) -> ExperimentDefinition[QualityEvaluationExperiment]:
    layout = ExperimentLayout(identity)
    config = _materialize_config(template, identity, layout)
    workflow = QualityEvaluationExperiment(
        request,
        layout.quality_output,
        resolve_model_from_config=resolve_model_from_config,
        markdown_path=layout.quality_markdown_output,
    )
    return ExperimentDefinition(identity, config, workflow, layout)


def define_compression_quality_experiment(
    identity: ExperimentIdentity,
    template: RunConfig,
    *,
    expected_blocks: int,
    export: CompressionExportPolicy = CompressionExportPolicy(),
    maximum_wddm_shared_gib: float | None = None,
    restore_completed_blocks: bool = True,
    quality_backend: str = "factorized",
    large_model_guards: bool = False,
) -> ExperimentDefinition[CompressionQualityExperiment]:
    layout = ExperimentLayout(identity)
    config = _materialize_config(template, identity, layout)
    workflow = CompressionQualityExperiment(
        export=_export_recipe(config, layout, export),
        summary_output=layout.summary_output,
        quality_output=layout.quality_output,
        quality_markdown_output=layout.quality_markdown_output,
        expected_blocks=expected_blocks,
        maximum_wddm_shared_gib=maximum_wddm_shared_gib,
        restore_completed_blocks=restore_completed_blocks,
        quality_backend=quality_backend,
        large_model_guards=large_model_guards,
    )
    return ExperimentDefinition(identity, config, workflow, layout)


def define_rank_expansion_experiment(
    identity: ExperimentIdentity,
    template: RunConfig,
    *,
    parent: ExperimentDefinition[CompressionQualityExperiment],
    release_name: str,
    bit_multiplier: float = 1.30,
    layer_suffix: str = "self_attn.v_proj",
    expected_blocks: int = 34,
    maximum_wddm_shared_gib: float = 0.75,
) -> ExperimentDefinition[RankExpansionExperiment]:
    layout = ExperimentLayout(identity)
    config = _materialize_config(template, identity, layout)
    _require_safe_name(release_name, "release name")
    workflow = RankExpansionExperiment(
        parent_run=parent.layout.run_output,
        source_packed=parent.layout.packed_output,
        output_packed=layout.packed_output,
        checkpoint_output=layout.checkpoint_output,
        gguf_output=layout.gguf_output(release_name),
        expansion_report=layout.expansion_report,
        quality_output=layout.quality_output,
        quality_markdown_output=layout.quality_markdown_output,
        summary_output=layout.summary_output,
        baseline_quality=parent.layout.published_quality_output(),
        llama_cpp_root=_LLAMA_CPP_ROOT,
        expected_blocks=expected_blocks,
        layer_suffix=layer_suffix,
        bit_multiplier=bit_multiplier,
        maximum_wddm_shared_gib=maximum_wddm_shared_gib,
    )
    return ExperimentDefinition(identity, config, workflow, layout)


def validate_experiment_registry(
    definitions: tuple[ExperimentDefinition[object], ...],
) -> None:
    numbers = tuple(item.identity.number for item in definitions)
    names = tuple(item.identity.canonical_name for item in definitions)
    if len(set(numbers)) != len(numbers):
        raise ValueError("active experiment numbers must be unique")
    if len(set(names)) != len(names):
        raise ValueError("active experiment names must be unique")


__all__ = [
    "BaselineKind",
    "BaselineRef",
    "CompressionExportPolicy",
    "ExperimentDefinition",
    "ExperimentIdentity",
    "ExperimentLayout",
    "define_compression_benchmark_experiment",
    "define_compression_quality_experiment",
    "define_quality_evaluation_experiment",
    "define_rank_expansion_experiment",
    "validate_experiment_registry",
]
