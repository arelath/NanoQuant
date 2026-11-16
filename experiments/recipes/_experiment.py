"""Required numbered-experiment identity and derived repository layout."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Generic, TypeVar

from nanoquant.compression_benchmark_workflow import CompressionBenchmarkExperiment
from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    HuggingFaceUploadConfig,
)
from nanoquant.compression_quality_workflow import CompressionQualityExperiment
from nanoquant.config.schema import IntentConfig, RunConfig
from nanoquant.domain.constants import BackendType
from nanoquant.infrastructure.hf_mmlu import MMLU_QUESTIONS, MMLU_REPO, MMLU_REVISION, mmlu_files
from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.quality_evaluation import QualityEvaluationRequest
from nanoquant.quality_evaluation_workflow import QualityEvaluationExperiment
from nanoquant.rank_expansion_experiment import RankExpansionExperiment
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

_SAFE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_NUMBERED_NAME = re.compile(r"^\d{3}-")
_LLAMA_CPP_ROOT = Path(
    os.environ.get("NANOQUANT_LLAMA_CPP_ROOT", r"D:\dev\research\llama.cpp")
)


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
    def experiment(cls, reference: ExperimentIdentity | ExperimentRef) -> BaselineRef:
        return cls(BaselineKind.EXPERIMENT, reference.canonical_name)

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
class ExperimentRef:
    """Typed reference to another standardized experiment without importing its launcher."""

    number: int
    name: str

    def __post_init__(self) -> None:
        _validate_number_and_name(self.number, self.name)

    @property
    def number_token(self) -> str:
        return f"{self.number:03d}"

    @property
    def canonical_name(self) -> str:
        return f"{self.number_token}-{self.name}"

    @property
    def run_output(self) -> Path:
        return Path("evidence") / self.number_token / self.canonical_name

    @property
    def packed_output(self) -> Path:
        return Path("outputs") / self.number_token / "packed"

    @property
    def published_quality_output(self) -> Path:
        return Path("Results") / self.number_token / f"{self.canonical_name}-quality.json"


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
        return self.results_root / f"{release_name}-nanoquant.gguf"

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
    output_tensor_type: str = "q8_0"

    def __post_init__(self) -> None:
        if self.release_name is not None:
            _require_safe_name(self.release_name, "release name")
        if not self.runtime_family.strip():
            raise ValueError("runtime family is required")


def _require_safe_name(value: str, field: str) -> None:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"experiment {field} must be lowercase kebab-case")


def _validate_number_and_name(number: int, name: str) -> None:
    if isinstance(number, bool) or number < 1 or number > 999:
        raise ValueError("experiment number must be between 1 and 999")
    if _NUMBERED_NAME.match(name):
        raise ValueError("experiment name must not include its numeric prefix")
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError("experiment name must be one lowercase kebab-case path component")


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
        output_tensor_type=policy.output_tensor_type,
        huggingface=policy.huggingface,
    )


def define_compression_benchmark_experiment(
    identity: ExperimentIdentity,
    template: RunConfig,
    *,
    export: CompressionExportPolicy | None = None,
) -> ExperimentDefinition[CompressionBenchmarkExperiment]:
    layout = ExperimentLayout(identity)
    config = _materialize_config(template, identity, layout)
    export_policy = CompressionExportPolicy() if export is None else export
    workflow = CompressionBenchmarkExperiment(
        export=_export_recipe(config, layout, export_policy),
        benchmark_output=layout.benchmark_output,
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
    export: CompressionExportPolicy | None = None,
    maximum_wddm_shared_gib: float | None = None,
    restore_completed_blocks: bool = True,
    interrupt_after_block_commits: int | None = None,
    quality_backend: str | None = BackendType.FACTORIZED.value,
    large_model_guards: bool = False,
    llamacpp_quality: bool = False,
    llamacpp_quality_parallel: int = 4,
    reasoning_sequence_length_override: int | None = None,
    task_limit: int = 200,
    allow_relocated_completed_run: bool = False,
    include_mmlu: bool = False,
) -> ExperimentDefinition[CompressionQualityExperiment]:
    layout = ExperimentLayout(identity)
    config = _materialize_config(template, identity, layout)
    export_policy = CompressionExportPolicy() if export is None else export
    workflow = CompressionQualityExperiment(
        export=_export_recipe(config, layout, export_policy),
        summary_output=layout.summary_output,
        quality_output=layout.quality_output,
        quality_markdown_output=layout.quality_markdown_output,
        maximum_wddm_shared_gib=maximum_wddm_shared_gib,
        restore_completed_blocks=restore_completed_blocks,
        interrupt_after_block_commits=interrupt_after_block_commits,
        quality_backend=quality_backend,
        large_model_guards=large_model_guards,
        llamacpp_quality=llamacpp_quality,
        llama_cpp_root=_LLAMA_CPP_ROOT if llamacpp_quality else None,
        llamacpp_quality_parallel=llamacpp_quality_parallel,
        reasoning_sequence_length_override=reasoning_sequence_length_override,
        task_limit=task_limit,
        allow_relocated_completed_run=allow_relocated_completed_run,
        include_mmlu=include_mmlu,
    )
    return ExperimentDefinition(identity, config, workflow, layout)


def define_rank_expansion_experiment(
    identity: ExperimentIdentity,
    template: RunConfig,
    *,
    parent: ExperimentRef,
    release_name: str,
    bit_multiplier: float = 1.30,
    layer_suffix: str = "self_attn.v_proj",
    maximum_wddm_shared_gib: float = 0.75,
) -> ExperimentDefinition[RankExpansionExperiment]:
    layout = ExperimentLayout(identity)
    config = _materialize_config(template, identity, layout)
    _require_safe_name(release_name, "release name")
    workflow = RankExpansionExperiment(
        parent_run=parent.run_output,
        source_packed=parent.packed_output,
        output_packed=layout.packed_output,
        checkpoint_output=layout.checkpoint_output,
        gguf_output=layout.gguf_output(release_name),
        expansion_report=layout.expansion_report,
        quality_output=layout.quality_output,
        quality_markdown_output=layout.quality_markdown_output,
        summary_output=layout.summary_output,
        baseline_quality=parent.published_quality_output,
        llama_cpp_root=_LLAMA_CPP_ROOT,
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


ExperimentWorkflow = (
    CompressionBenchmarkExperiment
    | CompressionQualityExperiment
    | QualityEvaluationExperiment
    | RankExpansionExperiment
)


def run_experiment(
    definition: ExperimentDefinition[ExperimentWorkflow],
    *,
    launcher_path: str | Path,
) -> int:
    """Dispatch a declarative experiment definition to its workflow runner."""

    workflow = definition.workflow
    if isinstance(workflow, CompressionBenchmarkExperiment):
        from nanoquant.compression_benchmark_workflow import run_compression_benchmark_experiment

        return run_compression_benchmark_experiment(
            definition.config, workflow, launcher_path=launcher_path
        )
    if isinstance(workflow, CompressionQualityExperiment):
        from nanoquant.compression_quality_workflow import run_compression_quality_experiment

        return run_compression_quality_experiment(
            definition.config, workflow, launcher_path=launcher_path
        )
    if isinstance(workflow, QualityEvaluationExperiment):
        from nanoquant.quality_evaluation_workflow import run_quality_evaluation_experiment

        return run_quality_evaluation_experiment(
            definition.config, workflow, launcher_path=launcher_path
        )
    if isinstance(workflow, RankExpansionExperiment):
        from nanoquant.rank_expansion_experiment import run_rank_expansion_experiment

        return run_rank_expansion_experiment(
            definition.config, workflow, launcher_path=launcher_path
        )
    raise TypeError(f"unsupported experiment workflow: {type(workflow).__name__}")


def experiment_main(
    module_name: str,
    launcher_path: str | Path,
    definition: ExperimentDefinition[ExperimentWorkflow],
) -> None:
    """Run a numbered launcher only when its module is the process entry point."""

    if module_name == "__main__":
        raise SystemExit(run_experiment(definition, launcher_path=launcher_path))


def experiment_callable_main(module_name: str, runner: Callable[[], int]) -> None:
    """Run a specialized launcher without repeating the module-entry guard."""

    if module_name == "__main__":
        raise SystemExit(runner())


def experiment_pre_post_callable_main(
    module_name: str,
    launcher_path: str | Path,
    runner: Callable[[], int],
    *,
    uniform_control: bool = False,
) -> None:
    """Run the normal post-KD workflow, then export/score its immutable pre-KD commits.

    Previews remain JSON and never execute either model. Repeated real launches
    validate/reuse exports and refresh the pre-KD benchmark against the current
    post-KD report, rather than trusting a stale quality file.
    """
    if module_name != "__main__":
        return
    launcher = Path(launcher_path).resolve()
    root = launcher.parent.parent
    command = [sys.executable, str(root / "tools/export_pre_global_tuning.py"), str(launcher)]
    if uniform_control:
        command.append("--uniform-control")
    preview = any(flag in sys.argv[1:] for flag in ("--plan-only", "--dry-run"))
    if preview:
        captured = StringIO()
        with redirect_stdout(captured):
            result = runner()
        if result:
            print(captured.getvalue(), end="")
            raise SystemExit(result)
        post_plan = json.loads(captured.getvalue())
        pre_plan = subprocess.run(
            command + ["--plan-only"], cwd=root, check=True, capture_output=True, text=True,
        )
        post_plan["pre_global_tuning"] = json.loads(pre_plan.stdout)
        print(json.dumps(post_plan, indent=2))
        raise SystemExit(0)
    result = runner()
    if result == 0:
        result = subprocess.run(command + ["--refresh-quality"], cwd=root, check=False).returncode
    raise SystemExit(result)


def run_imatrix_attribution_experiment(
    experiment: ExperimentDefinition[CompressionQualityExperiment],
    *,
    launcher_path: str,
    profile_options: SelfMeasuredD2ProfileOptions,
    sweep: Path,
    fits_sha256: str,
    attribution: Path,
) -> int:
    """Include the pinned attribution and MMLU inputs in Experiment 080 previews."""
    def compression() -> int:
        return run_self_measured_d2_experiment(
            experiment, launcher_path=launcher_path, profile_options=profile_options, reuse_maximum_rank_control=True,
        )

    arguments = set(sys.argv[1:])
    plan_only = "--plan-only" in arguments
    if arguments & {"-h", "--help"} or not arguments & {"--plan-only", "--dry-run"}:
        return compression()
    root = Path(launcher_path).resolve().parent.parent
    imatrix = experiment.config.calibration.imatrix
    gguf = experiment.workflow.export.gguf_output
    if imatrix is None or imatrix.sha256 is None or gguf is None:
        raise ValueError("imatrix attribution requires a pinned imatrix and a canonical GGUF export")
    command = [
        sys.executable, str(root / "tools/predict_imatrix_weighted_kl.py"), str(sweep),
        "--predict-only", "--imatrix", imatrix.path, "--imatrix-sha256", imatrix.sha256,
        "--fits-sha256", fits_sha256, "--gguf", str(root / gguf), "--output", str(attribution),
    ]
    captured = StringIO()
    with redirect_stdout(captured):
        result = compression()
    if result != 0:
        print(captured.getvalue(), end="")
        return result
    payload = json.loads(captured.getvalue())

    attribution_plan: dict[str, object] = {
        "markdown": str(attribution), "json": str(attribution.with_suffix(".json")), "command": command,
    }
    payload["attribution"] = attribution_plan
    if experiment.workflow.include_mmlu:
        payload["mmlu"] = {
            "dataset": MMLU_REPO, "revision": MMLU_REVISION, "test_examples": MMLU_QUESTIONS, "few_shot": 5,
        }
    if not plan_only:
        checked = subprocess.run(command + ["--check-inputs"], cwd=root, check=True, capture_output=True, text=True)
        inputs = json.loads(checked.stdout)
        teacher = inputs["protocol"]["teacher"]
        model = experiment.config.model
        if (teacher["source"], teacher["revision"]) != (model.source, model.revision):
            raise ValueError("sensitivity teacher does not match the recipe's model")
        attribution_plan["validated_inputs"] = inputs
        if experiment.workflow.include_mmlu:
            payload["mmlu"]["split_sha256"] = {
                k: hash_file(v) for k, v in mmlu_files(local_files_only=experiment.workflow.local_files_only).items()
            }
    print(json.dumps(payload, indent=2))
    return 0


__all__ = [
    "BaselineKind",
    "BaselineRef",
    "CompressionExportPolicy",
    "ExperimentDefinition",
    "ExperimentIdentity",
    "ExperimentLayout",
    "ExperimentRef",
    "ExperimentWorkflow",
    "define_compression_benchmark_experiment",
    "define_compression_quality_experiment",
    "define_quality_evaluation_experiment",
    "define_rank_expansion_experiment",
    "experiment_callable_main",
    "experiment_main",
    "experiment_pre_post_callable_main",
    "run_experiment",
    "validate_experiment_registry",
]
