"""Interactive, non-numbered compression launcher and persisted settings."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml
from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import validate_repo_id  # type: ignore[attr-defined]

from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    complete_deferred_huggingface_upload,
    execute_complete_compression,
)
from nanoquant.compression_quality_workflow import (
    CompressionQualityExperiment,
    run_compression_quality_experiment,
)
from nanoquant.config.codec import from_dict, semantic_hash, to_dict
from nanoquant.config.schema import IntentConfig, RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.environment import load_repository_dotenv
from nanoquant.infrastructure.huggingface_model_card import load_huggingface_model_card_metadata
from nanoquant.infrastructure.huggingface_upload import (
    HuggingFaceUploadConfig,
    huggingface_upload_summary,
)
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file
from nanoquant.infrastructure.model_adapters import decoder_block_count_from_config
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_run_artifacts,
)
from nanoquant.infrastructure.resolved_model_config import resolve_model_config
from nanoquant.infrastructure.run_session import open_run_event_append_session
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    resolve_resident_experiment_inputs,
)

INTERACTIVE_SETTINGS_SCHEMA_VERSION = 2
INTERACTIVE_SETTINGS_KIND = "interactive_compression"
INTERACTIVE_SETTINGS_NAME = "settings.yaml"
INTERACTIVE_COMPLETION_NAME = "interactive-completion.json"
_SAFE_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

InputFn = Callable[[str], str]
WriteFn = Callable[[str], None]
ExecuteFn = Callable[[Path, Path], int]


@dataclass(frozen=True, slots=True)
class RecommendedModel:
    """One promoted model choice and its complete reusable recipe."""

    family: str
    family_label: str
    variant: str
    variant_label: str
    source: str
    revision: str
    runtime_family: str
    release_name: str
    profile_id: str
    evidence: tuple[str, ...]
    template: RunConfig
    default_target_bpw: float = 1.0
    default_family: bool = False
    default_variant: bool = False
    maximum_wddm_shared_gib: float | None = 0.75
    restore_completed_blocks: bool = False
    quality_backend: str | None = None
    large_model_guards: bool = False
    llamacpp_quality: bool = True
    llamacpp_quality_parallel: int = 4

    def __post_init__(self) -> None:
        for value, label in (
            (self.family, "family"),
            (self.variant, "variant"),
            (self.release_name, "release name"),
            (self.profile_id, "profile ID"),
        ):
            if not _SAFE_SLUG.fullmatch(value):
                raise ValueError(f"recommended model {label} must be lowercase kebab-case")
        if not self.family_label.strip() or not self.variant_label.strip():
            raise ValueError("recommended model labels are required")
        if not self.source.strip() or not self.revision.strip():
            raise ValueError("recommended model source and revision are required")
        if not self.runtime_family.strip() or not self.evidence:
            raise ValueError("recommended model runtime family and evidence are required")
        if not math.isfinite(self.default_target_bpw) or self.default_target_bpw <= 0:
            raise ValueError("recommended target BPW must be finite and positive")
        if self.template.model.source != self.source or self.template.model.revision != self.revision:
            raise ValueError("recommended model template identity does not match its catalog identity")
        if self.quality_backend is None and not self.llamacpp_quality:
            raise ValueError("recommended quality requires a candidate backend")
        if self.llamacpp_quality_parallel <= 0:
            raise ValueError("llama.cpp quality parallelism must be positive")

    @property
    def profile_hash(self) -> str:
        return semantic_hash(
            {
                "family": self.family,
                "variant": self.variant,
                "source": self.source,
                "revision": self.revision,
                "runtime_family": self.runtime_family,
                "release_name": self.release_name,
                "profile_id": self.profile_id,
                "evidence": self.evidence,
                "template": self.template,
                "default_target_bpw": self.default_target_bpw,
                "maximum_wddm_shared_gib": self.maximum_wddm_shared_gib,
                "restore_completed_blocks": self.restore_completed_blocks,
                "quality_backend": self.quality_backend,
                "large_model_guards": self.large_model_guards,
                "llamacpp_quality": self.llamacpp_quality,
                "llamacpp_quality_parallel": self.llamacpp_quality_parallel,
            }
        )


@dataclass(frozen=True, slots=True)
class InteractiveSelection:
    model_family: str
    model_variant: str
    model_input: str
    target_bpw: float
    quality_requested: bool
    huggingface_upload_requested: bool
    huggingface: HuggingFaceUploadConfig | None = None

    def __post_init__(self) -> None:
        if not self.model_family.strip() or not self.model_variant.strip() or not self.model_input.strip():
            raise ValueError("interactive model selection is incomplete")
        if not math.isfinite(self.target_bpw) or self.target_bpw <= 0:
            raise ValueError("interactive target BPW must be finite and positive")
        if self.huggingface_upload_requested != (self.huggingface is not None):
            raise ValueError("interactive Hugging Face selection is inconsistent")


@dataclass(frozen=True, slots=True)
class InteractiveProfile:
    id: str
    schema_version: int
    sha256: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InteractiveResolvedSource:
    model: str
    revision: str
    tokenizer_revision: str
    architecture: str
    block_count: int

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.revision.strip() or not self.tokenizer_revision.strip():
            raise ValueError("interactive resolved model identity is incomplete")
        if not self.architecture.strip():
            raise ValueError("interactive resolved model architecture is required")
        if self.block_count <= 0:
            raise ValueError("interactive resolved model block count must be positive")


@dataclass(frozen=True, slots=True)
class InteractiveWorkflow:
    maximum_wddm_shared_gib: float | None
    restore_completed_blocks: bool
    quality_backend: str | None
    large_model_guards: bool
    llamacpp_quality: bool
    llamacpp_quality_parallel: int


@dataclass(frozen=True, slots=True)
class InteractivePaths:
    run_output: Path
    outputs_root: Path
    results_root: Path
    logical_output: Path
    packed_output: Path
    checkpoint_output: Path
    gguf_output: Path
    summary_output: Path
    quality_output: Path
    quality_markdown_output: Path


@dataclass(frozen=True, slots=True)
class InteractiveExport:
    llama_cpp_root: Path
    runtime_family: str
    token_embedding_type: str = "q8_0"
    output_tensor_type: str = "q8_0"


@dataclass(frozen=True, slots=True)
class InteractiveSettings:
    schema_version: int
    kind: str
    created_at: str
    selection: InteractiveSelection
    profile: InteractiveProfile
    resolved_source: InteractiveResolvedSource
    run_config: RunConfig
    workflow: InteractiveWorkflow
    paths: InteractivePaths
    export: InteractiveExport

    def __post_init__(self) -> None:
        if self.schema_version != INTERACTIVE_SETTINGS_SCHEMA_VERSION:
            raise ValueError(f"unsupported interactive settings schema: {self.schema_version}")
        if self.kind != INTERACTIVE_SETTINGS_KIND:
            raise ValueError(f"unsupported interactive settings kind: {self.kind!r}")
        if self.run_config.intent.experiment_number is not None:
            raise ValueError("interactive settings cannot own an experiment number")
        if self.run_config.intent.name != self.paths.run_output.name:
            raise ValueError("interactive run name and output path disagree")
        if self.run_config.model.source != self.resolved_source.model:
            raise ValueError("interactive model source disagrees with resolved source")
        if str(self.run_config.model.revision) != self.resolved_source.revision:
            raise ValueError("interactive model revision disagrees with resolved source")
        if self.run_config.allocation.target_bpw != self.selection.target_bpw:
            raise ValueError("interactive target BPW disagrees with resolved config")
        expected_results = Path("Results") / "interactive" / self.run_config.intent.name
        if self.paths.results_root != expected_results:
            raise ValueError("interactive Results path is not canonical")
        if self.paths.gguf_output.parent != self.paths.results_root:
            raise ValueError("interactive GGUF is not written directly to its Results directory")


@dataclass(frozen=True, slots=True)
class InteractiveRunRecord:
    settings_path: Path
    settings: InteractiveSettings
    settings_hash: str
    status: str
    completed_blocks: int
    active_owner: dict[str, object] | None


def _settings_body(settings: InteractiveSettings) -> dict[str, Any]:
    return cast(dict[str, Any], to_dict(settings))


def settings_hash(settings: InteractiveSettings) -> str:
    return semantic_hash(_settings_body(settings))


def write_interactive_settings(path: str | Path, settings: InteractiveSettings) -> str:
    """Atomically persist immutable settings and return their semantic hash."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"interactive settings already exist: {destination}")
    digest = settings_hash(settings)
    payload = {"settings_hash": digest, **_settings_body(settings)}
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    atomic_write_text(destination, rendered)
    return digest


def load_interactive_settings(path: str | Path) -> tuple[InteractiveSettings, str]:
    """Load strict settings and reject manual or torn edits."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"interactive settings root must be an object: {source}")
    expected = payload.pop("settings_hash", None)
    if not isinstance(expected, str):
        raise ValueError(f"interactive settings hash is missing: {source}")
    observed = semantic_hash(payload)
    if observed != expected:
        raise ValueError(
            f"interactive settings hash mismatch: {source}; expected={expected}, observed={observed}"
        )
    schema_version = payload.get("schema_version")
    if schema_version == 1:
        raw_source = payload.get("resolved_source")
        if not isinstance(raw_source, dict) or type(raw_source.get("expected_blocks")) is not int:
            raise ValueError(f"legacy interactive settings have no valid block count: {source}")
        migrated_source = dict(raw_source)
        migrated_source["block_count"] = migrated_source.pop("expected_blocks")
        payload["resolved_source"] = migrated_source
        payload["schema_version"] = INTERACTIVE_SETTINGS_SCHEMA_VERSION
    settings = from_dict(InteractiveSettings, cast(dict[str, Any], payload), path="settings")
    return settings, observed


def _manifest_status(run_output: Path) -> str:
    completion = run_output / INTERACTIVE_COMPLETION_NAME
    if completion.is_file():
        return "completed"
    manifest = run_output / "manifest.json"
    if not manifest.is_file():
        return "created"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return str(payload.get("status", "unknown"))
    except (OSError, json.JSONDecodeError, TypeError):
        return "corrupt"


def _completed_blocks(run_output: Path) -> int:
    journal = run_output / "state" / "journal.jsonl"
    if not journal.is_file():
        return 0
    completed: set[int] = set()
    try:
        with journal.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("kind") == "block" and type(payload.get("block")) is int:
                    completed.add(int(payload["block"]))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0
    return len(completed)


def _active_owner(run_output: Path) -> dict[str, object] | None:
    lease = run_output / ".active-lease.json"
    if not lease.is_file():
        return None
    try:
        payload = json.loads(lease.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"unreadable": True}
    if not isinstance(payload, dict):
        return {"unreadable": True}
    hostname = str(payload.get("hostname", ""))
    raw_pid = payload.get("pid")
    pid = raw_pid if type(raw_pid) is int else -1
    if hostname == socket.gethostname() and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            return cast(dict[str, object], payload)
        except OSError:
            return None
    return cast(dict[str, object], payload)


def discover_interactive_runs(repository_root: str | Path) -> tuple[InteractiveRunRecord, ...]:
    """Discover valid settings files newest first."""

    root = Path(repository_root).resolve() / "evidence" / "interactive"
    if not root.is_dir():
        return ()
    records: list[InteractiveRunRecord] = []
    for child in root.iterdir():
        settings_path = child / INTERACTIVE_SETTINGS_NAME
        if not settings_path.is_file():
            continue
        try:
            settings, digest = load_interactive_settings(settings_path)
        except (OSError, ValueError, TypeError):
            continue
        records.append(
            InteractiveRunRecord(
                settings_path,
                settings,
                digest,
                _manifest_status(child),
                _completed_blocks(child),
                _active_owner(child),
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda item: (item.settings.created_at, item.settings.run_config.intent.name),
            reverse=True,
        )
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not normalized:
        raise ValueError(f"cannot derive a safe slug from {value!r}")
    return normalized


def _new_run_name(model: RecommendedModel, now: datetime | None = None) -> str:
    instant = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    timestamp = instant.strftime("%Y%m%dT%H%M%SZ").lower()
    return f"{timestamp}-{model.release_name}-{secrets.token_hex(3)}"


def create_interactive_settings(
    model: RecommendedModel,
    *,
    target_bpw: float,
    quality_requested: bool,
    huggingface: HuggingFaceUploadConfig | None,
    llama_cpp_root: str | Path,
    now: datetime | None = None,
    run_name: str | None = None,
) -> InteractiveSettings:
    """Materialize one immutable non-numbered run from a promoted model profile."""

    if not math.isfinite(target_bpw) or target_bpw <= 0:
        raise ValueError("target BPW must be finite and positive")
    name = _new_run_name(model, now) if run_name is None else run_name
    if not _SAFE_SLUG.fullmatch(name):
        raise ValueError("interactive run name must be lowercase kebab-case")
    resolved_source = _resolved_source_for_model(model)
    run_root = Path("evidence") / "interactive"
    run_output = run_root / name
    outputs_root = Path("outputs") / "interactive" / name
    results_root = Path("Results") / "interactive" / name
    config = replace(
        model.template,
        model=replace(
            model.template.model,
            source=resolved_source.model,
            revision=resolved_source.revision,
            tokenizer_source=None,
            tokenizer_revision=resolved_source.tokenizer_revision,
        ),
        intent=IntentConfig(
            experiment_number=None,
            name=name,
            purpose=f"Compress {model.source} through the interactive production workflow.",
            hypothesis=f"Apply promoted profile {model.profile_id} at {target_bpw:g} target BPW.",
            baseline_run=f"none:interactive profile {model.profile_id}",
            tags=("interactive-compression", model.family, model.variant, model.profile_id),
        ),
        allocation=replace(model.template.allocation, target_bpw=target_bpw),
        output=replace(model.template.output, run_root=run_root.as_posix()),
    )
    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    paths = InteractivePaths(
        run_output=run_output,
        outputs_root=outputs_root,
        results_root=results_root,
        logical_output=outputs_root / "logical",
        packed_output=outputs_root / "packed",
        checkpoint_output=outputs_root / "llamacpp-checkpoint",
        gguf_output=results_root / f"{model.release_name}-nanoquant.gguf",
        summary_output=outputs_root / "summary.json",
        quality_output=outputs_root / "quality.json",
        quality_markdown_output=results_root / "quality.md",
    )
    created = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    return InteractiveSettings(
        INTERACTIVE_SETTINGS_SCHEMA_VERSION,
        INTERACTIVE_SETTINGS_KIND,
        created.isoformat(),
        InteractiveSelection(
            model.family,
            model.variant,
            model.source,
            target_bpw,
            quality_requested,
            huggingface is not None,
            huggingface,
        ),
        InteractiveProfile(model.profile_id, 1, model.profile_hash, model.evidence),
        resolved_source,
        config,
        InteractiveWorkflow(
            model.maximum_wddm_shared_gib,
            model.restore_completed_blocks,
            model.quality_backend,
            model.large_model_guards,
            model.llamacpp_quality,
            model.llamacpp_quality_parallel,
        ),
        paths,
        InteractiveExport(Path(llama_cpp_root), model.runtime_family),
    )


def _repository_path(repository_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _export_recipe(
    settings: InteractiveSettings,
    *,
    include_huggingface: bool = True,
) -> CompressionExportRecipe:
    return CompressionExportRecipe(
        settings.paths.logical_output,
        settings.paths.packed_output,
        settings.paths.checkpoint_output,
        settings.paths.gguf_output,
        settings.export.llama_cpp_root,
        runtime_family=settings.export.runtime_family,
        token_embedding_type=settings.export.token_embedding_type,
        output_tensor_type=settings.export.output_tensor_type,
        huggingface=(
            settings.selection.huggingface
            if include_huggingface
            else None
        ),
    )


def _quality_experiment(settings: InteractiveSettings) -> CompressionQualityExperiment:
    workflow = settings.workflow
    return CompressionQualityExperiment(
        export=_export_recipe(settings),
        summary_output=settings.paths.summary_output,
        quality_output=settings.paths.quality_output,
        quality_markdown_output=settings.paths.quality_markdown_output,
        maximum_wddm_shared_gib=workflow.maximum_wddm_shared_gib,
        restore_completed_blocks=workflow.restore_completed_blocks,
        quality_backend=workflow.quality_backend,
        large_model_guards=workflow.large_model_guards,
        llamacpp_quality=workflow.llamacpp_quality,
        llama_cpp_root=settings.export.llama_cpp_root if workflow.llamacpp_quality else None,
        llamacpp_quality_parallel=workflow.llamacpp_quality_parallel,
    )


def _publishable_export_members(
    exports: Any,
    summary_output: Path,
    run_output: Path,
) -> tuple[PublishableArtifact, ...]:
    gguf = exports.gguf
    members: list[PublishableArtifact] = [
        PublishableArtifact(gguf.output, PublishableArtifactKind.MODEL),
        PublishableArtifact(exports.summary_output, PublishableArtifactKind.STATISTICS),
        PublishableArtifact(
            gguf.output.with_suffix(gguf.output.suffix + ".export.json"),
            PublishableArtifactKind.STATISTICS,
        ),
        PublishableArtifact(summary_output, PublishableArtifactKind.STATISTICS),
    ]
    if gguf.mmproj is not None:
        members.extend(
            (
                PublishableArtifact(gguf.mmproj.output, PublishableArtifactKind.MODEL),
                PublishableArtifact(
                    gguf.mmproj.output.with_suffix(gguf.mmproj.output.suffix + ".export.json"),
                    PublishableArtifactKind.STATISTICS,
                ),
            )
        )
    if exports.huggingface is not None:
        members.append(
            PublishableArtifact(
                exports.huggingface.receipt_output,
                PublishableArtifactKind.STATISTICS,
            )
        )
    weight_report = run_output / "weight-errors.md"
    if weight_report.is_file():
        members.append(PublishableArtifact(weight_report, PublishableArtifactKind.REPORT))
    return tuple(members)


def _execute_without_quality(
    settings: InteractiveSettings,
    *,
    repository_root: Path,
    launcher_path: Path,
) -> dict[str, Any]:
    config = settings.run_config
    inputs = resolve_resident_experiment_inputs(config, launcher_path=launcher_path)
    maximum_shared = settings.workflow.maximum_wddm_shared_gib
    complete = execute_complete_compression(
        config,
        inputs,
        _export_recipe(settings, include_huggingface=False),
        options=ResidentExecutionOptions(
            restore_completed_blocks=settings.workflow.restore_completed_blocks,
            maximum_wddm_shared_bytes=(
                None if maximum_shared is None else int(maximum_shared * 2**30)
            ),
        ),
    )
    exports = complete.exports
    results_root = _repository_path(repository_root, settings.paths.results_root)
    body = results_root / "quality-not-run.md"
    atomic_write_text(
        body,
        (
            f"# {settings.selection.model_input} NanoQuant\n\n"
            "The quality benchmark was not requested for this interactive run. "
            "The GGUF passed structural, hash, packing, and export validation only.\n"
        ),
    )
    if settings.selection.huggingface is not None:
        metadata = load_huggingface_model_card_metadata(
            config.model.source,
            str(config.model.revision),
            inputs.snapshot,
        )
        with open_run_event_append_session(
            inputs.output,
            observability=config.observability,
        ) as upload_events:
            exports = complete_deferred_huggingface_upload(
                exports,
                settings.selection.huggingface,
                ((body, "README.md"),),
                model_card_metadata=metadata,
                events=upload_events,
            )
    quantization = complete.workflow.quantization
    summary_output = _repository_path(repository_root, settings.paths.summary_output)
    payload = {
        "schema_version": 1,
        "passed": True,
        "settings_hash": settings_hash(settings),
        "quality": {"requested": False, "passed": None},
        "compression": {
            "run_output": str(inputs.output.resolve()),
            "blocks": len(quantization.inventory.blocks),
            "effective_bpw": quantization.frozen_model.effective_bpw,
            "peak_device_bytes": quantization.peak_device_bytes,
            "peak_host_bytes": quantization.peak_host_bytes,
            "artifact_bytes": quantization.artifact_bytes,
            "elapsed_seconds": quantization.elapsed_seconds,
            "reused_commit_count": quantization.reused_commit_count,
        },
        "exports": {
            "gguf": {
                "output": str(exports.gguf.output),
                "bytes": exports.gguf.bytes,
                "sha256": exports.gguf.sha256,
                "reused": exports.gguf.reused,
            },
            "huggingface": (
                None
                if exports.huggingface is None
                else huggingface_upload_summary(exports.huggingface)
            ),
        },
    }
    atomic_write_json(summary_output, payload)
    publishable = (
        *_publishable_export_members(exports, summary_output, inputs.output),
        PublishableArtifact(body, PublishableArtifactKind.REPORT),
    )
    model_card = exports.gguf.output.with_suffix(".model-card.md")
    if model_card.is_file():
        publishable = (
            *publishable,
            PublishableArtifact(model_card, PublishableArtifactKind.REPORT),
        )
    publish_run_artifacts(repository_root, config.intent.name, publishable)
    return payload


def _completion_payload(
    settings: InteractiveSettings,
    settings_digest: str,
    summary: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    exports = summary.get("exports")
    if not isinstance(exports, dict) or not isinstance(exports.get("gguf"), dict):
        raise ValueError("interactive summary contains no validated GGUF result")
    gguf = cast(dict[str, Any], exports["gguf"])
    huggingface = exports.get("huggingface")
    return {
        "schema_version": 1,
        "settings_hash": settings_digest,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(summary.get("passed")),
        "summary": str(_repository_path(repository_root, settings.paths.summary_output)),
        "gguf": {
            "path": str(_repository_path(repository_root, settings.paths.gguf_output)),
            "bytes": int(gguf["bytes"]),
            "sha256": str(gguf["sha256"]),
        },
        "huggingface_receipt": (
            None
            if not isinstance(huggingface, dict)
            else str(huggingface.get("receipt"))
        ),
    }


def _validate_completion(
    payload: dict[str, Any],
    *,
    settings_digest: str,
) -> tuple[Path, Path] | None:
    if payload.get("schema_version") != 1 or payload.get("settings_hash") != settings_digest:
        raise ValueError("interactive completion belongs to different settings")
    summary_path = Path(str(payload.get("summary", "")))
    raw_gguf = payload.get("gguf")
    if not isinstance(raw_gguf, dict):
        raise ValueError("interactive completion has no GGUF identity")
    gguf_path = Path(str(raw_gguf.get("path", "")))
    try:
        expected_bytes = int(raw_gguf["bytes"])
        expected_hash = str(raw_gguf["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("interactive completion GGUF identity is invalid") from exc
    if not summary_path.is_file() or not gguf_path.is_file():
        return None
    if gguf_path.stat().st_size != expected_bytes or hash_file(gguf_path) != expected_hash:
        raise ValueError("completed interactive GGUF no longer matches its validated identity")
    receipt = payload.get("huggingface_receipt")
    if receipt is not None and not Path(str(receipt)).is_file():
        return None
    return summary_path, gguf_path


def execute_interactive_run(
    settings_path: str | Path,
    launcher_path: str | Path,
) -> int:
    """Execute or continue one persisted interactive run."""

    source = Path(settings_path).resolve()
    launcher = Path(launcher_path).resolve()
    repository_root = launcher.parent.parent
    load_repository_dotenv(repository_root)
    settings, digest = load_interactive_settings(source)
    expected_source = _repository_path(repository_root, settings.paths.run_output) / INTERACTIVE_SETTINGS_NAME
    if source != expected_source:
        raise ValueError(
            f"interactive settings are outside their resolved run directory: {source} != {expected_source}"
        )
    completion_path = expected_source.parent / INTERACTIVE_COMPLETION_NAME
    if completion_path.is_file():
        payload = cast(
            dict[str, Any],
            json.loads(completion_path.read_text(encoding="utf-8")),
        )
        completed = _validate_completion(payload, settings_digest=digest)
        if completed is not None:
            summary_path, _gguf_path = completed
            print(f"Interactive run is already complete: {summary_path}", flush=True)
            return 0
    if settings.selection.quality_requested:
        run_compression_quality_experiment(
            settings.run_config,
            _quality_experiment(settings),
            launcher_path=launcher,
        )
        summary_path = _repository_path(repository_root, settings.paths.summary_output)
        summary = cast(dict[str, Any], json.loads(summary_path.read_text(encoding="utf-8")))
        if not bool(summary.get("passed")):
            raise RuntimeError(
                f"quality benchmark failed; local outputs were retained: {summary_path}"
            )
    else:
        summary = _execute_without_quality(
            settings,
            repository_root=repository_root,
            launcher_path=launcher,
        )
    atomic_write_json(
        completion_path,
        _completion_payload(settings, digest, summary, repository_root),
    )
    print(
        f"Interactive compression complete: "
        f"{_repository_path(repository_root, settings.paths.gguf_output)}",
        flush=True,
    )
    return 0


class InteractiveConsole:
    def __init__(self, input_fn: InputFn = input, write: WriteFn = print) -> None:
        self.input = input_fn
        self.write = write

    def choose(self, prompt: str, count: int, *, default: int = 1) -> int:
        if count <= 0 or default < 1 or default > count:
            raise ValueError("interactive menu bounds are invalid")
        while True:
            raw = self.input(f"{prompt} [{default}]: ").strip()
            if not raw:
                return default
            try:
                selected = int(raw)
            except ValueError:
                self.write(f"Enter a number from 1 to {count}.")
                continue
            if 1 <= selected <= count:
                return selected
            self.write(f"Enter a number from 1 to {count}.")

    def yes_no(self, prompt: str, *, default: bool) -> bool:
        marker = "Y/n" if default else "y/N"
        while True:
            raw = self.input(f"{prompt} [{marker}]: ").strip().lower()
            if not raw:
                return default
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            self.write("Enter yes or no.")

    def positive_float(self, prompt: str, *, default: float) -> float:
        while True:
            raw = self.input(f"{prompt} [{default:.2f}]: ").strip()
            if not raw:
                return default
            try:
                value = float(raw)
            except ValueError:
                self.write("Enter a positive finite number.")
                continue
            if math.isfinite(value) and value > 0:
                return value
            self.write("Enter a positive finite number.")


def _families(catalog: Iterable[RecommendedModel]) -> tuple[tuple[str, str], ...]:
    ordered: dict[str, str] = {}
    for model in catalog:
        current = ordered.get(model.family)
        if current is None:
            ordered[model.family] = model.family_label
        elif current != model.family_label:
            raise ValueError(f"catalog family metadata disagrees for {model.family}")
    return tuple(ordered.items())


def _variants(
    catalog: Iterable[RecommendedModel],
    family: str,
) -> tuple[RecommendedModel, ...]:
    return tuple(model for model in catalog if model.family == family)


def _default_family(
    catalog: tuple[RecommendedModel, ...],
    previous: InteractiveRunRecord | None,
) -> str:
    families = {model.family for model in catalog}
    if previous is not None and previous.settings.selection.model_family in families:
        return previous.settings.selection.model_family
    defaults = {model.family for model in catalog if model.default_family}
    if len(defaults) != 1:
        raise ValueError("recommended catalog must declare exactly one default family")
    return next(iter(defaults))


def _default_variant(
    variants: tuple[RecommendedModel, ...],
    previous: InteractiveRunRecord | None,
) -> str:
    available = {model.variant for model in variants}
    if (
        previous is not None
        and previous.settings.selection.model_family == variants[0].family
        and previous.settings.selection.model_variant in available
    ):
        return previous.settings.selection.model_variant
    defaults = [model.variant for model in variants if model.default_variant]
    if len(defaults) != 1:
        raise ValueError(
            f"recommended family {variants[0].family} must declare exactly one default variant"
        )
    return defaults[0]


def _parse_custom_model(text: str) -> tuple[str, str | None]:
    value = text.strip()
    if not value:
        raise ValueError("model ID or local snapshot is required")
    path = Path(value)
    if path.exists():
        return str(path.resolve()), None
    if "@" in value:
        source, revision = value.rsplit("@", 1)
        if not source.strip() or not revision.strip():
            raise ValueError("custom remote model must use owner/model@revision")
        return source.strip(), revision.strip()
    return value, None


def _model_config(source: str, revision: str | None) -> tuple[dict[str, Any], str]:
    resolved = resolve_model_config(source, revision)
    return resolved.values, resolved.revision


def _architecture_and_block_count(config: dict[str, Any]) -> tuple[str, int]:
    architecture = str(config.get("model_type", ""))
    if not architecture:
        raise ValueError("model config does not declare an architecture")
    return architecture, decoder_block_count_from_config(cast(dict[str, object], config))


def _family_for_model(source: str, architecture: str) -> str:
    if architecture == "qwen3":
        return "qwen3"
    if architecture in {"gemma3", "gemma3_text"}:
        return "gemma3"
    if architecture == "llama":
        return "llama3-2" if "3.2" in source.lower() else "llama3"
    raise ValueError(f"no promoted interactive profile supports model_type={architecture!r}")


def _resolved_source_for_model(model: RecommendedModel) -> InteractiveResolvedSource:
    config_payload, revision = _model_config(model.source, model.revision)
    architecture, block_count = _architecture_and_block_count(config_payload)
    family = _family_for_model(model.source, architecture)
    if family != model.family:
        raise ValueError(
            f"resolved model family differs from promoted profile: {family!r} != {model.family!r}"
        )
    return InteractiveResolvedSource(
        model.source,
        revision,
        revision,
        architecture,
        block_count,
    )


def resolve_custom_model(
    text: str,
    catalog: tuple[RecommendedModel, ...],
) -> RecommendedModel:
    """Resolve a custom model and inherit the nearest compatible promoted profile."""

    source, requested_revision = _parse_custom_model(text)
    config_payload, revision = _model_config(source, requested_revision)
    architecture, block_count = _architecture_and_block_count(config_payload)
    family = _family_for_model(source, architecture)
    candidates = _variants(catalog, family)
    if not candidates:
        raise ValueError(f"no promoted interactive profile supports family {family!r}")
    parent = min(
        candidates,
        key=lambda model: abs(_resolved_source_for_model(model).block_count - block_count),
    )
    model_config = replace(
        parent.template.model,
        source=source,
        revision=revision,
        tokenizer_source=None,
        tokenizer_revision=revision,
    )
    release_name = _slug(Path(source).name if Path(source).exists() else source.rsplit("/", 1)[-1])
    variant = f"custom-{release_name}"
    return replace(
        parent,
        variant=variant,
        variant_label=source,
        source=source,
        revision=revision,
        release_name=release_name,
        profile_id=f"{parent.profile_id}-custom",
        evidence=(*parent.evidence, f"custom-source:{source}@{revision}"),
        template=replace(parent.template, model=model_config),
        default_family=False,
        default_variant=False,
    )


def authenticated_huggingface_owner() -> str:
    token = os.environ.get("HF_TOKEN", "").strip() or (get_token() or "").strip()
    if not token:
        raise RuntimeError(
            "Hugging Face upload requires HF_TOKEN or a cached `huggingface-cli login` session"
        )
    identity = HfApi(token=token).whoami()
    name = identity.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("authenticated Hugging Face account has no usable name")
    return name.strip()


def _choose_model(
    console: InteractiveConsole,
    catalog: tuple[RecommendedModel, ...],
    previous: InteractiveRunRecord | None,
    custom_resolver: Callable[[str, tuple[RecommendedModel, ...]], RecommendedModel],
) -> RecommendedModel:
    while True:
        families = _families(catalog)
        default_family = _default_family(catalog, previous)
        default_family_index = next(
            index for index, (family, _label) in enumerate(families, start=1)
            if family == default_family
        )
        console.write("Choose a model family:")
        for index, (_family, label) in enumerate(families, start=1):
            console.write(f"  {index}. {label}")
        custom_index = len(families) + 1
        console.write(f"  {custom_index}. Enter another Hugging Face model ID or local snapshot")
        selected = console.choose("Selection", custom_index, default=default_family_index)
        if selected == custom_index:
            while True:
                raw = console.input("Model ID or local snapshot: ").strip()
                try:
                    return custom_resolver(raw, catalog)
                except (OSError, RuntimeError, ValueError) as exc:
                    console.write(f"Cannot resolve model: {exc}")
                    if not console.yes_no("Try another custom model?", default=True):
                        break
            continue
        family = families[selected - 1][0]
        variants = _variants(catalog, family)
        default_variant = _default_variant(variants, previous)
        default_variant_index = next(
            index for index, model in enumerate(variants, start=1)
            if model.variant == default_variant
        )
        console.write(f"Choose a {families[selected - 1][1]} model:")
        for index, model in enumerate(variants, start=1):
            console.write(f"  {index}. {model.variant_label}")
        back = len(variants) + 1
        console.write(f"  {back}. Back to model families")
        selected_variant = console.choose("Selection", back, default=default_variant_index)
        if selected_variant != back:
            return variants[selected_variant - 1]


def _huggingface_selection(
    console: InteractiveConsole,
    model: RecommendedModel,
    *,
    target_bpw: float,
    quality_requested: bool,
    owner_resolver: Callable[[], str],
) -> HuggingFaceUploadConfig | None:
    if not console.yes_no(
        "Upload validated outputs to Hugging Face when local stages finish?",
        default=False,
    ):
        return None
    if not quality_requested:
        console.write(
            "Warning: this artifact will be published without a quality benchmark. "
            "The model card will state that quality was not evaluated."
        )
        if not console.yes_no("Continue with upload?", default=False):
            return None
    owner = owner_resolver()
    default_repo = f"{owner}/{model.release_name}-nanoquant-GGUF"
    while True:
        repo_id = console.input(f"Repository [{default_repo}]: ").strip() or default_repo
        try:
            validate_repo_id(repo_id)
            break
        except ValueError as exc:
            console.write(f"Invalid Hugging Face repository: {exc}")
    console.write("Visibility:")
    console.write("  1. Private")
    console.write("  2. Public")
    private = console.choose("Selection", 2, default=1) == 1
    default_message = f"Publish NanoQuant {model.variant_label} at {target_bpw:.2f} BPW"
    message = console.input(f"Commit message [{default_message}]: ").strip() or default_message
    return HuggingFaceUploadConfig(repo_id, private=private, commit_message=message)


def _render_previous(console: InteractiveConsole, record: InteractiveRunRecord) -> None:
    selection = record.settings.selection
    console.write("Previous run")
    console.write(f"  Model:        {selection.model_input}")
    console.write(f"  Target:       {selection.target_bpw:.2f} BPW")
    console.write(f"  Profile:      {record.settings.profile.id}")
    console.write(f"  Quality:      {'yes' if selection.quality_requested else 'no'}")
    console.write(
        f"  Hugging Face: {'yes' if selection.huggingface_upload_requested else 'no'}"
    )
    console.write(
        f"  Status:       {record.status} — {record.completed_blocks} of "
        f"{record.settings.resolved_source.block_count} blocks complete"
    )
    if record.active_owner is not None:
        console.write(f"  Active owner: {json.dumps(record.active_owner, sort_keys=True)}")


def _choose_existing(
    console: InteractiveConsole,
    records: tuple[InteractiveRunRecord, ...],
) -> InteractiveRunRecord | None:
    console.write("Existing interactive runs:")
    for index, record in enumerate(records, start=1):
        console.write(
            f"  {index}. {record.settings.selection.model_input} — {record.status} — "
            f"{record.settings.created_at}"
        )
    console.write(f"  {len(records) + 1}. Back")
    selected = console.choose("Selection", len(records) + 1, default=1)
    return None if selected == len(records) + 1 else records[selected - 1]


def _select_start_action(
    console: InteractiveConsole,
    records: tuple[InteractiveRunRecord, ...],
) -> InteractiveRunRecord | None | str:
    if not records:
        return "new"
    selected_record = records[0]
    while True:
        _render_previous(console, selected_record)
        console.write("What would you like to do?")
        console.write("  1. Continue previous run")
        console.write("  2. Start a new run")
        console.write("  3. Show previous settings and progress")
        console.write("  4. Choose another existing run")
        console.write("  5. Exit")
        choice = console.choose("Selection", 5, default=1)
        if choice == 1:
            if selected_record.active_owner is not None:
                console.write("That run is active in another process; a second worker was not started.")
                continue
            return selected_record
        if choice == 2:
            return "new"
        if choice == 3:
            console.write(
                yaml.safe_dump(
                    {"settings_hash": selected_record.settings_hash, **_settings_body(selected_record.settings)},
                    sort_keys=False,
                ).rstrip()
            )
            continue
        if choice == 4:
            selected = _choose_existing(console, records)
            if selected is not None:
                selected_record = selected
            continue
        return None


def _persist_new_settings(
    repository_root: Path,
    settings: InteractiveSettings,
) -> Path:
    settings_path = _repository_path(repository_root, settings.paths.run_output) / INTERACTIVE_SETTINGS_NAME
    settings_path.parent.mkdir(parents=True, exist_ok=False)
    write_interactive_settings(settings_path, settings)
    return settings_path


def run_interactive_launcher(
    repository_root: str | Path,
    launcher_path: str | Path,
    catalog: Iterable[RecommendedModel],
    *,
    input_fn: InputFn = input,
    write: WriteFn = print,
    execute: ExecuteFn = execute_interactive_run,
    custom_resolver: Callable[
        [str, tuple[RecommendedModel, ...]], RecommendedModel
    ] = resolve_custom_model,
    owner_resolver: Callable[[], str] = authenticated_huggingface_owner,
) -> int:
    """Run the terminal interaction and dispatch a persisted settings file."""

    root = Path(repository_root).resolve()
    launcher = Path(launcher_path).resolve()
    load_repository_dotenv(root)
    promoted = tuple(catalog)
    if not promoted:
        raise ValueError("interactive compression catalog is empty")
    console = InteractiveConsole(input_fn, write)
    console.write("NanoQuant interactive compression")
    records = discover_interactive_runs(root)
    action = _select_start_action(console, records)
    if action is None:
        return 0
    if isinstance(action, InteractiveRunRecord):
        return execute(action.settings_path, launcher)
    previous = records[0] if records else None
    model = _choose_model(console, promoted, previous, custom_resolver)
    target_bpw = console.positive_float(
        "Target bits per parameter (BPW)",
        default=model.default_target_bpw,
    )
    quality_requested = console.yes_no(
        "Run the recommended quality benchmark after export?",
        default=True,
    )
    huggingface = _huggingface_selection(
        console,
        model,
        target_bpw=target_bpw,
        quality_requested=quality_requested,
        owner_resolver=owner_resolver,
    )
    llama_cpp_root = Path(
        os.environ.get("NANOQUANT_LLAMA_CPP_ROOT", r"D:\dev\research\llama.cpp")
    )
    settings = create_interactive_settings(
        model,
        target_bpw=target_bpw,
        quality_requested=quality_requested,
        huggingface=huggingface,
        llama_cpp_root=llama_cpp_root,
    )
    console.write("Ready to create run")
    console.write(f"  Model:               {settings.selection.model_input}")
    console.write(f"  Revision:            {settings.resolved_source.revision}")
    console.write(f"  Recommended profile: {settings.profile.id}")
    console.write(f"  Target BPW:          {settings.selection.target_bpw:.2f}")
    console.write(
        f"  Quality benchmark:   {'yes' if settings.selection.quality_requested else 'no'}"
    )
    console.write(
        f"  Hugging Face upload: {'yes' if settings.selection.huggingface_upload_requested else 'no'}"
    )
    console.write(f"  Run directory:       {settings.paths.run_output}")
    console.write(f"  Results directory:   {settings.paths.results_root}")
    console.write("  1. Start")
    console.write("  2. Show full resolved settings")
    console.write("  3. Cancel")
    while True:
        choice = console.choose("Selection", 3, default=1)
        if choice == 2:
            console.write(yaml.safe_dump(_settings_body(settings), sort_keys=False).rstrip())
            continue
        if choice == 3:
            return 0
        break
    settings_path = _persist_new_settings(root, settings)
    console.write(f"Settings written: {settings_path}")
    return execute(settings_path, launcher)


__all__ = [
    "INTERACTIVE_COMPLETION_NAME",
    "INTERACTIVE_SETTINGS_KIND",
    "INTERACTIVE_SETTINGS_NAME",
    "INTERACTIVE_SETTINGS_SCHEMA_VERSION",
    "InteractiveConsole",
    "InteractiveExport",
    "InteractivePaths",
    "InteractiveProfile",
    "InteractiveResolvedSource",
    "InteractiveRunRecord",
    "InteractiveSelection",
    "InteractiveSettings",
    "InteractiveWorkflow",
    "RecommendedModel",
    "authenticated_huggingface_owner",
    "create_interactive_settings",
    "discover_interactive_runs",
    "execute_interactive_run",
    "load_interactive_settings",
    "resolve_custom_model",
    "run_interactive_launcher",
    "settings_hash",
    "write_interactive_settings",
]
