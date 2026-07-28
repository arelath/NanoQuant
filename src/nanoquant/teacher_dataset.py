"""Reusable, resumable teacher-response dataset generation and publication."""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml
from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.utils import validate_repo_id  # type: ignore[attr-defined]
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.config.codec import from_dict, semantic_hash, to_dict
from nanoquant.config.schema import (
    LLAMACPP_TEACHER_TRACE_IMPLEMENTATION,
    BehaviorSliceConfig,
    DatasetSourceConfig,
    ReasoningMode,
    TeacherTraceGenerationConfig,
)
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.chat_behaviors import chat_behavior_for_snapshot
from nanoquant.infrastructure.environment import load_repository_dotenv
from nanoquant.infrastructure.io_utils import (
    atomic_workspace,
    atomic_write_json,
    atomic_write_text,
    hash_file,
)
from nanoquant.infrastructure.runs import RunLease
from nanoquant.infrastructure.teacher_trace_generation import (
    PreparedTeacherTraces,
    prepare_teacher_traces,
)
from nanoquant.ports.chat_behavior import ChatBehaviorPort

TEACHER_DATASET_SETTINGS_SCHEMA_VERSION = 2
TEACHER_DATASET_ARTIFACT_SCHEMA_VERSION = 1
TEACHER_DATASET_SETTINGS_KIND = "teacher_response_dataset"
TEACHER_DATASET_SETTINGS_NAME = "settings.yaml"
TEACHER_DATASET_DIRECTORY_NAME = "dataset"
TEACHER_DATASET_MANIFEST_NAME = "manifest.json"
TEACHER_DATASET_COMPLETION_NAME = "completion.json"
TEACHER_DATASET_UPLOAD_RECEIPT_NAME = "huggingface-upload.json"
ULTRACHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
ULTRACHAT_SPLIT = "train_sft"
DEFAULT_SAMPLES_PER_MODE = 512

TeacherDatasetProgress = Callable[[str, Mapping[str, object]], None]
TeacherTracePreparer = Callable[..., PreparedTeacherTraces]
SourceRecordLoader = Callable[["TeacherPromptSource", int], Iterable[dict[str, object]]]
SnapshotResolver = Callable[["TeacherModel"], Path]
TokenizerSnapshotResolver = Callable[[str, str], Path]
InputFn = Callable[[str], str]
WriteFn = Callable[[str], None]
ExecuteFn = Callable[[Path], int]

_SAFE_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "human": "user",
    "prompter": "user",
    "system": "system",
    "tool": "tool",
    "user": "user",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not normalized:
        raise ValueError(f"cannot derive a safe slug from {value!r}")
    return normalized


def _source_basename(source: str) -> str:
    value = source.strip().rstrip("/\\")
    if not value:
        raise ValueError("source name is required")
    return re.split(r"[/\\]", value)[-1]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


@dataclass(frozen=True, slots=True)
class TeacherPromptSource:
    """Pinned conversational prompt source with an OpenAI-style message adapter."""

    name: str
    revision: str
    split: str
    subset: str | None = None
    messages_column: str = "messages"
    shuffle_buffer_size: int = 10_000

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.revision.strip() or not self.split.strip():
            raise ValueError("teacher dataset prompt source must be fully pinned")
        if not self.messages_column.strip():
            raise ValueError("teacher dataset messages column is required")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("teacher dataset shuffle buffer must be positive")


@dataclass(frozen=True, slots=True)
class TeacherModel:
    """Pinned model and deterministic generation backend."""

    source: str
    revision: str
    tokenizer_source: str
    tokenizer_revision: str
    gguf_filename: str | None = None
    implementation: str = LLAMACPP_TEACHER_TRACE_IMPLEMENTATION
    device: str = "cuda"

    def __post_init__(self) -> None:
        if (
            not self.source.strip()
            or not self.revision.strip()
            or not self.tokenizer_source.strip()
            or not self.tokenizer_revision.strip()
        ):
            raise ValueError("teacher model and tokenizer sources and revisions are required")
        if self.implementation not in {
            "hf-greedy-qwen3-v1",
            LLAMACPP_TEACHER_TRACE_IMPLEMENTATION,
        }:
            raise ValueError(f"unsupported teacher generation backend: {self.implementation!r}")
        if not self.device.strip():
            raise ValueError("teacher generation device is required")
        if self.gguf_filename is not None:
            relative = Path(self.gguf_filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix.lower() != ".gguf"
            ):
                raise ValueError("teacher GGUF filename must be a safe relative GGUF path")
            if self.implementation != LLAMACPP_TEACHER_TRACE_IMPLEMENTATION:
                raise ValueError("prebuilt GGUF teachers require the llama.cpp backend")


@dataclass(frozen=True, slots=True)
class TeacherDatasetGeneration:
    """Bounded subset and complete-response generation policy."""

    modes: tuple[ReasoningMode, ...] = (
        ReasoningMode.THINKING,
        ReasoningMode.NON_THINKING,
    )
    samples_per_mode: int = DEFAULT_SAMPLES_PER_MODE
    sequence_length: int = 2048
    maximum_new_tokens: int = 1536
    minimum_new_tokens: int = 16
    maximum_attempt_multiplier: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        allowed = {ReasoningMode.THINKING, ReasoningMode.NON_THINKING}
        if not self.modes or any(mode not in allowed for mode in self.modes):
            raise ValueError("teacher dataset modes must be thinking and/or non-thinking")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("teacher dataset modes must be unique")
        if self.samples_per_mode <= 0:
            raise ValueError("teacher dataset samples per mode must be positive")
        if self.sequence_length <= 0:
            raise ValueError("teacher dataset sequence length must be positive")
        if (
            self.minimum_new_tokens <= 0
            or self.maximum_new_tokens < self.minimum_new_tokens
            or self.maximum_attempt_multiplier <= 0
        ):
            raise ValueError("teacher dataset generation limits are invalid")


@dataclass(frozen=True, slots=True)
class TeacherDatasetUpload:
    """Optional Hugging Face dataset-repository publication."""

    repo_id: str
    private: bool = True
    commit_message: str = "Publish generated teacher-response dataset"

    def __post_init__(self) -> None:
        validate_repo_id(self.repo_id)
        if not self.commit_message.strip():
            raise ValueError("teacher dataset upload commit message is required")


@dataclass(frozen=True, slots=True)
class TeacherDatasetSettings:
    """Immutable settings written before generation starts."""

    schema_version: int
    kind: str
    created_at: str
    prompt_source: TeacherPromptSource
    teacher: TeacherModel
    generation: TeacherDatasetGeneration
    upload: TeacherDatasetUpload | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TEACHER_DATASET_SETTINGS_SCHEMA_VERSION:
            raise ValueError(f"unsupported teacher dataset settings schema: {self.schema_version}")
        if self.kind != TEACHER_DATASET_SETTINGS_KIND:
            raise ValueError(f"unsupported teacher dataset settings kind: {self.kind!r}")
        if not self.created_at.strip():
            raise ValueError("teacher dataset creation timestamp is required")


@dataclass(frozen=True, slots=True)
class TeacherCatalogModel:
    family: str
    family_label: str
    source: str
    tokenizer_source: str
    gguf_filename: str
    default: bool = False

    @property
    def label(self) -> str:
        return _source_basename(self.source)


@dataclass(frozen=True, slots=True)
class TeacherDatasetRunRecord:
    settings_path: Path
    settings: TeacherDatasetSettings
    settings_hash: str
    status: str


def teacher_dataset_identity(settings: TeacherDatasetSettings) -> str:
    """Return the semantic dataset identity, excluding publication location."""

    return semantic_hash(
        {
            "schema_version": TEACHER_DATASET_ARTIFACT_SCHEMA_VERSION,
            "producer": "nanoquant-teacher-response-dataset-v1",
            "prompt_source": settings.prompt_source,
            "teacher": settings.teacher,
            "generation": settings.generation,
        }
    )


def _settings_body(settings: TeacherDatasetSettings) -> dict[str, Any]:
    return cast(dict[str, Any], to_dict(settings))


def teacher_dataset_settings_hash(settings: TeacherDatasetSettings) -> str:
    return semantic_hash(_settings_body(settings))


def write_teacher_dataset_settings(
    path: str | Path,
    settings: TeacherDatasetSettings,
) -> str:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"teacher dataset settings already exist: {destination}")
    digest = teacher_dataset_settings_hash(settings)
    atomic_write_text(
        destination,
        yaml.safe_dump(
            {"settings_hash": digest, **_settings_body(settings)},
            sort_keys=False,
            allow_unicode=True,
        ),
    )
    return digest


def load_teacher_dataset_settings(
    path: str | Path,
) -> tuple[TeacherDatasetSettings, str]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"teacher dataset settings root must be an object: {source}")
    expected = payload.pop("settings_hash", None)
    if not isinstance(expected, str):
        raise ValueError(f"teacher dataset settings hash is missing: {source}")
    observed = semantic_hash(payload)
    if observed != expected:
        raise ValueError(
            f"teacher dataset settings hash mismatch: {source}; "
            f"expected={expected}, observed={observed}"
        )
    if payload.get("schema_version") == 1:
        teacher = payload.get("teacher")
        if not isinstance(teacher, dict):
            raise ValueError("legacy teacher dataset settings have no teacher object")
        teacher_source = teacher.get("source")
        teacher_revision = teacher.get("revision")
        if not isinstance(teacher_source, str) or not isinstance(teacher_revision, str):
            raise ValueError("legacy teacher dataset settings have no pinned teacher")
        migrated_teacher = dict(teacher)
        migrated_teacher["tokenizer_source"] = teacher_source
        migrated_teacher["tokenizer_revision"] = teacher_revision
        migrated_teacher["gguf_filename"] = None
        payload["teacher"] = migrated_teacher
        payload["schema_version"] = TEACHER_DATASET_SETTINGS_SCHEMA_VERSION
    settings = from_dict(
        TeacherDatasetSettings,
        cast(dict[str, Any], payload),
        path="teacher_dataset_settings",
    )
    return settings, observed


def new_teacher_dataset_settings(
    *,
    prompt_source: TeacherPromptSource,
    teacher: TeacherModel,
    generation: TeacherDatasetGeneration,
    upload: TeacherDatasetUpload | None,
    created_at: str | None = None,
) -> TeacherDatasetSettings:
    return TeacherDatasetSettings(
        TEACHER_DATASET_SETTINGS_SCHEMA_VERSION,
        TEACHER_DATASET_SETTINGS_KIND,
        created_at or _now(),
        prompt_source,
        teacher,
        generation,
        upload,
    )


def resolve_model_revision(source: str, revision: str | None, *, api: HfApi | None = None) -> str:
    if revision is not None and revision.strip():
        return revision.strip()
    client = api or HfApi()
    resolved = client.model_info(source).sha
    if not isinstance(resolved, str) or not resolved.strip():
        raise RuntimeError(f"Hugging Face returned no revision for teacher model {source!r}")
    return resolved


def resolve_dataset_revision(source: str, revision: str | None, *, api: HfApi | None = None) -> str:
    if revision is not None and revision.strip():
        return revision.strip()
    client = api or HfApi()
    resolved = client.dataset_info(source).sha
    if not isinstance(resolved, str) or not resolved.strip():
        raise RuntimeError(f"Hugging Face returned no revision for prompt dataset {source!r}")
    return resolved


def resolve_gguf_filename(
    source: str,
    revision: str,
    requested: str | None,
    *,
    api: HfApi | None = None,
) -> str | None:
    """Resolve the BF16 GGUF entrypoint for a GGUF repository."""

    if requested is not None and requested.strip():
        resolved_request = requested.strip().replace("\\", "/")
        if source.lower().endswith("-gguf"):
            client = api or HfApi()
            files = set(client.list_repo_files(source, revision=revision))
            if resolved_request not in files:
                raise FileNotFoundError(
                    f"requested teacher GGUF is absent from {source}@{revision}: "
                    f"{resolved_request}"
                )
            match = re.match(
                r"^(.*)-00001-of-(\d{5})\.gguf$",
                resolved_request,
                flags=re.IGNORECASE,
            )
            if match is not None:
                expected = int(match.group(2))
                present = sum(
                    bool(
                        re.fullmatch(
                            re.escape(match.group(1))
                            + rf"-\d{{5}}-of-{re.escape(match.group(2))}\.gguf",
                            value,
                            flags=re.IGNORECASE,
                        )
                    )
                    for value in files
                )
                if present != expected:
                    raise FileNotFoundError(
                        f"requested teacher GGUF has {present} of {expected} repository shards"
                    )
        return resolved_request
    if not source.lower().endswith("-gguf"):
        return None
    client = api or HfApi()
    candidates = [
        value
        for value in client.list_repo_files(source, revision=revision)
        if value.lower().endswith(".gguf")
        and "bf16" in value.lower()
        and not Path(value).name.lower().startswith("mmproj-")
    ]
    entrypoints = [
        value
        for value in candidates
        if "-of-" not in value.lower() or "-00001-of-" in value.lower()
    ]
    if len(entrypoints) != 1:
        raise ValueError(
            f"expected exactly one BF16 GGUF entrypoint in {source}@{revision}; "
            f"found {entrypoints}"
        )
    return entrypoints[0]


def _gguf_download_pattern(filename: str) -> str:
    match = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", filename, flags=re.IGNORECASE)
    if match is None:
        return filename
    return f"{match.group(1)}-*-of-{match.group(2)}.gguf"


def _validate_gguf_snapshot(snapshot: Path, filename: str) -> None:
    entrypoint = snapshot / filename
    if not entrypoint.is_file():
        raise FileNotFoundError(f"pinned Unsloth teacher GGUF is missing: {entrypoint}")
    match = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", filename, flags=re.IGNORECASE)
    if match is None:
        return
    expected = int(match.group(2))
    found = tuple(entrypoint.parent.glob(f"{Path(match.group(1)).name}-*-of-{match.group(2)}.gguf"))
    if len(found) != expected:
        raise FileNotFoundError(
            f"pinned Unsloth teacher GGUF has {len(found)} of {expected} required shards"
        )


def resolve_teacher_snapshot(teacher: TeacherModel) -> Path:
    patterns = None
    if teacher.gguf_filename is not None:
        patterns = [_gguf_download_pattern(teacher.gguf_filename)]
    snapshot = Path(
        snapshot_download(
            repo_id=teacher.source,
            revision=teacher.revision,
            allow_patterns=patterns,
        )
    ).resolve()
    if teacher.gguf_filename is not None:
        _validate_gguf_snapshot(snapshot, teacher.gguf_filename)
    return snapshot


def resolve_tokenizer_snapshot(source: str, revision: str) -> Path:
    return Path(
        snapshot_download(
            repo_id=source,
            revision=revision,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "*.model",
            ],
        )
    ).resolve()


def _normalized_role(value: object) -> str:
    role = str(value or "").strip().lower()
    try:
        return _ROLE_ALIASES[role]
    except KeyError as exc:
        raise ValueError(f"unsupported conversational role {role!r}") from exc


def normalize_prompt_record(
    record: Mapping[str, object],
    *,
    messages_column: str,
) -> dict[str, object]:
    """Normalize common role/content and from/value chat schemas."""

    raw_messages = record.get(messages_column)
    if not isinstance(raw_messages, (list, tuple)) or len(raw_messages) < 2:
        raise ValueError(f"record field {messages_column!r} has no complete conversation")
    messages: list[dict[str, object]] = []
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            raise ValueError("conversation contains a non-object message")
        role = _normalized_role(raw.get("role", raw.get("from")))
        content_value = raw.get("content", raw.get("value", raw.get("text")))
        content = str(content_value or "").strip()
        if not content:
            raise ValueError("conversation contains an empty message")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "assistant":
        raise ValueError("conversation does not end in an assistant response to replace")
    if messages[-2]["role"] != "user":
        raise ValueError("conversation prefix does not end in a user turn")
    source_record_id = record.get("id", record.get("prompt_id"))
    normalized: dict[str, object] = {"messages": messages}
    if source_record_id is not None:
        normalized["source_record_id"] = str(source_record_id)
    return normalized


def load_prompt_records(source: TeacherPromptSource, seed: int) -> Iterable[dict[str, object]]:
    """Stream a deterministic shuffled prompt source without materializing it."""

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "teacher dataset generation requires the evaluation dependencies; "
            "install with `pip install -e .[evaluation]`"
        ) from exc
    positional = (source.name,) if source.subset is None else (source.name, source.subset)
    dataset = load_dataset(
        *positional,
        revision=source.revision,
        split=source.split,
        streaming=True,
    )
    dataset = dataset.shuffle(buffer_size=source.shuffle_buffer_size, seed=seed)

    def records() -> Iterator[dict[str, object]]:
        for raw in cast(Iterable[dict[str, object]], dataset):
            try:
                yield normalize_prompt_record(raw, messages_column=source.messages_column)
            except (TypeError, ValueError) as exc:
                # The trace generator journals this as a rejected source record.
                yield {
                    "messages": None,
                    "source_normalization_error": f"{type(exc).__name__}: {exc}",
                    "source_record": raw,
                }

    return records()


def _behavior_slice(
    settings: TeacherDatasetSettings,
    mode: ReasoningMode,
) -> BehaviorSliceConfig:
    source = settings.prompt_source
    generation = settings.generation
    return BehaviorSliceConfig(
        name=mode.value,
        mode=mode,
        source=DatasetSourceConfig(
            source.name,
            revision=source.revision,
            split=source.split,
            subset=source.subset,
        ),
        record_format="ultrachat_messages",
        target_valid_token_fraction=1.0,
        partition="train",
        teacher_trace_generation=TeacherTraceGenerationConfig(
            implementation=settings.teacher.implementation,
            maximum_new_tokens=generation.maximum_new_tokens,
            minimum_new_tokens=generation.minimum_new_tokens,
            maximum_attempt_multiplier=generation.maximum_attempt_multiplier,
        ),
    )


def _source_adapter_identity(source: TeacherPromptSource) -> str:
    return semantic_hash(
        {
            "implementation": "nanoquant-conversation-normalizer-v1",
            "messages_column": source.messages_column,
            "shuffle_buffer_size": source.shuffle_buffer_size,
        }
    )


def _console_progress(event: str, fields: Mapping[str, object]) -> None:
    print(
        "Teacher dataset: "
        + json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
        flush=True,
    )


def _trace_records(
    run_root: Path,
    traces: PreparedTeacherTraces,
) -> tuple[dict[str, object], ...]:
    artifacts = LocalArtifactStore(run_root / "artifacts")
    descriptor = artifacts.validate(traces.reference.artifact_id)
    if descriptor.artifact_type != "teacher-trace-dataset":
        raise ValueError("teacher trace reference has the wrong artifact type")
    root = artifacts.path_for(traces.reference.artifact_id)
    values = tuple(
        cast(dict[str, object], json.loads(line))
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(values) != len(traces.messages):
        raise ValueError("teacher trace artifact record count changed before publication")
    return values


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for value in values:
            output.write(_canonical_json(value) + "\n")
            count += 1
        output.flush()
        os.fsync(output.fileno())
    return count


def _dataset_card(settings: TeacherDatasetSettings, counts: Mapping[str, int]) -> str:
    configurations: list[dict[str, object]] = []
    mode_paths = [f"data/{mode.value}.jsonl" for mode in settings.generation.modes]
    if len(mode_paths) > 1:
        configurations.append(
            {
                "config_name": "all",
                "default": True,
                "data_files": [{"split": "train", "path": mode_paths}],
            }
        )
    for mode in settings.generation.modes:
        configurations.append(
            {
                "config_name": mode.value,
                "default": len(mode_paths) == 1,
                "data_files": [{"split": "train", "path": f"data/{mode.value}.jsonl"}],
            }
        )
    front_matter = yaml.safe_dump(
        {
            "configs": configurations,
            "tags": ["nanoquant", "distillation", "reasoning", "qwen3"],
        },
        sort_keys=False,
    ).strip()
    mode_summary = ", ".join(
        f"`{mode.value}`: {counts[mode.value]:,}" for mode in settings.generation.modes
    )
    return f"""---
{front_matter}
---

# Teacher-generated conversational responses

This dataset contains complete deterministic responses from
`{settings.teacher.source}` at revision `{settings.teacher.revision}` over a bounded subset of
`{settings.prompt_source.name}` at revision `{settings.prompt_source.revision}`.

The teacher backend loaded `{settings.teacher.gguf_filename or "the pinned safetensors checkpoint"}`. Prompt
rendering used `{settings.teacher.tokenizer_source}` at revision
`{settings.teacher.tokenizer_revision}`.

The source dataset's final assistant answer was discarded. Each replacement assistant turn was generated wholly by
the pinned teacher in its declared mode and accepted only after EOS, delimiter, non-empty-answer, sequence-length,
and chat-template round-trip validation.

## Contents

- Records per configuration: {mode_summary}
- Generation backend: `{settings.teacher.implementation}`
- Maximum sequence length: {settings.generation.sequence_length:,} tokens
- Sampling: deterministic greedy decoding
- Compatible NanoQuant record format: `ultrachat_messages`

Use the `thinking` or `non_thinking` configuration with split `train`. The optional `all` configuration combines
both files and retains the row-level `mode` field.

Every row contains `messages`, mode and teacher/source provenance, source and token hashes, token counts, and the
normal termination reason. See `manifest.json` for the immutable generation identity and file hashes.

## Redistribution

Review the prompt dataset and teacher-model licenses, terms, and content policy before redistributing or making the
repository public. Private publication is the interactive default.
"""


def _load_completed_dataset(
    run_root: Path,
    identity: str,
) -> dict[str, Any] | None:
    dataset_root = run_root / TEACHER_DATASET_DIRECTORY_NAME
    manifest_path = dataset_root / TEACHER_DATASET_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError("teacher dataset manifest is invalid JSON") from exc
    if manifest.get("identity") != identity:
        raise ValueError("completed teacher dataset identity differs from its settings")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("teacher dataset manifest has no file inventory")
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("teacher dataset manifest contains an invalid file entry")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("teacher dataset manifest contains an unsafe path")
        path = dataset_root / relative
        if not path.is_file():
            raise ValueError(f"teacher dataset file is missing: {relative}")
        if path.stat().st_size != int(item.get("bytes", -1)):
            raise ValueError(f"teacher dataset file size changed: {relative}")
        if "sha256:" + hash_file(path) != item.get("sha256"):
            raise ValueError(f"teacher dataset file hash changed: {relative}")
    return manifest


def _publish_dataset(
    run_root: Path,
    settings: TeacherDatasetSettings,
    traces_by_mode: Mapping[ReasoningMode, PreparedTeacherTraces],
    *,
    progress: TeacherDatasetProgress | None,
) -> dict[str, Any]:
    identity = teacher_dataset_identity(settings)
    completed = _load_completed_dataset(run_root, identity)
    if completed is not None:
        if progress is not None:
            progress(
                "teacher_dataset_local_reused",
                {"identity": identity, "dataset": str(run_root / TEACHER_DATASET_DIRECTORY_NAME)},
            )
        return completed
    destination = run_root / TEACHER_DATASET_DIRECTORY_NAME
    started = time.perf_counter()
    if progress is not None:
        progress("teacher_dataset_local_publish_started", {"identity": identity})
    with atomic_workspace(destination) as temporary:
        counts: dict[str, int] = {}
        data_paths: list[Path] = []
        ordered_ids: dict[str, list[str]] = {}
        for mode in settings.generation.modes:
            traces = traces_by_mode[mode]
            records = _trace_records(run_root, traces)
            rows: list[dict[str, object]] = []
            ids: list[str] = []
            for record in records:
                row_id = semantic_hash(
                    {
                        "identity": identity,
                        "mode": mode.value,
                        "source_hash": record["source_hash"],
                        "response_token_hash": record["response_token_hash"],
                    }
                )
                ids.append(row_id)
                rows.append(
                    {
                        "id": row_id,
                        "mode": mode.value,
                        "messages": record["messages"],
                        "source_dataset": settings.prompt_source.name,
                        "source_revision": settings.prompt_source.revision,
                        "source_split": settings.prompt_source.split,
                        "source_subset": settings.prompt_source.subset,
                        "source_hash": record["source_hash"],
                        "teacher_model": settings.teacher.source,
                        "teacher_revision": settings.teacher.revision,
                        "teacher_gguf_file": settings.teacher.gguf_filename,
                        "tokenizer_model": settings.teacher.tokenizer_source,
                        "tokenizer_revision": settings.teacher.tokenizer_revision,
                        "generation_implementation": settings.teacher.implementation,
                        "prompt_token_hash": record["prompt_token_hash"],
                        "response_token_hash": record["response_token_hash"],
                        "complete_token_hash": record["complete_token_hash"],
                        "prompt_tokens": record["prompt_tokens"],
                        "response_tokens": record["response_tokens"],
                        "stop_reason": record["stop_reason"],
                    }
                )
            data_path = temporary / "data" / f"{mode.value}.jsonl"
            counts[mode.value] = _write_jsonl(data_path, rows)
            data_paths.append(data_path)
            ordered_ids[mode.value] = ids
        card_path = temporary / "README.md"
        card_path.write_text(_dataset_card(settings, counts), encoding="utf-8", newline="\n")
        files = [
            {
                "path": path.relative_to(temporary).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": "sha256:" + hash_file(path),
            }
            for path in (*data_paths, card_path)
        ]
        manifest: dict[str, Any] = {
            "schema_version": TEACHER_DATASET_ARTIFACT_SCHEMA_VERSION,
            "producer": "nanoquant-teacher-response-dataset-v1",
            "identity": identity,
            "settings_hash": teacher_dataset_settings_hash(settings),
            "prompt_source": to_dict(settings.prompt_source),
            "teacher": to_dict(settings.teacher),
            "generation": to_dict(settings.generation),
            "compatible_record_format": "ultrachat_messages",
            "record_counts": counts,
            "ordered_record_ids": ordered_ids,
            "files": files,
        }
        atomic_write_json(temporary / TEACHER_DATASET_MANIFEST_NAME, manifest)
    if progress is not None:
        progress(
            "teacher_dataset_local_publish_completed",
            {
                "identity": identity,
                "dataset": str(destination),
                "record_counts": manifest["record_counts"],
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    return manifest


def _authenticated_api(api: HfApi | None = None) -> HfApi:
    if api is not None:
        return api
    token = os.environ.get("HF_TOKEN", "").strip() or (get_token() or "").strip()
    if not token:
        raise RuntimeError(
            "Hugging Face dataset upload requires HF_TOKEN or a cached Hugging Face login"
        )
    return HfApi(token=token)


def authenticated_huggingface_owner(*, api: HfApi | None = None) -> str:
    identity = _authenticated_api(api).whoami()
    name = identity.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("authenticated Hugging Face account has no usable name")
    return name.strip()


def _upload_dataset(
    run_root: Path,
    settings: TeacherDatasetSettings,
    manifest: Mapping[str, object],
    *,
    api: HfApi | None,
    progress: TeacherDatasetProgress | None,
) -> dict[str, object] | None:
    config = settings.upload
    if config is None:
        return None
    receipt_path = run_root / TEACHER_DATASET_UPLOAD_RECEIPT_NAME
    identity = str(manifest["identity"])
    try:
        existing = cast(dict[str, object], json.loads(receipt_path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        existing = {}
    except json.JSONDecodeError as exc:
        raise ValueError("teacher dataset upload receipt is invalid JSON") from exc
    if (
        existing.get("identity") == identity
        and existing.get("repo_id") == config.repo_id
        and isinstance(existing.get("commit_oid"), str)
    ):
        if progress is not None:
            progress("teacher_dataset_upload_reused", existing)
        return existing

    client = _authenticated_api(api)
    started = time.perf_counter()
    if progress is not None:
        progress(
            "teacher_dataset_upload_started",
            {"repo_id": config.repo_id, "private": config.private, "identity": identity},
        )
    repo_url = client.create_repo(
        config.repo_id,
        repo_type="dataset",
        private=config.private,
        exist_ok=True,
    )
    resolved_repo = str(getattr(repo_url, "repo_id", config.repo_id))
    with _upload_heartbeat(
        progress,
        repo_id=resolved_repo,
        started=started,
    ):
        commit = client.upload_folder(
            repo_id=resolved_repo,
            repo_type="dataset",
            folder_path=run_root / TEACHER_DATASET_DIRECTORY_NAME,
            path_in_repo=".",
            commit_message=config.commit_message,
        )
    commit_oid = str(getattr(commit, "oid", getattr(commit, "commit_oid", "")))
    if not commit_oid:
        raise RuntimeError("Hugging Face dataset upload returned no commit identity")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "identity": identity,
        "repo_id": resolved_repo,
        "commit_oid": commit_oid,
        "private": config.private,
        "uploaded_at": _now(),
    }
    atomic_write_json(receipt_path, receipt)
    if progress is not None:
        progress(
            "teacher_dataset_upload_completed",
            {
                **receipt,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    return receipt


@contextmanager
def _upload_heartbeat(
    progress: TeacherDatasetProgress | None,
    *,
    repo_id: str,
    started: float,
) -> Iterator[None]:
    if progress is None:
        yield
        return
    stopped = threading.Event()

    def report() -> None:
        while not stopped.wait(30.0):
            progress(
                "teacher_dataset_upload_progress",
                {
                    "repo_id": repo_id,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

    worker = threading.Thread(
        target=report,
        name="nanoquant-teacher-dataset-upload",
        daemon=True,
    )
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def execute_teacher_dataset(
    settings_path: str | Path,
    *,
    api: HfApi | None = None,
    snapshot_resolver: SnapshotResolver = resolve_teacher_snapshot,
    tokenizer_snapshot_resolver: TokenizerSnapshotResolver = resolve_tokenizer_snapshot,
    source_loader: SourceRecordLoader = load_prompt_records,
    trace_preparer: TeacherTracePreparer = prepare_teacher_traces,
    progress: TeacherDatasetProgress = _console_progress,
) -> int:
    """Resume generation, atomically publish the local dataset, and optionally upload it."""

    settings_file = Path(settings_path).resolve()
    settings, settings_digest = load_teacher_dataset_settings(settings_file)
    run_root = settings_file.parent
    load_repository_dotenv(Path(__file__).resolve().parents[2])
    identity = teacher_dataset_identity(settings)
    with RunLease(run_root / ".active-lease.json"):
        manifest = _load_completed_dataset(run_root, identity)
        if manifest is None:
            progress(
                "teacher_dataset_started",
                {
                    "identity": identity,
                    "teacher_source": settings.teacher.source,
                    "teacher_revision": settings.teacher.revision,
                    "source_dataset": settings.prompt_source.name,
                    "source_revision": settings.prompt_source.revision,
                    "modes": [mode.value for mode in settings.generation.modes],
                    "samples_per_mode": settings.generation.samples_per_mode,
                },
            )
            snapshot = snapshot_resolver(settings.teacher)
            tokenizer_snapshot = tokenizer_snapshot_resolver(
                settings.teacher.tokenizer_source,
                settings.teacher.tokenizer_revision,
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, local_files_only=False)
            behavior: ChatBehaviorPort = chat_behavior_for_snapshot(tokenizer_snapshot)
            unsupported = set(settings.generation.modes) - set(behavior.supported_modes)
            if unsupported:
                rendered = ", ".join(sorted(mode.value for mode in unsupported))
                raise ValueError(f"teacher chat adapter does not support requested modes: {rendered}")
            traces_by_mode: dict[ReasoningMode, PreparedTeacherTraces] = {}
            for mode in settings.generation.modes:
                progress(
                    "teacher_dataset_mode_started",
                    {
                        "mode": mode.value,
                        "target_records": settings.generation.samples_per_mode,
                    },
                )
                traces_by_mode[mode] = trace_preparer(
                    snapshot,
                    run_root,
                    _behavior_slice(settings, mode),
                    tokenizer,
                    behavior,
                    source_loader(settings.prompt_source, settings.generation.seed),
                    teacher_source=settings.teacher.source,
                    teacher_revision=settings.teacher.revision,
                    count=settings.generation.samples_per_mode,
                    sequence_length=settings.generation.sequence_length,
                    seed=settings.generation.seed,
                    device=settings.teacher.device,
                    source_adapter_identity=_source_adapter_identity(settings.prompt_source),
                    teacher_gguf_file=settings.teacher.gguf_filename,
                    teacher_tokenizer_source=settings.teacher.tokenizer_source,
                    teacher_tokenizer_revision=settings.teacher.tokenizer_revision,
                    progress=progress,
                )
                progress(
                    "teacher_dataset_mode_completed",
                    {
                        "mode": mode.value,
                        "record_count": len(traces_by_mode[mode].messages),
                        "artifact_id": traces_by_mode[mode].reference.artifact_id,
                    },
                )
            manifest = _publish_dataset(
                run_root,
                settings,
                traces_by_mode,
                progress=progress,
            )
        else:
            progress(
                "teacher_dataset_local_reused",
                {
                    "identity": identity,
                    "dataset": str(run_root / TEACHER_DATASET_DIRECTORY_NAME),
                },
            )
        upload_receipt = _upload_dataset(
            run_root,
            settings,
            manifest,
            api=api,
            progress=progress,
        )
        completion = {
            "schema_version": 1,
            "identity": identity,
            "settings_hash": settings_digest,
            "dataset": str((run_root / TEACHER_DATASET_DIRECTORY_NAME).resolve()),
            "record_counts": manifest["record_counts"],
            "upload": upload_receipt,
            "completed_at": _now(),
        }
        atomic_write_json(run_root / TEACHER_DATASET_COMPLETION_NAME, completion)
        progress("teacher_dataset_completed", completion)
    return 0


def load_teacher_catalog(path: str | Path) -> tuple[TeacherCatalogModel, ...]:
    """Read the ordered, prebuilt-GGUF teacher model catalog."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("families"), list):
        raise ValueError("interactive model catalog has no families")
    models: list[TeacherCatalogModel] = []
    for family in payload["families"]:
        if not isinstance(family, dict):
            raise ValueError("interactive model catalog contains an invalid family")
        family_id = str(family.get("id") or "")
        if not family_id.startswith("qwen3"):
            continue
        family_label = str(family.get("label") or family_id)
        variants = family.get("variants")
        if not isinstance(variants, list):
            raise ValueError(f"interactive model family {family_id!r} has no variants")
        for variant in variants:
            if (
                not isinstance(variant, dict)
                or not str(variant.get("source") or "").strip()
                or not str(variant.get("tokenizer_source") or "").strip()
                or not str(variant.get("gguf_filename") or "").strip()
            ):
                raise ValueError(f"interactive model family {family_id!r} has an invalid variant")
            models.append(
                TeacherCatalogModel(
                    family_id,
                    family_label,
                    str(variant["source"]),
                    str(variant["tokenizer_source"]),
                    str(variant["gguf_filename"]),
                    bool(variant.get("default", False)),
                )
            )
    if not models:
        raise ValueError("interactive model catalog has no reasoning-capable teacher models")
    for family, _label in _catalog_families(tuple(models)):
        defaults = sum(model.default for model in models if model.family == family)
        if defaults != 1:
            raise ValueError(
                f"teacher model family {family!r} must declare exactly one default"
            )
    return tuple(models)


def _run_status(root: Path, settings: TeacherDatasetSettings) -> str:
    if (root / ".active-lease.json").is_file():
        return "active"
    if (root / TEACHER_DATASET_COMPLETION_NAME).is_file():
        return "completed"
    if (root / TEACHER_DATASET_DIRECTORY_NAME / TEACHER_DATASET_MANIFEST_NAME).is_file():
        return "upload pending" if settings.upload is not None else "locally complete"
    journals = root / "state" / "teacher-traces"
    if journals.is_dir() and any(journals.glob("*.jsonl")):
        return "interrupted"
    return "created"


def discover_teacher_dataset_runs(
    repository_root: str | Path,
) -> tuple[TeacherDatasetRunRecord, ...]:
    root = Path(repository_root).resolve() / "evidence" / "teacher-datasets"
    if not root.is_dir():
        return ()
    records: list[TeacherDatasetRunRecord] = []
    for child in root.iterdir():
        settings_path = child / TEACHER_DATASET_SETTINGS_NAME
        if not settings_path.is_file():
            continue
        try:
            settings, digest = load_teacher_dataset_settings(settings_path)
        except (OSError, TypeError, ValueError):
            continue
        records.append(
            TeacherDatasetRunRecord(
                settings_path,
                settings,
                digest,
                _run_status(child, settings),
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda item: (item.settings.created_at, item.settings_path.parent.name),
            reverse=True,
        )
    )


class TeacherDatasetConsole:
    def __init__(self, input_fn: InputFn = input, write: WriteFn = print) -> None:
        self.input = input_fn
        self.write = write

    def choose(self, prompt: str, count: int, *, default: int = 1) -> int:
        if count <= 0 or not 1 <= default <= count:
            raise ValueError("teacher dataset menu bounds are invalid")
        while True:
            raw = self.input(f"{prompt} [{default}]: ").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                self.write(f"Enter a number from 1 to {count}.")
                continue
            if 1 <= value <= count:
                return value
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

    def positive_int(self, prompt: str, *, default: int) -> int:
        while True:
            raw = self.input(f"{prompt} [{default}]: ").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                self.write("Enter a positive integer.")
                continue
            if value > 0:
                return value
            self.write("Enter a positive integer.")


def _catalog_families(
    catalog: tuple[TeacherCatalogModel, ...],
) -> tuple[tuple[str, str], ...]:
    values: dict[str, str] = {}
    for model in catalog:
        values.setdefault(model.family, model.family_label)
    return tuple(values.items())


def _choose_teacher(
    console: TeacherDatasetConsole,
    catalog: tuple[TeacherCatalogModel, ...],
) -> TeacherCatalogModel:
    families = _catalog_families(catalog)
    console.write("Choose the teacher model family:")
    for index, (_family, label) in enumerate(families, start=1):
        console.write(f"  {index}. {label}")
    console.write(f"  {len(families) + 1}. Enter another Hugging Face model")
    family_choice = console.choose("Selection", len(families) + 1, default=1)
    if family_choice == len(families) + 1:
        value = console.input("Teacher model ID: ").strip()
        if not value:
            raise ValueError("teacher model ID is required")
        default_tokenizer = value[:-5] if value.lower().endswith("-gguf") else value
        tokenizer_source = (
            console.input(f"Tokenizer model [{default_tokenizer}]: ").strip()
            or default_tokenizer
        )
        gguf_filename = console.input(
            "BF16 GGUF filename [auto-detect, blank for safetensors]: "
        ).strip()
        return TeacherCatalogModel(
            "custom",
            "Custom",
            value,
            tokenizer_source,
            gguf_filename,
            True,
        )
    family = families[family_choice - 1][0]
    variants = tuple(model for model in catalog if model.family == family)
    default_variant = next(
        (index for index, model in enumerate(variants, start=1) if model.default),
        1,
    )
    console.write(f"Choose a {families[family_choice - 1][1]} teacher:")
    for index, model in enumerate(variants, start=1):
        console.write(f"  {index}. {model.label}")
    choice = console.choose("Selection", len(variants), default=default_variant)
    return variants[choice - 1]


def _choose_prompt_source(
    console: TeacherDatasetConsole,
) -> tuple[str, str | None, str, str | None, str]:
    console.write("Choose the prompt dataset:")
    console.write("  1. UltraChat 200K (use a small deterministic subset)")
    console.write("  2. Another Hugging Face conversational dataset")
    choice = console.choose("Selection", 2, default=1)
    if choice == 1:
        return ULTRACHAT_DATASET, ULTRACHAT_REVISION, ULTRACHAT_SPLIT, None, "messages"
    source = console.input("Dataset ID: ").strip()
    if not source:
        raise ValueError("prompt dataset ID is required")
    revision = console.input("Dataset revision [resolve current commit]: ").strip() or None
    split = console.input("Dataset split [train]: ").strip() or "train"
    subset = console.input("Dataset configuration [none]: ").strip() or None
    messages = console.input("Messages column [messages]: ").strip() or "messages"
    return source, revision, split, subset, messages


def _choose_modes(console: TeacherDatasetConsole) -> tuple[ReasoningMode, ...]:
    console.write("Generate which responses?")
    console.write("  1. Thinking and non-thinking")
    console.write("  2. Thinking only")
    console.write("  3. Non-thinking only")
    choice = console.choose("Selection", 3, default=1)
    if choice == 1:
        return ReasoningMode.THINKING, ReasoningMode.NON_THINKING
    if choice == 2:
        return (ReasoningMode.THINKING,)
    return (ReasoningMode.NON_THINKING,)


def _choose_backend(console: TeacherDatasetConsole) -> str:
    console.write("Choose the generation backend:")
    console.write("  1. llama.cpp server (recommended, resumable parallel generation)")
    console.write("  2. Transformers greedy generation")
    return (
        LLAMACPP_TEACHER_TRACE_IMPLEMENTATION
        if console.choose("Selection", 2, default=1) == 1
        else "hf-greedy-qwen3-v1"
    )


def _new_output_path(
    repository_root: Path,
    teacher_source: str,
    samples_per_mode: int,
    modes: tuple[ReasoningMode, ...],
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    mode_slug = "dual-mode" if len(modes) == 2 else _slug(modes[0].value)
    suffix = secrets.token_hex(3)
    name = (
        f"{timestamp}-{_slug(_source_basename(teacher_source))}-"
        f"{samples_per_mode}x-{mode_slug}-{suffix}"
    )
    if not _SAFE_SLUG.fullmatch(name):
        raise ValueError("derived teacher dataset run name is invalid")
    return repository_root / "evidence" / "teacher-datasets" / name


def run_interactive_teacher_dataset(
    repository_root: str | Path,
    catalog_path: str | Path,
    *,
    input_fn: InputFn = input,
    write: WriteFn = print,
    execute: ExecuteFn = execute_teacher_dataset,
    api: HfApi | None = None,
) -> int:
    """Run the resumable terminal menu and dispatch immutable settings."""

    root = Path(repository_root).resolve()
    load_repository_dotenv(root)
    console = TeacherDatasetConsole(input_fn, write)
    console.write("NanoQuant teacher-response dataset builder")
    existing = discover_teacher_dataset_runs(root)
    if existing:
        latest = existing[0]
        console.write("Previous dataset run")
        console.write(f"  Teacher: {latest.settings.teacher.source}")
        console.write(f"  Status:  {latest.status}")
        console.write(f"  Output:  {latest.settings_path.parent}")
        console.write("  1. Continue previous run")
        console.write("  2. Start a new dataset")
        console.write("  3. Exit")
        action = console.choose("Selection", 3, default=1)
        if action == 1:
            return execute(latest.settings_path)
        if action == 3:
            return 0

    source, source_revision, split, subset, messages_column = _choose_prompt_source(console)
    selected_teacher = _choose_teacher(console, load_teacher_catalog(catalog_path))
    modes = _choose_modes(console)
    samples = console.positive_int(
        "Accepted responses per mode",
        default=DEFAULT_SAMPLES_PER_MODE,
    )
    sequence_length = console.positive_int("Maximum complete sequence length", default=2048)
    implementation = _choose_backend(console)
    device = console.input("Generation device [cuda]: ").strip() or "cuda"
    console.write("Resolving immutable Hugging Face revisions...")
    pinned_source_revision = resolve_dataset_revision(source, source_revision, api=api)
    teacher_revision = resolve_model_revision(selected_teacher.source, None, api=api)
    tokenizer_revision = resolve_model_revision(
        selected_teacher.tokenizer_source,
        None,
        api=api,
    )
    gguf_filename = resolve_gguf_filename(
        selected_teacher.source,
        teacher_revision,
        selected_teacher.gguf_filename or None,
        api=api,
    )

    upload: TeacherDatasetUpload | None = None
    if console.yes_no("Upload the completed dataset to Hugging Face?", default=True):
        owner = authenticated_huggingface_owner(api=api)
        default_repo = (
            f"{owner}/{_slug(_source_basename(selected_teacher.source))}-"
            f"{_slug(_source_basename(source))}-teacher-responses"
        )
        repo_id = console.input(f"Dataset repository [{default_repo}]: ").strip() or default_repo
        private = console.yes_no("Keep the dataset private?", default=True)
        upload = TeacherDatasetUpload(repo_id, private=private)

    settings = new_teacher_dataset_settings(
        prompt_source=TeacherPromptSource(
            source,
            pinned_source_revision,
            split,
            subset,
            messages_column,
        ),
        teacher=TeacherModel(
            selected_teacher.source,
            teacher_revision,
            selected_teacher.tokenizer_source,
            tokenizer_revision,
            gguf_filename,
            implementation,
            device,
        ),
        generation=TeacherDatasetGeneration(
            modes=modes,
            samples_per_mode=samples,
            sequence_length=sequence_length,
        ),
        upload=upload,
    )
    output = _new_output_path(root, selected_teacher.source, samples, modes)
    console.write("Ready to generate")
    console.write(f"  Prompt dataset:  {source}@{pinned_source_revision}")
    console.write(f"  Teacher:         {selected_teacher.source}@{teacher_revision}")
    console.write(
        f"  Tokenizer:       {selected_teacher.tokenizer_source}@{tokenizer_revision}"
    )
    console.write(f"  GGUF:            {gguf_filename or 'convert the pinned checkpoint'}")
    console.write(f"  Modes:           {', '.join(mode.value for mode in modes)}")
    console.write(f"  Samples/mode:    {samples}")
    console.write(f"  Backend:         {implementation}")
    console.write(f"  Hugging Face:    {upload.repo_id if upload is not None else 'no upload'}")
    console.write(f"  Output:          {output}")
    if not console.yes_no("Start generation?", default=True):
        return 0
    output.mkdir(parents=True, exist_ok=False)
    settings_path = output / TEACHER_DATASET_SETTINGS_NAME
    write_teacher_dataset_settings(settings_path, settings)
    console.write(f"Settings written: {settings_path}")
    return execute(settings_path)


__all__ = [
    "DEFAULT_SAMPLES_PER_MODE",
    "TEACHER_DATASET_COMPLETION_NAME",
    "TEACHER_DATASET_DIRECTORY_NAME",
    "TEACHER_DATASET_MANIFEST_NAME",
    "TEACHER_DATASET_SETTINGS_KIND",
    "TEACHER_DATASET_SETTINGS_NAME",
    "TEACHER_DATASET_SETTINGS_SCHEMA_VERSION",
    "TEACHER_DATASET_UPLOAD_RECEIPT_NAME",
    "ULTRACHAT_DATASET",
    "ULTRACHAT_REVISION",
    "ULTRACHAT_SPLIT",
    "TeacherCatalogModel",
    "TeacherDatasetGeneration",
    "TeacherDatasetRunRecord",
    "TeacherDatasetSettings",
    "TeacherDatasetUpload",
    "TeacherModel",
    "TeacherPromptSource",
    "authenticated_huggingface_owner",
    "discover_teacher_dataset_runs",
    "execute_teacher_dataset",
    "load_prompt_records",
    "load_teacher_catalog",
    "load_teacher_dataset_settings",
    "new_teacher_dataset_settings",
    "normalize_prompt_record",
    "resolve_dataset_revision",
    "resolve_gguf_filename",
    "resolve_model_revision",
    "resolve_teacher_snapshot",
    "resolve_tokenizer_snapshot",
    "run_interactive_teacher_dataset",
    "teacher_dataset_identity",
    "teacher_dataset_settings_hash",
    "write_teacher_dataset_settings",
]
