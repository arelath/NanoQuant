"""Resolve pinned Hugging Face model configuration without loading weights."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

from huggingface_hub import HfApi, hf_hub_download

_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ResolvedModelConfig:
    values: dict[str, Any]
    revision: str
    path: Path


def load_snapshot_model_config(snapshot: str | Path) -> dict[str, Any]:
    """Load and validate the configuration stored in a resolved snapshot."""

    path = Path(snapshot).resolve() / "config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resolved model config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"resolved model config root must be an object: {path}")
    return cast(dict[str, Any], payload)


@cache
def resolve_model_config(source: str, revision: str | None) -> ResolvedModelConfig:
    """Resolve a local or Hub model config and pin a mutable Hub revision."""

    local = Path(source)
    if local.exists():
        path = local.resolve() / "config.json"
        values = load_snapshot_model_config(local)
        digest = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ResolvedModelConfig(values, f"local-{digest}", path)

    resolved_revision = revision
    if resolved_revision is None or not _COMMIT_REVISION.fullmatch(resolved_revision):
        info = HfApi().model_info(source, revision=revision)
        resolved_revision = str(info.sha)
    path = Path(
        hf_hub_download(
            repo_id=source,
            filename="config.json",
            revision=resolved_revision,
        )
    ).resolve()
    values = load_snapshot_model_config(path.parent)
    return ResolvedModelConfig(values, resolved_revision, path)


__all__ = [
    "ResolvedModelConfig",
    "load_snapshot_model_config",
    "resolve_model_config",
]
