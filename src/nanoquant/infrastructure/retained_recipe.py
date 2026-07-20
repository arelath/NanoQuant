"""Load a validated canonical recipe identity from a retained run manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nanoquant.config.codec import from_dict
from nanoquant.config.schema import RunConfig


@dataclass(frozen=True, slots=True)
class RetainedRunRecipe:
    config: RunConfig
    maximum_wddm_shared_bytes: int | None


def load_retained_run_recipe(
    run_output: str | Path,
    *,
    expected_source: str,
    expected_revision: str,
    allowed_statuses: tuple[str, ...] = ("completed",),
) -> RetainedRunRecipe:
    """Read one completed run's canonical config without executing its launcher."""

    manifest_path = Path(run_output).resolve() / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not allowed_statuses or manifest["status"] not in allowed_statuses:
            raise ValueError(
                f"retained recipe run status {manifest['status']!r} is not one of {allowed_statuses!r}"
            )
        resolved = manifest["resolved_config"]
        config = from_dict(
            RunConfig,
            resolved["canonical_run_config"],
            path="recipe_run.canonical_run_config",
        )
        maximum_wddm_shared_bytes = resolved.get("maximum_wddm_shared_bytes")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"retained recipe manifest is invalid: {manifest_path}") from exc
    if config.model.source != expected_source or str(config.model.revision) != expected_revision:
        raise ValueError(
            "retained recipe model identity differs from the requested model: "
            f"{config.model.source}@{config.model.revision} != {expected_source}@{expected_revision}"
        )
    if maximum_wddm_shared_bytes is not None and (
        not isinstance(maximum_wddm_shared_bytes, int) or maximum_wddm_shared_bytes <= 0
    ):
        raise ValueError("retained recipe maximum_wddm_shared_bytes is invalid")
    return RetainedRunRecipe(config, maximum_wddm_shared_bytes)
