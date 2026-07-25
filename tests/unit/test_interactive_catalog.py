from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from recipes import INTERACTIVE_RECOMMENDED_MODELS
from recipes._interactive_catalog import load_interactive_recommended_models

from nanoquant.config.codec import ConfigDecodeError


def _entry() -> dict[str, Any]:
    model = INTERACTIVE_RECOMMENDED_MODELS[0]
    return {
        "family": model.family,
        "family_label": model.family_label,
        "family_order": model.family_order,
        "variant": model.variant,
        "variant_label": model.variant_label,
        "variant_order": model.variant_order,
        "source": model.source,
        "revision": model.revision,
        "runtime_family": model.runtime_family,
        "release_name": model.release_name,
        "profile_id": model.profile_id,
        "evidence": list(model.evidence),
        "template_id": "fixture-template",
        "default_family": True,
        "default_variant": True,
    }


def _write(path: Path, models: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "models": models}, sort_keys=False),
        encoding="utf-8",
    )


def test_repository_interactive_catalog_is_loaded_from_yaml() -> None:
    catalog_path = (
        Path(__file__).parents[2]
        / "experiments"
        / "recipes"
        / "interactive_recommended_models.yaml"
    )
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))

    assert len(payload["models"]) == len(INTERACTIVE_RECOMMENDED_MODELS) == 9
    assert [item["variant"] for item in payload["models"]] == [
        model.variant for model in INTERACTIVE_RECOMMENDED_MODELS
    ]
    assert all("template_id" in item for item in payload["models"])


def test_interactive_catalog_rejects_unknown_fields(tmp_path: Path) -> None:
    entry = _entry()
    entry["templat_id"] = entry.pop("template_id")
    path = tmp_path / "catalog.yaml"
    _write(path, [entry])

    with pytest.raises(ConfigDecodeError, match="template_id"):
        load_interactive_recommended_models(
            path,
            {"fixture-template": INTERACTIVE_RECOMMENDED_MODELS[0].template},
        )


def test_interactive_catalog_rejects_unknown_template_ids(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    _write(path, [_entry()])

    with pytest.raises(ConfigDecodeError, match="unknown template"):
        load_interactive_recommended_models(path, {})


def test_interactive_catalog_rejects_duplicate_variants(tmp_path: Path) -> None:
    entry = _entry()
    duplicate = dict(entry)
    duplicate["variant_order"] = 1
    duplicate["default_family"] = False
    duplicate["default_variant"] = False
    path = tmp_path / "catalog.yaml"
    _write(path, [entry, duplicate])

    with pytest.raises(ValueError, match="repeats variant"):
        load_interactive_recommended_models(
            path,
            {"fixture-template": INTERACTIVE_RECOMMENDED_MODELS[0].template},
        )
