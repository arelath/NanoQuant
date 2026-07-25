from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from recipes import INTERACTIVE_RECOMMENDED_MODELS
from recipes._interactive_catalog import load_interactive_recommended_models

from nanoquant.config.codec import ConfigDecodeError


def _variant() -> dict[str, Any]:
    model = INTERACTIVE_RECOMMENDED_MODELS[0]
    return {
        "id": model.variant,
        "label": model.variant_label,
        "source": model.source,
        "revision": model.revision,
        "runtime_family": model.runtime_family,
        "release_name": model.release_name,
        "profile_id": model.profile_id,
        "evidence": list(model.evidence),
        "template_id": "fixture-template",
        "default": True,
    }


def _family(variants: list[dict[str, Any]]) -> dict[str, Any]:
    model = INTERACTIVE_RECOMMENDED_MODELS[0]
    return {
        "id": model.family,
        "label": model.family_label,
        "default": True,
        "variants": variants,
    }


def _write(path: Path, families: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump({"schema_version": 2, "families": families}, sort_keys=False),
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
    variants = [
        variant
        for family in payload["families"]
        for variant in family["variants"]
    ]

    assert len(payload["families"]) == 4
    assert len(variants) == len(INTERACTIVE_RECOMMENDED_MODELS) == 9
    assert [item["id"] for item in variants] == [
        model.variant for model in INTERACTIVE_RECOMMENDED_MODELS
    ]
    assert all("template_id" in item for item in variants)
    assert all("family_order" not in family for family in payload["families"])
    assert all("variant_order" not in variant for variant in variants)


def test_interactive_catalog_rejects_unknown_fields(tmp_path: Path) -> None:
    variant = _variant()
    variant["templat_id"] = variant.pop("template_id")
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([variant])])

    with pytest.raises(ConfigDecodeError, match="template_id"):
        load_interactive_recommended_models(
            path,
            {"fixture-template": INTERACTIVE_RECOMMENDED_MODELS[0].template},
        )


def test_interactive_catalog_rejects_unknown_template_ids(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([_variant()])])

    with pytest.raises(ConfigDecodeError, match="unknown template"):
        load_interactive_recommended_models(path, {})


def test_interactive_catalog_rejects_duplicate_variants(tmp_path: Path) -> None:
    variant = _variant()
    duplicate = dict(variant)
    duplicate["default"] = False
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([variant, duplicate])])

    with pytest.raises(ValueError, match="repeats variant"):
        load_interactive_recommended_models(
            path,
            {"fixture-template": INTERACTIVE_RECOMMENDED_MODELS[0].template},
        )
