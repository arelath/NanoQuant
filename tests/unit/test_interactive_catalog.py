from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from recipes import INTERACTIVE_RECOMMENDED_MODELS
from recipes._interactive_catalog import load_interactive_recommended_models

from nanoquant.config.codec import ConfigDecodeError
from nanoquant.config.schema import LLAMACPP_TEACHER_TRACE_IMPLEMENTATION
from nanoquant.interactive_compression import _slug as interactive_slug


def _variant() -> dict[str, Any]:
    model = INTERACTIVE_RECOMMENDED_MODELS[0]
    return {
        "source": model.source,
        "runtime_family": model.runtime_family,
        "profile_id": model.profile_id,
        "evidence": list(model.evidence),
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
        yaml.safe_dump({"schema_version": 5, "families": families}, sort_keys=False),
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

    assert len(payload["families"]) == 6
    assert len(variants) == len(INTERACTIVE_RECOMMENDED_MODELS) == 19
    assert payload["schema_version"] == 5
    assert all("profile_id" in item for item in variants)
    assert all("id" not in item for item in variants)
    assert all("label" not in item for item in variants)
    assert all("release_name" not in item for item in variants)
    assert all("template_id" not in item for item in variants)
    assert all("family_order" not in family for family in payload["families"])
    assert all("variant_order" not in variant for variant in variants)
    assert all("revision" not in variant for variant in variants)
    assert all(
        model.revision == model.template.model.revision
        for model in INTERACTIVE_RECOMMENDED_MODELS
    )
    assert all(
        model.variant == model.release_name == interactive_slug(model.variant_label)
        for model in INTERACTIVE_RECOMMENDED_MODELS
    )
    assert all(
        model.variant_label == model.source.rsplit("/", 1)[-1]
        for model in INTERACTIVE_RECOMMENDED_MODELS
    )
    qwen_0_6b = next(
        model
        for model in INTERACTIVE_RECOMMENDED_MODELS
        if model.source == "Qwen/Qwen3-0.6B"
    )
    generated = tuple(
        item.teacher_trace_generation
        for item in qwen_0_6b.template.dataset.behavior_slices
        if item.teacher_trace_generation is not None
    )
    assert generated
    assert all(
        item.implementation == LLAMACPP_TEACHER_TRACE_IMPLEMENTATION
        for item in generated
    )
    qwen_sources = tuple(
        model.source
        for model in INTERACTIVE_RECOMMENDED_MODELS
        if model.family.startswith("qwen3")
    )
    assert qwen_sources == (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.6-27B",
    )
    assert {
        model.source: model.revision
        for model in INTERACTIVE_RECOMMENDED_MODELS
        if model.source in qwen_sources[1:] and model.source != "Qwen/Qwen3-8B"
    } == {
        "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
        "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
        "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
        "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "Qwen/Qwen3.6-27B": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    }
    assert all(
        marker not in model.source
        for model in INTERACTIVE_RECOMMENDED_MODELS
        if model.family.startswith("qwen3")
        for marker in ("-Base", "SAE-", "-FP8", "-GGUF", "-GPTQ", "-AWQ", "-MLX")
    )
    assert all(
        all(
            item.teacher_trace_generation is not None
            for item in model.template.dataset.behavior_slices
            if item.mode.value in {"thinking", "non_thinking"}
        )
        for model in INTERACTIVE_RECOMMENDED_MODELS
        if model.family.startswith("qwen3")
    )


def test_interactive_catalog_rejects_unknown_fields(tmp_path: Path) -> None:
    variant = _variant()
    variant["template_id"] = "obsolete-template-key"
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([variant])])

    with pytest.raises(ConfigDecodeError, match="template_id"):
        load_interactive_recommended_models(
            path,
            {variant["source"]: INTERACTIVE_RECOMMENDED_MODELS[0].template},
        )


def test_interactive_catalog_requires_a_template_for_each_variant(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([_variant()])])

    with pytest.raises(ConfigDecodeError, match="no reusable template registered for source"):
        load_interactive_recommended_models(path, {})


def test_interactive_catalog_rejects_unpinned_template_revision(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([_variant()])])
    template = INTERACTIVE_RECOMMENDED_MODELS[0].template
    unpinned = replace(
        template,
        model=replace(template.model, revision=None),
    )

    with pytest.raises(ValueError, match="template revision is required"):
        load_interactive_recommended_models(
            path,
            {INTERACTIVE_RECOMMENDED_MODELS[0].source: unpinned},
        )


def test_interactive_catalog_allows_id_and_release_name_overrides(tmp_path: Path) -> None:
    variant = _variant()
    variant["id"] = "special-variant"
    variant["release_name"] = "special-release"
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([variant])])
    model = INTERACTIVE_RECOMMENDED_MODELS[0]

    loaded = load_interactive_recommended_models(
        path,
        {model.source: model.template},
    )

    assert loaded[0].variant == "special-variant"
    assert loaded[0].release_name == "special-release"
    assert loaded[0].variant_label == "Qwen3-0.6B"


def test_interactive_catalog_rejects_duplicate_variants(tmp_path: Path) -> None:
    variant = _variant()
    duplicate = dict(variant)
    duplicate["default"] = False
    path = tmp_path / "catalog.yaml"
    _write(path, [_family([variant, duplicate])])
    model = INTERACTIVE_RECOMMENDED_MODELS[0]

    with pytest.raises(ValueError, match="repeats variant"):
        load_interactive_recommended_models(
            path,
            {model.source: model.template},
        )
