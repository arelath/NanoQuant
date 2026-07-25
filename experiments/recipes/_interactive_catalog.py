"""Strict YAML codec for the promoted interactive model catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from nanoquant.config.codec import ConfigDecodeError, from_dict
from nanoquant.config.schema import RunConfig
from nanoquant.interactive_compression import RecommendedModel

INTERACTIVE_MODEL_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _RecommendedModelEntry:
    family: str
    family_label: str
    family_order: int
    variant: str
    variant_label: str
    variant_order: int
    source: str
    revision: str
    runtime_family: str
    release_name: str
    profile_id: str
    evidence: tuple[str, ...]
    template_id: str
    default_target_bpw: float = 1.0
    default_family: bool = False
    default_variant: bool = False
    maximum_wddm_shared_gib: float | None = 0.75
    restore_completed_blocks: bool = False
    quality_backend: str | None = None
    large_model_guards: bool = False
    llamacpp_quality: bool = True
    llamacpp_quality_parallel: int = 4

    def materialize(self, templates: Mapping[str, RunConfig]) -> RecommendedModel:
        try:
            template = templates[self.template_id]
        except KeyError as exc:
            raise ConfigDecodeError(
                f"interactive_model_catalog.models[{self.variant}].template_id",
                f"unknown template {self.template_id!r}",
            ) from exc
        return RecommendedModel(
            family=self.family,
            family_label=self.family_label,
            family_order=self.family_order,
            variant=self.variant,
            variant_label=self.variant_label,
            variant_order=self.variant_order,
            source=self.source,
            revision=self.revision,
            runtime_family=self.runtime_family,
            release_name=self.release_name,
            profile_id=self.profile_id,
            evidence=self.evidence,
            template=template,
            default_target_bpw=self.default_target_bpw,
            default_family=self.default_family,
            default_variant=self.default_variant,
            maximum_wddm_shared_gib=self.maximum_wddm_shared_gib,
            restore_completed_blocks=self.restore_completed_blocks,
            quality_backend=self.quality_backend,
            large_model_guards=self.large_model_guards,
            llamacpp_quality=self.llamacpp_quality,
            llamacpp_quality_parallel=self.llamacpp_quality_parallel,
        )


@dataclass(frozen=True, slots=True)
class _RecommendedModelCatalog:
    schema_version: int
    models: tuple[_RecommendedModelEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != INTERACTIVE_MODEL_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported interactive model catalog schema: {self.schema_version}"
            )
        if not self.models:
            raise ValueError("interactive model catalog must contain at least one model")


def _validate_catalog(models: tuple[RecommendedModel, ...]) -> None:
    variants: set[str] = set()
    positions: set[tuple[str, int]] = set()
    family_metadata: dict[str, tuple[str, int]] = {}
    default_families = 0
    default_variants: dict[str, int] = {}
    for model in models:
        if model.variant in variants:
            raise ValueError(f"interactive model catalog repeats variant {model.variant!r}")
        variants.add(model.variant)
        position = (model.family, model.variant_order)
        if position in positions:
            raise ValueError(
                "interactive model catalog repeats variant order "
                f"{model.variant_order} in family {model.family!r}"
            )
        positions.add(position)
        metadata = (model.family_label, model.family_order)
        previous = family_metadata.setdefault(model.family, metadata)
        if previous != metadata:
            raise ValueError(
                f"interactive model catalog has inconsistent metadata for family {model.family!r}"
            )
        default_families += int(model.default_family)
        default_variants[model.family] = (
            default_variants.get(model.family, 0) + int(model.default_variant)
        )
    if default_families != 1:
        raise ValueError("interactive model catalog must declare exactly one default family")
    invalid_defaults = {
        family: count for family, count in default_variants.items() if count != 1
    }
    if invalid_defaults:
        raise ValueError(
            "interactive model catalog must declare exactly one default variant per family: "
            f"{invalid_defaults}"
        )


def load_interactive_recommended_models(
    path: str | Path,
    templates: Mapping[str, RunConfig],
) -> tuple[RecommendedModel, ...]:
    """Load, materialize, and validate a versioned promoted-model catalog."""

    source = Path(path)
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigDecodeError(
            "interactive_model_catalog",
            f"unsupported catalog extension {source.suffix!r}",
        )
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigDecodeError("interactive_model_catalog", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ConfigDecodeError(
            "interactive_model_catalog",
            "catalog root must be an object",
        )
    catalog = from_dict(
        _RecommendedModelCatalog,
        payload,
        path="interactive_model_catalog",
    )
    models = tuple(entry.materialize(templates) for entry in catalog.models)
    _validate_catalog(models)
    return models


__all__ = [
    "INTERACTIVE_MODEL_CATALOG_SCHEMA_VERSION",
    "load_interactive_recommended_models",
]
