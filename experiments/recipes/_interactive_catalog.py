"""Strict YAML codec for the promoted interactive model catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from nanoquant.config.codec import ConfigDecodeError, from_dict
from nanoquant.config.schema import RunConfig
from nanoquant.interactive_compression import RecommendedModel

INTERACTIVE_MODEL_CATALOG_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class _RecommendedVariantEntry:
    id: str
    label: str
    source: str
    revision: str
    runtime_family: str
    release_name: str
    profile_id: str
    evidence: tuple[str, ...]
    template_id: str
    default_target_bpw: float = 1.0
    default: bool = False
    maximum_wddm_shared_gib: float | None = 0.75
    restore_completed_blocks: bool = False
    quality_backend: str | None = None
    large_model_guards: bool = False
    llamacpp_quality: bool = True
    llamacpp_quality_parallel: int = 4

    def materialize(
        self,
        *,
        family_id: str,
        family_label: str,
        default_family: bool,
        templates: Mapping[str, RunConfig],
    ) -> RecommendedModel:
        try:
            template = templates[self.template_id]
        except KeyError as exc:
            raise ConfigDecodeError(
                f"interactive_model_catalog.families[{family_id}].variants[{self.id}].template_id",
                f"unknown template {self.template_id!r}",
            ) from exc
        return RecommendedModel(
            family=family_id,
            family_label=family_label,
            variant=self.id,
            variant_label=self.label,
            source=self.source,
            revision=self.revision,
            runtime_family=self.runtime_family,
            release_name=self.release_name,
            profile_id=self.profile_id,
            evidence=self.evidence,
            template=template,
            default_target_bpw=self.default_target_bpw,
            default_family=default_family,
            default_variant=self.default,
            maximum_wddm_shared_gib=self.maximum_wddm_shared_gib,
            restore_completed_blocks=self.restore_completed_blocks,
            quality_backend=self.quality_backend,
            large_model_guards=self.large_model_guards,
            llamacpp_quality=self.llamacpp_quality,
            llamacpp_quality_parallel=self.llamacpp_quality_parallel,
        )


@dataclass(frozen=True, slots=True)
class _RecommendedFamilyEntry:
    id: str
    label: str
    variants: tuple[_RecommendedVariantEntry, ...]
    default: bool = False

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError(
                f"interactive model family {self.id!r} must contain at least one variant"
            )


@dataclass(frozen=True, slots=True)
class _RecommendedModelCatalog:
    schema_version: int
    families: tuple[_RecommendedFamilyEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != INTERACTIVE_MODEL_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported interactive model catalog schema: {self.schema_version}"
            )
        if not self.families:
            raise ValueError("interactive model catalog must contain at least one family")


def _validate_catalog(catalog: _RecommendedModelCatalog) -> None:
    family_ids: set[str] = set()
    variants: set[str] = set()
    for family in catalog.families:
        if family.id in family_ids:
            raise ValueError(
                f"interactive model catalog repeats family {family.id!r}"
            )
        family_ids.add(family.id)
        default_variants = 0
        for variant in family.variants:
            if variant.id in variants:
                raise ValueError(
                    f"interactive model catalog repeats variant {variant.id!r}"
                )
            variants.add(variant.id)
            default_variants += int(variant.default)
        if default_variants != 1:
            raise ValueError(
                "interactive model catalog must declare exactly one default variant "
                f"for family {family.id!r}"
            )
    if sum(int(family.default) for family in catalog.families) != 1:
        raise ValueError("interactive model catalog must declare exactly one default family")


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
    _validate_catalog(catalog)
    return tuple(
        variant.materialize(
            family_id=family.id,
            family_label=family.label,
            default_family=family.default,
            templates=templates,
        )
        for family in catalog.families
        for variant in family.variants
    )


__all__ = [
    "INTERACTIVE_MODEL_CATALOG_SCHEMA_VERSION",
    "load_interactive_recommended_models",
]
