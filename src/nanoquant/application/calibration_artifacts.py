"""Persist materialized calibration and build portable objective specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import ObjectiveConfig
from nanoquant.domain.models import (
    ArtifactRef,
    CalibrationStats,
    ComponentRef,
    DatasetIdentity,
    LayerCalibrationStats,
    LayerId,
    ModelIdentity,
    ObjectiveSpec,
    StatisticSummary,
)
from nanoquant.ports.artifact_store import ArtifactStore
from nanoquant.ports.tensor_store import TensorStore

from .calibration import MaterializedLayerCalibration


@dataclass(frozen=True, slots=True)
class PersistedCalibration:
    reference: ArtifactRef
    stats: CalibrationStats


@dataclass(frozen=True, slots=True)
class PersistedObjectives:
    reference: ArtifactRef
    objectives: tuple[ObjectiveSpec, ...]


def _summary(value: torch.Tensor) -> StatisticSummary:
    finite = torch.isfinite(value)
    safe = value[finite].float()
    if safe.numel() == 0:
        return StatisticSummary(float("nan"), float("nan"), float("nan"), 0.0, value.numel())
    return StatisticSummary(
        float(safe.min()),
        float(safe.max()),
        float(safe.mean()),
        float((value == 0).float().mean()),
        int((~finite).sum()),
    )


def persist_calibration(
    materialized: tuple[tuple[LayerId, MaterializedLayerCalibration], ...],
    model: ModelIdentity,
    dataset: DatasetIdentity,
    method: str,
    accumulation_dtype: str,
    artifacts: ArtifactStore,
    tensors: TensorStore,
    *,
    total_tokens: int = 0,
) -> PersistedCalibration:
    values = {}
    for layer, layer_stats in materialized:
        if (
            not torch.isfinite(layer_stats.input_importance).all()
            or not torch.isfinite(layer_stats.output_importance).all()
            or (layer_stats.input_mean is not None and not torch.isfinite(layer_stats.input_mean).all())
        ):
            raise ValueError(f"non-finite calibration statistics for {layer}")
        prefix = f"block_{layer.block.index}.{layer.path}"
        values[f"{prefix}.input_importance"] = layer_stats.input_importance
        values[f"{prefix}.output_importance"] = layer_stats.output_importance
        if layer_stats.input_mean is not None:
            values[f"{prefix}.input_mean"] = layer_stats.input_mean
    references = tensors.put("calibration-tensors", values)
    layers = tuple(
        LayerCalibrationStats(
            layer,
            references[f"block_{layer.block.index}.{layer.path}.input_importance"],
            references[f"block_{layer.block.index}.{layer.path}.output_importance"],
            None,
            _summary(layer_stats.input_importance),
            _summary(layer_stats.output_importance),
            (),
            references.get(f"block_{layer.block.index}.{layer.path}.input_mean"),
        )
        for layer, layer_stats in materialized
    )
    sample_count = materialized[0][1].sample_count if materialized else 0
    calibration_stats = CalibrationStats(
        1,
        ComponentRef("calibration", "1"),
        model,
        dataset,
        method,
        accumulation_dtype,
        layers,
        sample_count,
        total_tokens,
    )
    with artifacts.begin_write("calibration-stats") as writer:
        (writer.path / "stats.json").write_text(
            json.dumps(to_dict(calibration_stats), sort_keys=True, indent=2), encoding="utf-8"
        )
        descriptor = writer.commit()
    return PersistedCalibration(ArtifactRef("calibration-stats", descriptor.artifact_id, 1), calibration_stats)


def build_objectives(
    calibration: PersistedCalibration, config: ObjectiveConfig, artifacts: ArtifactStore
) -> PersistedObjectives:
    objectives = tuple(
        ObjectiveSpec(
            1,
            layer.layer,
            config.kind.value,
            layer.input_importance,
            layer.output_importance,
            layer.input_covariance,
            config.regularization.diagonal_damp_fraction,
            "target_weighted_norm_squared",
            None,
            calibration.reference,
            layer.input_mean,
        )
        for layer in calibration.stats.layers
    )
    with artifacts.begin_write("objective-specs") as writer:
        (writer.path / "objectives.json").write_text(
            json.dumps(to_dict(objectives), sort_keys=True, indent=2), encoding="utf-8"
        )
        descriptor = writer.commit()
    return PersistedObjectives(ArtifactRef("objective-specs", descriptor.artifact_id, 1), objectives)
