"""Versioned smoke evaluators over validated packed runtime artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nanoquant.application.evaluation import (
    EvaluatorRegistry,
    EvaluatorSpec,
    PackedArtifactStructureRequest,
    PackedArtifactStructureResult,
    PackedReferenceParityEvaluationResult,
    PackedReferenceParityRequest,
)
from nanoquant.runtime import open_packed_artifact, validate_packed_reference_parity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_packed_artifact_structure(
    request: object,
) -> PackedArtifactStructureResult:
    if not isinstance(request, PackedArtifactStructureRequest):
        raise TypeError("packed artifact structure evaluator requires PackedArtifactStructureRequest")
    artifact = open_packed_artifact(request.packed_artifact, verify_hashes=True)
    descriptor = artifact.root / "nanoquant-packed-model.json"
    tensor_count = sum(
        len(layer.tensors)
        for block in artifact.manifest.blocks
        for layer in block.layers
    )
    physical_bytes = descriptor.stat().st_size + sum(
        (artifact.root / block.path).stat().st_size for block in artifact.manifest.blocks
    )
    return PackedArtifactStructureResult(
        artifact.root,
        _sha256(descriptor),
        len(artifact.manifest.blocks),
        artifact.manifest.layer_count,
        tensor_count,
        artifact.manifest.weight_bytes,
        physical_bytes,
        True,
    )


def evaluate_packed_reference_parity(
    request: object,
) -> PackedReferenceParityEvaluationResult:
    if not isinstance(request, PackedReferenceParityRequest):
        raise TypeError("packed reference evaluator requires PackedReferenceParityRequest")
    result = validate_packed_reference_parity(
        request.logical_artifact,
        request.packed_artifact,
        absolute_tolerance=request.absolute_tolerance,
    )
    return PackedReferenceParityEvaluationResult(
        result.logical_artifact,
        result.packed_artifact,
        result.layer_count,
        result.output_elements,
        result.maximum_absolute_error,
        result.maximum_error_layer,
        True,
    )


def register_packed_smoke_evaluators(registry: EvaluatorRegistry) -> None:
    registry.register(
        EvaluatorSpec(
            "packed-artifact-structure",
            "1",
            "smoke",
            (("hashes_and_headers", True),),
        ),
        evaluate_packed_artifact_structure,
    )
    registry.register(
        EvaluatorSpec(
            "packed-reference-parity",
            "1",
            "smoke",
            (("default_absolute_tolerance", 0.0),),
        ),
        evaluate_packed_reference_parity,
    )
