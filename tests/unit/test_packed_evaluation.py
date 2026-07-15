from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from nanoquant.application.evaluation import (
    EvaluatorRegistry,
    PackedArtifactStructureRequest,
    PackedArtifactStructureResult,
    PackedReferenceParityEvaluationResult,
    PackedReferenceParityRequest,
)
from nanoquant.infrastructure.packed_evaluation import register_packed_smoke_evaluators
from nanoquant.runtime import (
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    convert_logical_to_packed,
    write_logical_artifact,
)


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    spec = QuantizedLinearSpec(
        "blocks.0.linear",
        "nanoquant-v1",
        35,
        3,
        33,
        "float32",
        "float32",
        has_bias=True,
    )
    state = LogicalLayerState(
        spec,
        torch.where(torch.arange(99).reshape(3, 33) % 2 == 0, 1.0, -1.0),
        torch.where(torch.arange(33 * 35).reshape(33, 35) % 3 == 0, -1.0, 1.0),
        torch.linspace(0.5, 1.5, 35),
        torch.linspace(0.75, 1.25, 33),
        torch.linspace(1.0, 1.5, 3),
        torch.tensor([0.1, -0.2, 0.3]),
    )
    logical = write_logical_artifact(
        tmp_path / "logical",
        RuntimeModelMetadata("fixture/model", "revision", "fixture", "config", "tokenizer"),
        {0: (state,)},
    )
    packed = convert_logical_to_packed(logical.root, tmp_path / "packed")
    return logical.root, packed.root


def test_packed_smoke_evaluators_validate_structure_and_reference_parity(tmp_path: Path) -> None:
    logical, packed = _artifacts(tmp_path)
    registry = EvaluatorRegistry()
    register_packed_smoke_evaluators(registry)

    results = registry.evaluate_tier(
        "smoke",
        {
            ("packed-artifact-structure", "1"): PackedArtifactStructureRequest(packed),
            ("packed-reference-parity", "1"): PackedReferenceParityRequest(logical, packed),
        },
    )

    assert [spec.name for spec, _result in results] == [
        "packed-artifact-structure",
        "packed-reference-parity",
    ]
    structure = results[0][1]
    assert isinstance(structure, PackedArtifactStructureResult)
    descriptor = packed / "nanoquant-packed-model.json"
    assert structure.descriptor_sha256 == hashlib.sha256(descriptor.read_bytes()).hexdigest()
    assert structure.block_count == structure.layer_count == 1
    assert structure.tensor_count == 6
    assert structure.physical_bytes == sum(
        path.stat().st_size for path in packed.rglob("*") if path.is_file()
    )
    assert structure.hashes_and_headers_verified

    parity = results[1][1]
    assert isinstance(parity, PackedReferenceParityEvaluationResult)
    assert parity.passed
    assert parity.layer_count == 1
    assert parity.output_elements == 3
    assert parity.maximum_absolute_error == 0.0
