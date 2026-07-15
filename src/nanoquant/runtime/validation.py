"""Bounded-memory correctness validation for logical runtime artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from nanoquant.runtime.artifact import open_logical_artifact
from nanoquant.runtime.reference import DenseReferenceBackend, FactorizedReferenceBackend


class ReferenceParityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReferenceParityResult:
    artifact: Path
    layer_count: int
    output_elements: int
    absolute_tolerance: float
    maximum_absolute_error: float
    maximum_error_layer: str


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ReferenceParityError(f"reference validation does not support input dtype: {name}") from error


def validate_logical_reference_parity(
    artifact_root: str | Path,
    *,
    absolute_tolerance: float = 0.03125,
) -> ReferenceParityResult:
    """Compare dense reconstruction and factorized execution for every logical layer."""

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("reference parity absolute tolerance must be finite and non-negative")
    artifact = open_logical_artifact(artifact_root, verify_hashes=True)
    dense = DenseReferenceBackend()
    factorized = FactorizedReferenceBackend()
    maximum_error = 0.0
    maximum_layer = ""
    output_elements = 0
    with torch.no_grad():
        for block in artifact.manifest.blocks:
            for entry in block.layers:
                state = artifact.load_layer(entry.spec.name)
                value = torch.linspace(
                    -0.5,
                    0.5,
                    state.spec.in_features,
                    dtype=torch.float32,
                ).to(dtype=_torch_dtype(state.spec.scale_dtype)).reshape(1, -1)
                expected = dense.linear(value, dense.prepare(state, "cpu")).float()
                actual = factorized.linear(value, factorized.prepare(state, "cpu")).float()
                if not bool(torch.all(torch.isfinite(expected))) or not bool(
                    torch.all(torch.isfinite(actual))
                ):
                    raise ReferenceParityError(
                        f"reference backend produced a non-finite value: {state.spec.name}"
                    )
                error = float(torch.max(torch.abs(expected - actual)).item())
                output_elements += actual.numel()
                if error > maximum_error or not maximum_layer:
                    maximum_error = error
                    maximum_layer = state.spec.name
                del actual, expected, state, value
    if maximum_error > absolute_tolerance:
        raise ReferenceParityError(
            "logical reference backends differ beyond the absolute tolerance: "
            f"{maximum_error} > {absolute_tolerance} at {maximum_layer}"
        )
    return ReferenceParityResult(
        artifact.root,
        artifact.manifest.layer_count,
        output_elements,
        absolute_tolerance,
        maximum_error,
        maximum_layer,
    )
