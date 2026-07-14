"""Compare the rewrite diagonal objective with the exact legacy source function."""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.linear_math import functional_dense_reconstruction
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, LayerId, LayerResult, TensorRef
from nanoquant.domain.objectives import DiagonalObjective
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, load_committed_layer
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore

LegacyObjective = Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _extract_legacy_weighted_error(source_path: Path) -> tuple[LegacyObjective, int, int]:
    """Compile only the oracle function, avoiding imports from the legacy package."""

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "_weighted_weight_error"
        ),
        None,
    )
    if not isinstance(node, ast.FunctionDef):
        raise ValueError(f"legacy source has no _weighted_weight_error function: {source_path}")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    function = cast(LegacyObjective, namespace["_weighted_weight_error"])
    return function, node.lineno, node.end_lineno or node.lineno


def _read_tensor(store: LocalTensorStore, reference: TensorRef) -> torch.Tensor:
    with store.read(reference, "cpu") as value:
        return value.clone()


def _latest_layer_record(run_output: Path, block: int, path: str) -> dict[str, Any]:
    journal = run_output / "state" / "journal.jsonl"
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    matching = [
        record
        for record in records
        if record.get("kind") == "layer" and record.get("block") == block and record.get("layer") == path
    ]
    if not matching:
        raise ValueError(f"run has no committed layer {block}:{path}")
    return cast(dict[str, Any], matching[-1])


def _load_layer(
    run_output: Path,
    source: SafetensorsModelSource,
    artifacts: LocalArtifactStore,
    tensors: LocalTensorStore,
    block: int,
    path: str,
) -> tuple[LayerResult, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    record = _latest_layer_record(run_output, block, path)
    identity = from_dict(CommitIdentity, record["identity"], path="identity")
    reference = ArtifactRef(ArtifactTypes.LAYER_RESULT, str(record["artifact_id"]), 1)
    layer = load_committed_layer(reference, artifacts, identity).result
    frozen = layer.frozen_state
    if frozen.scales.mid is None:
        raise ValueError(f"committed layer has no middle scale: {block}:{path}")

    with source.read_tensor(layer.plan.source_weight, "cpu") as value:
        target = value.clone()
    input_importance = _read_tensor(tensors, layer.plan.objective.input_importance)
    output_importance = _read_tensor(tensors, layer.plan.objective.output_importance)
    left = _read_tensor(tensors, frozen.left_binary)
    right = _read_tensor(tensors, frozen.right_binary)
    scale_pre = _read_tensor(tensors, frozen.scales.pre)
    scale_mid = _read_tensor(tensors, frozen.scales.mid)
    scale_post = _read_tensor(tensors, frozen.scales.post)
    indices = values = outlier_scales = None
    if frozen.outliers is not None:
        indices = _read_tensor(tensors, frozen.outliers.indices)
        values = _read_tensor(tensors, frozen.outliers.values)
        if frozen.outliers.scales is not None:
            outlier_scales = _read_tensor(tensors, frozen.outliers.scales)
    prediction = functional_dense_reconstruction(
        left,
        right,
        scale_pre,
        scale_mid,
        scale_post,
        indices,
        values,
        outlier_scales,
    )
    references = {
        "identity": to_dict(identity),
        "layer_result": to_dict(reference),
        "source_weight": to_dict(layer.plan.source_weight),
        "input_importance": to_dict(layer.plan.objective.input_importance),
        "output_importance": to_dict(layer.plan.objective.output_importance),
        "frozen_state": to_dict(frozen),
    }
    return layer, target, prediction, input_importance, output_importance, references


def _comparison(
    candidate: float,
    reference: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    absolute = abs(candidate - reference)
    relative = absolute / max(abs(reference), 1e-30)
    return {
        "legacy": reference,
        "rewrite": candidate,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "passed": absolute <= absolute_tolerance + relative_tolerance * abs(reference),
    }


def compare_layer(
    layer: LayerId,
    target: torch.Tensor,
    prediction: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    legacy: LegacyObjective,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    objective = DiagonalObjective(input_importance, output_importance)
    with torch.no_grad():
        rewrite_error = float(objective.weighted_error(target, prediction))
        rewrite_target_norm = float(objective.weighted_error(target, torch.zeros_like(target)))
        rewrite_normalized = rewrite_error / max(rewrite_target_norm, objective.epsilon)
        legacy_error, legacy_target_norm, legacy_normalized = legacy(
            prediction,
            target,
            input_importance,
            output_importance,
        )
    comparisons = {
        "weighted_error": _comparison(
            rewrite_error,
            float(legacy_error),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
        "target_weighted_norm_squared": _comparison(
            rewrite_target_norm,
            float(legacy_target_norm),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
        "weighted_normalized_error": _comparison(
            rewrite_normalized,
            float(legacy_normalized),
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        ),
    }
    return {
        "block": layer.block.index,
        "layer": layer.path,
        "shape": list(target.shape),
        "input_importance_at_or_below_floor": int((input_importance <= 1e-12).sum()),
        "output_importance_at_or_below_floor": int((output_importance <= 1e-12).sum()),
        "comparisons": comparisons,
        "passed": all(item["passed"] for item in comparisons.values()),
    }


def _layer_spec(value: str) -> tuple[str, int, str]:
    try:
        role, location = value.split("=", 1)
        block, path = location.split(":", 1)
        if not role or not path:
            raise ValueError
        return role, int(block), path
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("layer must have the form ROLE=BLOCK:PATH") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_output", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-name", default="google/gemma-3-1b-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--layer", type=_layer_spec, action="append", required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=2e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_output = args.run_output.resolve()
    legacy_source = args.legacy_source.resolve()
    legacy, source_start, source_end = _extract_legacy_weighted_error(legacy_source)
    artifacts = LocalArtifactStore(run_output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    source = SafetensorsModelSource(
        args.snapshot,
        source=args.source_name,
        revision=args.revision,
        verify_hashes=True,
    )
    results = []
    for role, block, path in args.layer:
        layer, target, prediction, input_importance, output_importance, references = _load_layer(
            run_output, source, artifacts, tensors, block, path
        )
        result = compare_layer(
            layer.layer,
            target,
            prediction,
            input_importance,
            output_importance,
            legacy,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        result["fixture_role"] = role
        result["references"] = references
        results.append(result)
    payload = {
        "schema_version": 1,
        "protocol": {
            "run_output": str(run_output),
            "snapshot": str(args.snapshot.resolve()),
            "source_name": args.source_name,
            "revision": args.revision,
            "legacy_source": str(legacy_source),
            "legacy_source_sha256": hash_file(legacy_source),
            "legacy_function_lines": [source_start, source_end],
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
        },
        "layers": results,
        "passed": all(result["passed"] for result in results),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
