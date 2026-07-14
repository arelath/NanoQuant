"""Replay retained layers through the current and exact legacy NanoQuant ADMM solvers."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.factorization import ADMMResult, factorize_admm
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, LayerResult, TensorRef
from nanoquant.domain.objectives import DiagonalObjective
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity, load_committed_layer
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource
from nanoquant.infrastructure.tensor_store import LocalTensorStore

LegacyFactorizer = Callable[..., dict[str, Any]]
OUTLIER_SELECTION_ARTIFACT = "outlier-selection"


def _load_legacy_factorizer(source_path: Path) -> LegacyFactorizer:
    spec = importlib.util.spec_from_file_location("_nanoquant_legacy_admm_oracle", source_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load legacy factorizer source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    cast(Any, spec.loader).exec_module(module)
    function = getattr(module, "factorize_admm_nanoquant", None)
    if not callable(function):
        raise ValueError(f"legacy source has no factorize_admm_nanoquant function: {source_path}")
    return cast(LegacyFactorizer, function)


def _read_tensor(store: LocalTensorStore, reference: TensorRef) -> torch.Tensor:
    with store.read(reference, "cpu") as value:
        return value.clone()


def _latest_layer_record(run_output: Path, block: int, path: str) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (run_output / "state" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
) -> tuple[LayerResult, torch.Tensor, float, dict[str, Any]]:
    record = _latest_layer_record(run_output, block, path)
    identity = from_dict(CommitIdentity, record["identity"], path="identity")
    reference = ArtifactRef(ArtifactTypes.LAYER_RESULT, str(record["artifact_id"]), 1)
    layer = load_committed_layer(reference, artifacts, identity).result
    with source.read_tensor(layer.plan.source_weight, "cpu") as value:
        source_shape = tuple(value.shape)
    if source_shape != layer.plan.source_weight.spec.shape:
        raise ValueError(f"source tensor shape changed for {layer.layer}")
    output_importance = _read_tensor(tensors, layer.plan.objective.output_importance)
    descriptor = json.loads(
        (artifacts.path_for(reference.artifact_id) / "descriptor.json").read_text(encoding="utf-8")
    )
    return layer, output_importance, float(descriptor["committed_at"]), {
        "identity": to_dict(identity),
        "layer_result": to_dict(reference),
        "source_weight": to_dict(layer.plan.source_weight),
        "output_importance": to_dict(layer.plan.objective.output_importance),
        "retained_factorization": to_dict(layer.attempts[layer.accepted_attempt].result),
    }


def _outlier_artifact_paths(artifacts: LocalArtifactStore) -> list[tuple[str, float, Path]]:
    result = []
    for descriptor_path in artifacts.root.rglob("descriptor.json"):
        try:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if descriptor.get("artifact_type") == OUTLIER_SELECTION_ARTIFACT:
            result.append(
                (
                    str(descriptor["artifact_id"]),
                    float(descriptor["committed_at"]),
                    descriptor_path.parent / "tensors.safetensors",
                )
            )
    return result


def _load_factorization_input(
    layer: LayerResult,
    layer_committed_at: float,
    artifacts: LocalArtifactStore,
    tensors: LocalTensorStore,
    candidates: list[tuple[str, float, Path]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    frozen_outliers = layer.frozen_state.outliers
    if frozen_outliers is None:
        raise ValueError(f"captured layer has no retained outlier state: {layer.layer}")
    expected_indices = _read_tensor(tensors, frozen_outliers.indices).long()
    matches: list[tuple[str, float, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for artifact_id, committed_at, tensor_path in candidates:
        if committed_at >= layer_committed_at:
            continue
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if "residual_weight" not in handle.keys():
                continue
            residual_slice = handle.get_slice("residual_weight")
            if tuple(residual_slice.get_shape()) != layer.plan.source_weight.spec.shape:
                continue
            indices = handle.get_tensor("indices").long()
            if not torch.equal(indices, expected_indices):
                continue
            residual = handle.get_tensor("residual_weight")
            if bool(torch.count_nonzero(residual[:, indices])):
                continue
            artifacts.validate(artifact_id)
            matches.append(
                (
                    artifact_id,
                    committed_at,
                    residual.clone(),
                    handle.get_tensor("factor_input_importance").clone(),
                    handle.get_tensor("factor_generator_state").clone(),
                )
            )
    if not matches:
        raise ValueError(f"found no preceding outlier input for {layer.layer}")
    artifact_id, _, residual, input_importance, generator_state = max(matches, key=lambda item: item[1])
    expected_importance = _read_tensor(tensors, layer.plan.objective.input_importance).float()
    if layer.plan.outliers.removed_column_importance == "zero":
        floor = expected_importance.median().clamp_min(1e-12) * 1e-4
        expected_importance[expected_indices] = floor
    if not torch.equal(input_importance.float(), expected_importance):
        raise ValueError(f"factor input importance does not match the layer policy: {layer.layer}")
    return residual, input_importance, generator_state, artifact_id


def _tensor_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    left32 = left.float()
    right32 = right.float()
    difference = left32 - right32
    return {
        "shape_equal": True,
        "exact": bool(torch.equal(left, right)),
        "agreement": float((left == right).float().mean()),
        "maximum_absolute_difference": float(difference.abs().max()) if difference.numel() else 0.0,
        "relative_l2_difference": float(difference.norm() / right32.norm().clamp_min(1e-30)),
    }


def _legacy_tensors(result: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "left_latent": cast(torch.Tensor, result["A_latent"]).mT.contiguous(),
        "right_latent": cast(torch.Tensor, result["B_latent"]).contiguous(),
        "left_binary": cast(torch.Tensor, result["A"]).mT.contiguous(),
        "right_binary": cast(torch.Tensor, result["B"]).contiguous(),
        "scale_pre": cast(torch.Tensor, result["scale_pre"]).reshape(-1),
        "scale_mid": cast(torch.Tensor, result["scale_mid"]).reshape(-1),
        "scale_post": cast(torch.Tensor, result["scale_post"]).reshape(-1),
        "reconstruction": cast(torch.Tensor, result["W_final"]).contiguous(),
    }


def _rewrite_tensors(result: ADMMResult) -> dict[str, torch.Tensor]:
    return {
        "left_latent": result.left_latent,
        "right_latent": result.right_latent,
        "left_binary": result.left_binary,
        "right_binary": result.right_binary,
        "scale_pre": result.scale_pre,
        "scale_mid": result.scale_mid,
        "scale_post": result.scale_post,
        "reconstruction": result.reconstruction,
    }


def _retained_comparison(
    rewrite: dict[str, torch.Tensor], reference: ArtifactRef, artifacts: LocalArtifactStore, device: str
) -> dict[str, Any]:
    artifacts.validate(reference.artifact_id)
    path = artifacts.path_for(reference.artifact_id) / "tensors.safetensors"
    comparisons = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in (
            "left_latent",
            "right_latent",
            "left_binary",
            "right_binary",
            "scale_pre",
            "scale_mid",
            "scale_post",
        ):
            comparisons[name] = _tensor_comparison(rewrite[name], handle.get_tensor(name).to(device))
    return {
        "tensors": comparisons,
        "exact": all(comparison.get("exact", False) for comparison in comparisons.values()),
    }


def compare_layer(
    layer: LayerResult,
    residual_cpu: torch.Tensor,
    input_importance_cpu: torch.Tensor,
    output_importance_cpu: torch.Tensor,
    generator_state: torch.Tensor,
    legacy_factorizer: LegacyFactorizer,
    artifacts: LocalArtifactStore,
    *,
    device: str,
    outer_iterations: int,
    inner_iterations: int,
    regularization: float,
    objective_relative_tolerance: float,
) -> dict[str, Any]:
    residual = residual_cpu.to(device)
    input_importance = input_importance_cpu.float().to(device)
    output_importance = output_importance_cpu.float().to(device)
    is_transposed = residual.shape[0] < residual.shape[1]

    torch.cuda.set_rng_state(generator_state, device=device)
    torch.cuda.synchronize(device)
    legacy_started = time.perf_counter()
    legacy_result = legacy_factorizer(
        residual,
        input_importance,
        output_importance,
        mid_rank=layer.plan.rank,
        outer_iters=outer_iterations,
        inner_iters=inner_iterations,
        reg=regularization,
        is_transpose=is_transposed,
        rho_scheduler="cubic",
        print_admm_steps=False,
    )
    torch.cuda.synchronize(device)
    legacy_seconds = time.perf_counter() - legacy_started
    legacy_final_state = torch.cuda.get_rng_state(device)

    generator = torch.Generator(device=device)
    generator.set_state(generator_state)
    torch.cuda.synchronize(device)
    rewrite_started = time.perf_counter()
    rewrite_result = factorize_admm(
        residual,
        input_importance,
        output_importance,
        layer.plan.rank,
        generator,
        outer_iterations=outer_iterations,
        inner_iterations=inner_iterations,
        regularization=regularization,
        penalty_schedule="cubic",
    )
    torch.cuda.synchronize(device)
    rewrite_seconds = time.perf_counter() - rewrite_started

    legacy_tensors = _legacy_tensors(legacy_result)
    rewrite_tensors = _rewrite_tensors(rewrite_result)
    comparisons = {
        name: _tensor_comparison(rewrite_tensors[name], legacy_tensors[name]) for name in rewrite_tensors
    }
    objective = DiagonalObjective(input_importance, output_importance)
    legacy_normalized = float(objective.normalized_error(residual, legacy_tensors["reconstruction"]))
    rewrite_normalized = float(objective.normalized_error(residual, rewrite_tensors["reconstruction"]))
    objective_delta = abs(rewrite_normalized - legacy_normalized) / max(abs(legacy_normalized), 1e-30)
    retained_reference = layer.attempts[layer.accepted_attempt].result
    retained = _retained_comparison(rewrite_tensors, retained_reference, artifacts, device)
    rng = _tensor_comparison(generator.get_state(), legacy_final_state)
    old_new_exact = all(comparison.get("exact", False) for comparison in comparisons.values())
    passed = old_new_exact and rng.get("exact", False) and objective_delta <= objective_relative_tolerance
    return {
        "block": layer.layer.block.index,
        "layer": layer.layer.path,
        "shape": list(residual.shape),
        "rank": layer.plan.rank,
        "legacy_transposed_wide_matrix": is_transposed,
        "wall_seconds": {"legacy": legacy_seconds, "rewrite": rewrite_seconds},
        "objective": {
            "legacy_weighted_normalized_error": legacy_normalized,
            "rewrite_weighted_normalized_error": rewrite_normalized,
            "relative_difference": objective_delta,
            "relative_tolerance": objective_relative_tolerance,
        },
        "old_new_tensors": comparisons,
        "old_new_exact": old_new_exact,
        "rng_final_state": rng,
        "retained_rewrite_replay": retained,
        "passed": passed,
    }


def _layer_spec(value: str) -> tuple[str, int, str]:
    try:
        role, location = value.split("=", 1)
        block, path = location.split(":", 1)
        if not role or not path:
            raise ValueError
        return role, int(block), path
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("layer must have the form ROLE=BLOCK:PATH") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_output", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-name", default="google/gemma-3-1b-it")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--layer", type=_layer_spec, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--objective-relative-tolerance", type=float, default=0.022)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("real retained factorization parity currently requires a CUDA device")

    run_output = args.run_output.resolve()
    legacy_source = args.legacy_source.resolve()
    legacy_factorizer = _load_legacy_factorizer(legacy_source)
    artifacts = LocalArtifactStore(run_output / "artifacts")
    tensors = LocalTensorStore(artifacts)
    source = SafetensorsModelSource(
        args.snapshot,
        source=args.source_name,
        revision=args.revision,
        verify_hashes=True,
    )
    candidates = _outlier_artifact_paths(artifacts)
    results = []
    with acquire_device_lease(args.device):
        for role, block, path in args.layer:
            layer, output_importance, layer_committed_at, references = _load_layer(
                run_output, source, artifacts, tensors, block, path
            )
            residual, input_importance, generator_state, outlier_artifact = _load_factorization_input(
                layer, layer_committed_at, artifacts, tensors, candidates
            )
            result = compare_layer(
                layer,
                residual,
                input_importance,
                output_importance,
                generator_state,
                legacy_factorizer,
                artifacts,
                device=args.device,
                outer_iterations=args.outer_iterations,
                inner_iterations=args.inner_iterations,
                regularization=args.regularization,
                objective_relative_tolerance=args.objective_relative_tolerance,
            )
            result["fixture_role"] = role
            references["outlier_selection"] = {
                "artifact_type": OUTLIER_SELECTION_ARTIFACT,
                "artifact_id": outlier_artifact,
                "schema_version": 1,
            }
            result["references"] = references
            results.append(result)
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "protocol": {
            "run_output": str(run_output),
            "snapshot": str(args.snapshot.resolve()),
            "source_name": args.source_name,
            "revision": args.revision,
            "legacy_source": str(legacy_source),
            "legacy_source_sha256": hash_file(legacy_source),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "outer_iterations": args.outer_iterations,
            "inner_iterations": args.inner_iterations,
            "regularization": args.regularization,
            "objective_relative_tolerance": args.objective_relative_tolerance,
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
