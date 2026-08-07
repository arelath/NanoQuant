"""Materialize one exact mixed-product-code MLP policy into durable dense overlays."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch

from nanoquant.infrastructure.io_utils import (
    atomic_workspace,
    atomic_write_json,
    hash_file,
)
from nanoquant.infrastructure.probe_reconstruction_cache import (
    ProbeReconstructionCache,
)
from nanoquant.infrastructure.safetensors_io import SAFETENSORS

BLOCK_COUNT = 26
MLP_PATHS = {
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
OPTION_PATTERN = re.compile(r"^right_product_codebook_k16_free(?P<rows>\d+)_outliers(?P<outliers>\d+)$")


@dataclass(frozen=True, slots=True)
class MaterializationSettings:
    model_revision: str
    calibration_state: str
    baseline_rank: int
    candidate_rank: int
    outer_iterations: int
    inner_iterations: int
    regularization: float
    penalty_schedule: str
    convergence_check_interval: int
    codebook_update_interval: int
    codebook_freeze_fraction: float
    assignment_batch_words: int
    linear_assignment_sweeps: int
    corrected_assignment_candidates: int
    scale_fit_passes: int
    calibration_shrinkage: float
    seed: int


@dataclass(frozen=True, slots=True)
class MaterializationJob:
    job_id: str
    block: int
    projections: tuple[str, ...]
    right_free_rows: tuple[tuple[str, int], ...]
    fixed_outliers: tuple[int, ...]
    settings: MaterializationSettings

    @property
    def free_rows_by_projection(self) -> dict[str, int]:
        return dict(self.right_free_rows)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON receipt must be an object: {path}")
    return cast(dict[str, Any], payload)


def _selected_rows(selection: dict[str, Any]) -> tuple[int, int]:
    option = selection.get("option")
    if not isinstance(option, str):
        raise ValueError("allocation selection has no option identity")
    match = OPTION_PATTERN.fullmatch(option)
    if match is None:
        raise ValueError(f"allocation selection is not a product-code option: {option}")
    return int(match.group("rows")), int(match.group("outliers"))


def _settings_from_protocol(protocol: dict[str, Any]) -> MaterializationSettings:
    binary_search = protocol.get("binary_search")
    if not isinstance(binary_search, dict) or binary_search.get("enabled") is not True:
        raise ValueError("source sweep protocol did not enable binary search")
    if protocol.get("codebook_mode") != "product-right" or protocol.get("index_widths") != [16]:
        raise ValueError("source sweep protocol is not k16 right-product code")
    return MaterializationSettings(
        model_revision=str(protocol["model_revision"]),
        calibration_state=str(Path(str(protocol["calibration_state"])).resolve()),
        baseline_rank=int(protocol["baseline_rank"]),
        candidate_rank=int(protocol["candidate_rank"]),
        outer_iterations=int(protocol["outer_iterations"]),
        inner_iterations=int(protocol["inner_iterations"]),
        regularization=float(protocol["regularization"]),
        penalty_schedule=str(protocol["penalty_schedule"]),
        convergence_check_interval=int(protocol["convergence_check_interval"]),
        codebook_update_interval=int(protocol["codebook_update_interval"]),
        codebook_freeze_fraction=float(protocol["codebook_freeze_fraction"]),
        assignment_batch_words=int(protocol["assignment_batch_words"]),
        linear_assignment_sweeps=int(protocol["linear_assignment_sweeps"]),
        corrected_assignment_candidates=int(protocol["corrected_assignment_candidates"]),
        scale_fit_passes=int(protocol["scale_fit_passes"]),
        calibration_shrinkage=float(protocol["calibration_shrinkage"]),
        seed=int(protocol["seed"]),
    )


def _load_source_protocol(
    path: Path,
    *,
    block: int,
    projection: str,
    option: str,
) -> tuple[dict[str, Any], MaterializationSettings, tuple[int, ...]]:
    payload = _read_json(path)
    protocol = payload.get("protocol")
    results = payload.get("results")
    if (
        not isinstance(protocol, dict)
        or not isinstance(results, dict)
        or option not in results
        or protocol.get("block") != block
        or protocol.get("projection") != projection
    ):
        raise ValueError(f"source sweep receipt is incompatible: {path}")
    outliers_value = protocol.get("fixed_outlier_indices")
    if (
        not isinstance(outliers_value, list)
        or not outliers_value
        or any(not isinstance(value, int) or value < 0 for value in outliers_value)
        or len(outliers_value) != len(set(outliers_value))
    ):
        raise ValueError(f"source sweep outlier identity is invalid: {path}")
    return protocol, _settings_from_protocol(protocol), tuple(outliers_value)


def load_policy_jobs(allocation_path: Path) -> tuple[dict[str, Any], tuple[MaterializationJob, ...]]:
    allocation = _read_json(allocation_path)
    if allocation.get("schema_version") != 3 or allocation.get("status") != "completed":
        raise ValueError("mixed allocation receipt is not a completed schema-v3 result")
    if float(allocation["effective_bpw"]) > float(allocation["target_bpw"]):
        raise ValueError("mixed allocation exceeds its target BPW")
    inputs = allocation.get("inputs")
    selections_value = allocation.get("selections")
    if not isinstance(inputs, dict) or not isinstance(selections_value, list):
        raise ValueError("mixed allocation input or selection inventory is invalid")
    down_dir = Path(str(inputs["down_dir"]))
    mlp_dir = Path(str(inputs["mlp_dir"]))
    selections: dict[tuple[int, str], dict[str, Any]] = {}
    for value in selections_value:
        if not isinstance(value, dict):
            raise ValueError("mixed allocation selection must be an object")
        block = value.get("block")
        projection = value.get("projection")
        key = (block, projection)
        if (
            not isinstance(block, int)
            or not 0 <= block < BLOCK_COUNT
            or projection not in MLP_PATHS
            or key in selections
        ):
            raise ValueError("mixed allocation selection inventory is invalid")
        selections[cast(tuple[int, str], key)] = value
    expected = {(block, projection) for block in range(BLOCK_COUNT) for projection in MLP_PATHS}
    if set(selections) != expected:
        raise ValueError("mixed allocation must choose all 78 MLP matrices exactly once")

    jobs: list[MaterializationJob] = []
    for block in range(BLOCK_COUNT):
        protocols: dict[str, dict[str, Any]] = {}
        settings: dict[str, MaterializationSettings] = {}
        outliers: dict[str, tuple[int, ...]] = {}
        rows: dict[str, int] = {}
        for projection in MLP_PATHS:
            selection = selections[(block, projection)]
            selected_rows, outlier_count = _selected_rows(selection)
            option = cast(str, selection["option"])
            path = (
                down_dir / f"block-{block:02d}.json"
                if projection == "down"
                else mlp_dir / f"block-{block:02d}-{projection}.json"
            )
            protocol, current_settings, current_outliers = _load_source_protocol(
                path,
                block=block,
                projection=projection,
                option=option,
            )
            if len(current_outliers) != outlier_count:
                raise ValueError(f"selected option outlier count differs from source receipt: {path}")
            available_rows = protocol.get("right_free_row_counts")
            if not isinstance(available_rows, list) or selected_rows not in available_rows:
                raise ValueError(f"selected free rows were not measured by source sweep: {path}")
            protocols[projection] = protocol
            settings[projection] = current_settings
            outliers[projection] = current_outliers
            rows[projection] = selected_rows
        if settings["gate"] == settings["up"] and outliers["gate"] == outliers["up"]:
            jobs.append(
                MaterializationJob(
                    job_id=f"block-{block:02d}-gate-up",
                    block=block,
                    projections=("gate", "up"),
                    right_free_rows=(("gate", rows["gate"]), ("up", rows["up"])),
                    fixed_outliers=outliers["gate"],
                    settings=settings["gate"],
                )
            )
        else:
            jobs.extend(
                MaterializationJob(
                    job_id=f"block-{block:02d}-{projection}",
                    block=block,
                    projections=(projection,),
                    right_free_rows=((projection, rows[projection]),),
                    fixed_outliers=outliers[projection],
                    settings=settings[projection],
                )
                for projection in ("gate", "up")
            )
        jobs.append(
            MaterializationJob(
                job_id=f"block-{block:02d}-down",
                block=block,
                projections=("down",),
                right_free_rows=(("down", rows["down"]),),
                fixed_outliers=outliers["down"],
                settings=settings["down"],
            )
        )
    return allocation, tuple(jobs)


def build_probe_command(
    job: MaterializationJob,
    *,
    python: Path,
    probe: Path,
    model: Path,
    snapshot: Path,
    calibration_state: Path,
    output: Path,
    reconstruction_cache: Path,
    model_revision: str,
    device: str,
) -> list[str]:
    settings = job.settings
    command = [
        str(python),
        str(probe),
        "--model",
        str(model),
        "--snapshot",
        str(snapshot),
        "--calibration-state",
        str(calibration_state),
        "--output",
        str(output),
        "--reconstruction-cache",
        str(reconstruction_cache),
        "--model-revision",
        model_revision,
        "--block",
        str(job.block),
        "--baseline-rank",
        str(settings.baseline_rank),
        "--candidate-rank",
        str(settings.candidate_rank),
        "--fixed-outlier-indices",
        ",".join(str(index) for index in job.fixed_outliers),
        "--index-width",
        "16",
        "--codebook-mode",
        "product",
        "--corrections-per-word",
        "0",
        "--outer-iterations",
        str(settings.outer_iterations),
        "--inner-iterations",
        str(settings.inner_iterations),
        "--regularization",
        str(settings.regularization),
        "--penalty-schedule",
        settings.penalty_schedule,
        "--convergence-check-interval",
        str(settings.convergence_check_interval),
        "--codebook-update-interval",
        str(settings.codebook_update_interval),
        "--codebook-freeze-fraction",
        str(settings.codebook_freeze_fraction),
        "--assignment-batch-words",
        str(settings.assignment_batch_words),
        "--linear-assignment-sweeps",
        str(settings.linear_assignment_sweeps),
        "--corrected-assignment-candidates",
        str(settings.corrected_assignment_candidates),
        "--scale-fit-passes",
        str(settings.scale_fit_passes),
        "--calibration-shrinkage",
        str(settings.calibration_shrinkage),
        "--seed",
        str(settings.seed),
        "--binary-search",
        "--wikitext-samples",
        "1",
        "--wikitext-offset",
        "0",
        "--sequence-length",
        "512",
        "--device",
        device,
        "--local-files-only",
    ]
    if job.projections == ("gate", "up"):
        command.extend(
            (
                "--projections",
                "gate,up",
                "--right-free-rows-by-projection",
                ",".join(f"{projection}:{rows}" for projection, rows in job.right_free_rows),
                "--transpose-matrix",
            )
        )
    elif len(job.projections) == 1:
        projection = job.projections[0]
        command.extend(
            (
                "--projection",
                projection,
                "--right-free-rows",
                str(job.right_free_rows[0][1]),
            )
        )
        if projection in {"gate", "up"}:
            command.append("--transpose-matrix")
    else:
        raise ValueError(f"unsupported materialization projection group: {job.projections}")
    return command


def _expected_unit_keys(job: MaterializationJob) -> dict[str, str]:
    if len(job.projections) == 1:
        return {job.projections[0]: str(job.block)}
    return {projection: f"{job.block}:{projection}" for projection in job.projections}


def validate_job_receipt(
    path: Path,
    job: MaterializationJob,
    reconstruction_cache: Path,
) -> dict[str, str]:
    payload = _read_json(path)
    candidate = payload.get("candidate")
    fixed_outliers = payload.get("fixed_outlier_columns")
    cache = payload.get("reconstruction_cache")
    if (
        payload.get("status") != "completed"
        or payload.get("blocks") != [job.block]
        or payload.get("projections") != list(job.projections)
        or not isinstance(candidate, dict)
        or candidate.get("right_free_rows_by_projection") != job.free_rows_by_projection
        or not isinstance(fixed_outliers, dict)
        or fixed_outliers.get("indices") != list(job.fixed_outliers)
        or not isinstance(cache, dict)
        or Path(str(cache.get("directory"))).resolve() != reconstruction_cache.resolve()
    ):
        raise ValueError(f"materialization job receipt is incompatible: {path}")
    keys = cache.get("keys_by_unit")
    expected_units = _expected_unit_keys(job)
    if not isinstance(keys, dict) or set(keys) != set(expected_units.values()):
        raise ValueError(f"materialization cache-key inventory is incomplete: {path}")
    result = {
        projection: str(keys[unit])
        for projection, unit in expected_units.items()
        if isinstance(keys.get(unit), str) and keys[unit]
    }
    if set(result) != set(expected_units):
        raise ValueError(f"materialization cache keys are invalid: {path}")
    return result


def _load_cached_reconstructions(
    jobs: tuple[MaterializationJob, ...],
    job_receipts: Path,
    reconstruction_cache: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    cache = ProbeReconstructionCache(reconstruction_cache)
    baseline: dict[str, torch.Tensor] = {}
    candidate: dict[str, torch.Tensor] = {}
    cache_keys: dict[str, str] = {}
    for job in jobs:
        receipt_path = job_receipts / f"{job.job_id}.json"
        keys = validate_job_receipt(receipt_path, job, reconstruction_cache)
        for projection, key in keys.items():
            manifest_path = reconstruction_cache / key / "manifest.json"
            manifest = _read_json(manifest_path)
            identity = manifest.get("identity")
            if not isinstance(identity, dict):
                raise ValueError(f"cache identity is missing: {manifest_path}")
            expected_rows = job.free_rows_by_projection[projection]
            if (
                identity.get("block") != job.block
                or identity.get("projection") != projection
                or identity.get("projection_path") != MLP_PATHS[projection]
                or identity.get("right_free_rows") != expected_rows
                or identity.get("fixed_outlier_indices") != list(job.fixed_outliers)
                or cache.key(identity) != key
            ):
                raise ValueError(f"cache identity differs from materialization job: {manifest_path}")
            entry = cache.load(identity)
            if entry is None:
                raise ValueError(f"cache entry vanished: {manifest_path}")
            tensor_name = f"model.layers.{job.block}.{MLP_PATHS[projection]}.weight"
            if tensor_name in baseline:
                raise ValueError(f"materialization repeats a tensor: {tensor_name}")
            baseline[tensor_name] = entry.baseline
            candidate[tensor_name] = entry.candidate
            cache_keys[tensor_name] = key
    expected_count = BLOCK_COUNT * len(MLP_PATHS)
    if len(baseline) != expected_count or set(baseline) != set(candidate):
        raise ValueError("materialization did not recover all 78 MLP matrices")
    return baseline, candidate, cache_keys


def _write_overlay(destination: Path, arm: str, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    destination.mkdir(parents=True)
    normalized = {
        name: value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous() for name, value in tensors.items()
    }
    tensor_path = destination / "weights.safetensors"
    SAFETENSORS.save(normalized, tensor_path)
    manifest = {
        "schema_version": 1,
        "arm": arm,
        "layer_count": len(normalized),
        "blocks": list(range(BLOCK_COUNT)),
        "tensor_sha256": hash_file(tensor_path),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
            for name, value in normalized.items()
        },
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def write_overlay_bundle(
    destination: Path,
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    with atomic_workspace(destination) as temporary:
        manifests = {
            "free_words": _write_overlay(temporary / "free-words", "free_words", baseline),
            "corrected_codebook": _write_overlay(
                temporary / "corrected-codebook",
                "corrected_codebook",
                candidate,
            ),
        }
    return manifests


def validate_overlay_bundle(destination: Path) -> dict[str, dict[str, Any]]:
    expected_names = {
        f"model.layers.{block}.{path}.weight" for block in range(BLOCK_COUNT) for path in MLP_PATHS.values()
    }
    manifests: dict[str, dict[str, Any]] = {}
    for arm, directory in (
        ("free_words", destination / "free-words"),
        ("corrected_codebook", destination / "corrected-codebook"),
    ):
        manifest_path = directory / "manifest.json"
        tensor_path = directory / "weights.safetensors"
        if not manifest_path.is_file() or not tensor_path.is_file():
            raise ValueError(f"materialized overlay is incomplete: {directory}")
        manifest = _read_json(manifest_path)
        inventory = manifest.get("tensors")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("arm") != arm
            or manifest.get("layer_count") != len(expected_names)
            or manifest.get("blocks") != list(range(BLOCK_COUNT))
            or manifest.get("tensor_sha256") != hash_file(tensor_path)
            or not isinstance(inventory, dict)
            or set(inventory) != expected_names
        ):
            raise ValueError(f"materialized overlay identity or hash is invalid: {directory}")
        manifests[arm] = manifest
    return manifests


def _summary_identity(args: argparse.Namespace, allocation: dict[str, Any]) -> dict[str, Any]:
    return {
        "allocation": str(args.allocation.resolve()),
        "allocation_sha256": hash_file(args.allocation),
        "allocation_total_bits": int(allocation["total_bits"]),
        "allocation_effective_bpw": float(allocation["effective_bpw"]),
        "model": str(args.model.resolve()),
        "snapshot": str(args.snapshot.resolve()),
        "calibration_state": str(args.calibration_state.resolve()),
        "reconstruction_cache": str(args.reconstruction_cache.resolve()),
        "model_revision": args.model_revision,
        "device": args.device,
    }


def run(args: argparse.Namespace) -> int:
    allocation, jobs = load_policy_jobs(args.allocation)
    if any(job.settings.model_revision != args.model_revision for job in jobs):
        raise ValueError("materialization revision differs from source sweep receipts")
    if any(Path(job.settings.calibration_state) != args.calibration_state.resolve() for job in jobs):
        raise ValueError("materialization calibration state differs from source sweeps")
    identity = _summary_identity(args, allocation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts = args.output_dir / "jobs"
    receipts.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "manifest.json"
    previous = _read_json(summary_path) if summary_path.is_file() else None
    if previous is not None and previous.get("identity") != identity:
        raise ValueError("existing materialization manifest has a different identity")
    if previous is not None and previous.get("status") == "completed":
        _load_cached_reconstructions(jobs, receipts, args.reconstruction_cache)
        validate_overlay_bundle(args.output_dir / "overlays")
        return 0

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": previous.get("started_at", _utc_now()) if previous else _utc_now(),
        "updated_at": _utc_now(),
        "identity": identity,
        "job_count": len(jobs),
        "completed_jobs": [],
        "jobs": [
            {
                "job_id": job.job_id,
                "block": job.block,
                "projections": list(job.projections),
                "right_free_rows_by_projection": job.free_rows_by_projection,
                "fixed_outlier_indices": list(job.fixed_outliers),
                "receipt": str((receipts / f"{job.job_id}.json").resolve()),
            }
            for job in jobs
        ],
    }
    atomic_write_json(summary_path, summary)
    probe = Path(__file__).resolve().with_name("probe_corrected_codebook_splice.py")
    python = Path(sys.executable).resolve()
    try:
        for job in jobs:
            output = receipts / f"{job.job_id}.json"
            if output.is_file():
                validate_job_receipt(output, job, args.reconstruction_cache)
            else:
                summary["current_job"] = job.job_id
                summary["updated_at"] = _utc_now()
                atomic_write_json(summary_path, summary)
                print(
                    f"starting {job.job_id} rows={job.free_rows_by_projection} outliers={job.fixed_outliers}",
                    flush=True,
                )
                command = build_probe_command(
                    job,
                    python=python,
                    probe=probe,
                    model=args.model.resolve(),
                    snapshot=args.snapshot.resolve(),
                    calibration_state=args.calibration_state.resolve(),
                    output=output.resolve(),
                    reconstruction_cache=args.reconstruction_cache.resolve(),
                    model_revision=args.model_revision,
                    device=args.device,
                )
                subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
                validate_job_receipt(output, job, args.reconstruction_cache)
            completed = cast(list[str], summary["completed_jobs"])
            if job.job_id not in completed:
                completed.append(job.job_id)
            summary["updated_at"] = _utc_now()
            atomic_write_json(summary_path, summary)
            print(f"completed {job.job_id}", flush=True)

        baseline, candidate, cache_keys = _load_cached_reconstructions(
            jobs,
            receipts,
            args.reconstruction_cache,
        )
        overlays = args.output_dir / "overlays"
        overlay_manifests = (
            validate_overlay_bundle(overlays)
            if overlays.exists()
            else write_overlay_bundle(overlays, baseline, candidate)
        )
        summary["overlays"] = {
            "directory": str(overlays.resolve()),
            "free_words": overlay_manifests["free_words"],
            "corrected_codebook": overlay_manifests["corrected_codebook"],
            "cache_keys_by_tensor": cache_keys,
        }
        summary["status"] = "completed"
        summary["completed_at"] = _utc_now()
        summary["updated_at"] = summary["completed_at"]
        summary.pop("current_job", None)
        atomic_write_json(summary_path, summary)
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["updated_at"] = _utc_now()
        atomic_write_json(summary_path, summary)
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--reconstruction-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
