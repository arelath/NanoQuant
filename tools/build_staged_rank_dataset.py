"""Reusable, resumable multi-stage rank responses on original source weights.

Fresh ADMM plus scale fitting at every tail rank; identical full-stage prefixes
are shared. No activation tuning, model inference, embedding edits, or KL claims.
Publishes the existing allocation dataset contract with all measured curve knots
and additional per-member/stage evidence. KL anchors are deliberately optional.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_real_block_tabu import _load_cases
from safetensors.torch import load_file, save_file

from nanoquant.application.kl_budget import load_kl_budget_profile
from nanoquant.config.codec import to_dict
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.residual_factorization import residual_stage_ranks
from nanoquant.domain.scale_fit import fit_scales, reconstruct
from nanoquant.domain.seeds import logical_seed
from nanoquant.infrastructure.allocation_profile_dataset import publish_allocation_profile_dataset
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.runs import RunLease
from nanoquant.resident_quantization import _legacy_cuda_numerics

FRACTIONS = (.25, .5, .75, .875, 1., 1.125, 1.25, 1.5, 1.75, 1.875,
             2., 2.125, 2.25, 2.5, 2.75, 2.875, 3.)
OWNERS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.o_proj", "self_attn.attn_qkv")
METRICS = ("raw_nmse", "objective_nmse", "imatrix_noise_nmse")


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def record_hash(record):
    payload = {key: value for key, value in record.items() if key != "content_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_record(path, record):
    record["content_sha256"] = record_hash(record)
    atomic_write_json(path, record)


def validate_model_hash(source_hashes, digest):
    if not source_hashes or any(not value.startswith(f"sha256:{digest}#")
                                for values in source_hashes.values() for value in values):
        raise ValueError("source model file differs from the committed source weight identity")


def rank_grid(shape, fractions=FRACTIONS, multiple=32):
    full = min(shape) // multiple * multiple
    if full <= 0 or not fractions or any(not 0 < f <= 3 for f in fractions):
        raise ValueError("invalid rank grid")
    return tuple(sorted({max(multiple, round(full * f / multiple) * multiple) for f in fractions}))


def monotone_curve(ranks, errors):
    """Equal-weight decreasing isotonic fit in log NMSE, then log-linear knots."""
    if (len(ranks) != len(errors) or len(ranks) < 2 or tuple(ranks) != tuple(sorted(set(ranks)))
            or any(not math.isfinite(e) or e <= 0 for e in errors)):
        raise ValueError("curve requires ordered unique ranks and positive errors")
    pools = []
    for error in errors:
        pools.append([math.log(error), 1])
        while len(pools) > 1 and pools[-2][0] < pools[-1][0]:
            right, nr = pools.pop()
            left, nl = pools.pop()
            pools.append([(left * nl + right * nr) / (nl + nr), nl + nr])
    fitted = [math.exp(mean) for mean, count in pools for _ in range(count)]
    return {"kind": "decreasing-isotonic-log-nmse-piecewise-linear-v1", "ranks": list(ranks),
            "fitted_nmse": fitted, "measured_nmse": list(errors),
            "log_rmse": math.sqrt(sum(math.log(a / b) ** 2 for a, b in zip(errors, fitted, strict=True)) / len(errors)),
            "extrapolation": "unsupported"}


def curve_error(curve, rank):
    ranks, errors = curve["ranks"], curve["fitted_nmse"]
    if not ranks[0] <= rank <= ranks[-1]:
        raise ValueError("rank is outside measured coverage")
    for i in range(1, len(ranks)):
        if rank <= ranks[i]:
            fraction = (rank - ranks[i - 1]) / (ranks[i] - ranks[i - 1])
            return math.exp(math.log(errors[i - 1]) * (1 - fraction) + math.log(errors[i]) * fraction)
    return errors[-1]


def allocator_curve(name, baseline, curve):
    ranks, errors = curve["ranks"], curve["fitted_nmse"]
    return {"unit_pattern": name, "calibrated_rank_floor_fraction": ranks[0] / baseline,
            "calibrated_rank_ceiling_fraction": ranks[-1] / baseline,
            "segments": [{"maximum_rank_fraction": ranks[i] / baseline,
                          "beta_per_rank": max(0., math.log(errors[i - 1] / errors[i])
                                               / (2 * (ranks[i] - ranks[i - 1])))}
                         for i in range(1, len(ranks))]}


@torch.no_grad()
def error_metrics(weight, target, columns, rows):
    delta = (weight.float() - target.float()).square()
    energy = target.float().square()
    column_weights = columns.float().clamp_min(1e-12)
    column_weights = column_weights / column_weights.mean()
    objective = columns.float().clamp_min(1e-12)[None, :] * rows.float().clamp_min(1e-12)[:, None]
    raw = float(delta.sum())
    norm = float(energy.sum())
    weighted = float((delta * objective).sum())
    weighted_norm = float((energy * objective).sum())
    result = {"raw_squared_error": raw, "weight_norm_squared": norm,
              "objective_squared_error": weighted, "objective_weight_norm_squared": weighted_norm,
              "raw_nmse": raw / max(norm, 1e-30),
              "objective_nmse": weighted / max(weighted_norm, 1e-30),
              "imatrix_noise_nmse": float((delta * column_weights[None, :]).sum()) / max(norm, 1e-30),
              "sensitivity_energy": math.sqrt(weighted_norm / target.numel())}
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite reconstruction metrics")
    return result


@torch.no_grad()
def fit_stage(target, columns, rows, rank, seed, parameters, scale_config):
    fitted = factorize_admm_with_parameters(
        target, columns, rows, rank, torch.Generator(device=target.device).manual_seed(seed), parameters,
    )
    scales = fit_scales(target, fitted.left_binary, fitted.right_binary,
                        fitted.scale_pre, fitted.scale_mid, fitted.scale_post, columns, rows,
                        **{key: value for key, value in scale_config.items() if key != "enabled"})
    persisted = reconstruct(fitted.left_binary, fitted.right_binary,
                            scales.scale_pre.bfloat16(), scales.scale_mid.bfloat16(), scales.scale_post.bfloat16())
    optimized = reconstruct(fitted.left_binary, fitted.right_binary,
                            scales.scale_pre, scales.scale_mid, scales.scale_post)
    return optimized, persisted, {
        "iterations_completed": fitted.iterations_completed, "stopped_early": fitted.stopped_early,
        "scale_fit_accepted": scales.accepted, "scale_fit_before_error": scales.before_error,
        "scale_fit_after_error": scales.after_error,
    }


def save_prefix(path, fit_prediction, stored_prediction):
    temporary = path.with_suffix(".tmp")
    save_file({"fit_prediction": fit_prediction.cpu().contiguous(),
               "stored_prediction": stored_prediction.cpu().contiguous()}, str(temporary))
    temporary.replace(path)
    return hash_file(path)


def measure_unit(case, root, parameters, scale_config, seed, identity_key, source_hashes, device):
    key = f"{case.block}:{case.name}"
    stem = f"block-{case.block:02d}.{case.name}"
    path = root / "measurements" / f"{stem}.json"
    cache_paths = [root / "work" / f"{stem}.prefix-{i}.safetensors" for i in range(1, 4)]
    if (case.outlier_indices is not None and case.outlier_indices.numel()) or case.patch is not None:
        raise ValueError("this dataset protocol requires no outlier or patch sidepaths")
    target = case.target.to(device).bfloat16()
    columns, rows = case.input_importance.to(device), case.output_importance.to(device)
    shape = tuple(target.shape)
    full = min(shape) // 32 * 32
    ranks = rank_grid(shape)
    base_seed = logical_seed(seed, "staged-rank-dataset", case.block, case.name, 0)
    identity = {"dataset_identity": identity_key, "unit_id": key, "shape": list(shape),
                "target_sha256": tensor_hash(target), "columns_sha256": tensor_hash(columns),
                "rows_sha256": tensor_hash(rows), "logical_seed": base_seed,
                "source_weight_hashes": source_hashes}
    record = {"identity": identity, "members": [to_dict(member) for member in case.members],
              "member_rows": list(case.member_rows), "full_rank": full, "points": [], "complete": False}
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("content_sha256") != record_hash(record):
            raise ValueError("unit measurement checksum differs")
        if record["identity"] != identity:
            raise ValueError(f"changed unit identity: {key}")
        if record["complete"]:
            for cache_path in cache_paths:
                cache_path.unlink(missing_ok=True)
            return record
    completed = {point["rank"] for point in record["points"]}
    if sorted(completed) != list(ranks[:len(completed)]):
        raise ValueError("completed rank points are not a contiguous prefix")
    fit_prefix = torch.zeros_like(target, dtype=torch.float32)
    stored_prefix = torch.zeros_like(fit_prefix)
    full_count = 0
    if record.get("prefix"):
        receipt = record["prefix"]
        cache_path = cache_paths[receipt["stages"] - 1]
        if hash_file(cache_path) != receipt["sha256"]:
            raise ValueError("prefix checkpoint changed")
        saved = load_file(str(cache_path), device=device)
        fit_prefix, stored_prefix = saved["fit_prediction"], saved["stored_prediction"]
        full_count = receipt["stages"]
    for rank in ranks:
        if rank in completed:
            continue
        started = time.perf_counter()
        stage_ranks = residual_stage_ranks(*shape, rank, multiple=32)
        index = len(stage_ranks) - 1
        if index != full_count:
            raise ValueError("rank grid must include every full-stage boundary")
        stage_seed = base_seed if index == 0 else logical_seed(
            base_seed, "residual-factor-stage", case.block, case.name, index,
        )
        residual = (target.float() - fit_prefix).bfloat16()
        fitted, stored, details = fit_stage(residual, columns, rows, stage_ranks[-1],
                                             stage_seed, parameters, scale_config)
        fit_prediction, stored_prediction = fit_prefix + fitted, stored_prefix + stored
        point = {"rank": rank, "rank_fraction": rank / full, "stage_ranks": list(stage_ranks),
                 "stage_seeds": [base_seed if i == 0 else logical_seed(
                     base_seed, "residual-factor-stage", case.block, case.name, i,
                 ) for i in range(len(stage_ranks))],
                 "payload_bits": factor_bit_cost(*shape, rank, scale_bits=16, rank_alignment=32,
                                                 residual_stages=True).total,
                 "stored": error_metrics(stored_prediction, target, columns, rows),
                 "fit_precision": error_metrics(fit_prediction, target, columns, rows),
                 "members": [], "tail_fit": details}
        offset = 0
        for member, size in zip(case.members, case.member_rows, strict=True):
            sl = slice(offset, offset + size)
            point["members"].append({"target": f"{member.block.index}:{member.path}",
                                     **error_metrics(stored_prediction[sl], target[sl], columns, rows[sl])})
            offset += size
        point["seconds"] = time.perf_counter() - started
        record["points"].append(point)
        if stage_ranks[-1] == full:
            fit_prefix, stored_prefix = fit_prediction, stored_prediction
            full_count += 1
            record["prefix"] = {"stages": full_count,
                                "sha256": save_prefix(cache_paths[full_count - 1], fit_prefix, stored_prefix)}
        save_record(path, record)
        print(f"{key} rank={rank} stages={stage_ranks} weighted_nmse={point['stored']['objective_nmse']:.6g} "
              f"{point['seconds']:.1f}s", flush=True)
        del residual, fitted, stored, fit_prediction, stored_prediction
    record["curves"] = {metric: monotone_curve(ranks, [p["stored"][metric] for p in record["points"]])
                        for metric in METRICS}
    record["member_curves"] = {
        member["target"]: {metric: monotone_curve(ranks, [p["members"][i][metric] for p in record["points"]])
                           for metric in METRICS}
        for i, member in enumerate(record["points"][0]["members"])
    }
    record["complete"] = True
    save_record(path, record)
    # Only this tool's obsolete transient prefix; scalar evidence remains durable.
    for cache_path in cache_paths:
        cache_path.unlink(missing_ok=True)
    return record


def publish(root, identity, records, plan_payload, recipe, profile):
    expected_units = set(identity["unit_ids"])
    measured_units = {record["identity"]["unit_id"] for record in records}
    if (not expected_units or len(records) != len(expected_units) or measured_units != expected_units
            or any(not r["complete"] for r in records)):
        raise ValueError("refusing to publish incomplete all-unit dataset")
    store = LocalArtifactStore(root / "artifacts")
    with store.begin_write("staged-rank-measurement-protocol") as writer:
        atomic_write_json(writer.path / "protocol.json", identity)
        protocol_id = writer.commit().artifact_id
    units = []
    for record in records:
        with store.begin_write("staged-rank-unit-measurements") as writer:
            atomic_write_json(writer.path / "measurements.json", record)
            unit_id = writer.commit().artifact_id
        baseline = record["full_rank"] * 2
        point = next(p for p in record["points"] if p["rank"] == baseline)
        metrics = point["stored"]
        key = record["identity"]["unit_id"]
        name = key.split(":", 1)[1]
        evidence = {"schema_version": 4, "unit_id": key, "block": int(key.split(":")[0]), "name": name,
                    "members": record["members"], "source_weight_hashes": record["identity"]["source_weight_hashes"],
                    "baseline_rank": baseline,
                    "response_curve": allocator_curve(name, baseline, record["curves"]["objective_nmse"]),
                    "response_points": [{"rank": p["rank"], "raw_squared_error": p["stored"]["raw_squared_error"],
                                         "normalized_squared_error": p["stored"]["raw_nmse"],
                                         "weighted_normalized_squared_error": p["stored"]["objective_nmse"]}
                                        for p in record["points"]],
                    "raw_squared_error": metrics["raw_squared_error"],
                    "normalized_squared_error": metrics["raw_nmse"],
                    "weighted_normalized_squared_error": metrics["objective_nmse"],
                    "relative_frobenius_error": math.sqrt(metrics["raw_nmse"]),
                    "member_squared_errors": [m["raw_squared_error"] for m in point["members"]],
                    "member_normalized_squared_errors": [m["raw_nmse"] for m in point["members"]],
                    "member_sensitivity_energies": [m["sensitivity_energy"] for m in point["members"]],
                    "member_weight_norms_squared": [m["weight_norm_squared"] for m in point["members"]],
                    "logical_seed": record["identity"]["logical_seed"]}
        units.append({"evidence": evidence, "member_shapes": [[r, record["identity"]["shape"][1]]
                                                               for r in record["member_rows"]],
                      "source_probe_id": unit_id, "staged_measurements": record})
    payload = {"schema_version": 1, "model": plan_payload["model"],
               "source_run": identity["run_output"], "source_plan_id": identity["source_plan_id"],
               "source_profile_id": protocol_id, "source_probe_plan_id": protocol_id,
               "probe_protocol": identity, "source_recipe": recipe, "units": units,
               "kl_profile_key": None if profile is None else profile.profile_key}
    return publish_allocation_profile_dataset(root, payload, profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="pinned source model.safetensors")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-units", type=int, help="bounded feasibility invocation; resume the same output later")
    parser.add_argument("--kl-profile", type=Path, help="optional separately measured compatible KL anchors")
    args = parser.parse_args()
    root, run = args.output.resolve(), args.run_output.resolve()
    audit = json.loads(args.audit.read_text())
    if not audit.get("complete"):
        raise ValueError("requires a complete source resident artifact audit")
    manifest = json.loads((run / "manifest.json").read_text())
    recipe = manifest["resolved_config"]["canonical_run_config"]
    parameters = AdmmParameters(**recipe["factorization"]["admm"])
    scale_config = recipe["factorization"]["scale_fit"]
    if not scale_config["enabled"] or not recipe["factorization"]["residual_stages"]["enabled"]:
        raise ValueError("source must enable residual stages and scale fitting")
    state = json.loads((run / "state/preprocessing.json").read_text())
    if (Path(audit["run_output"]).resolve() != run
            or audit["identity"]["plan_hash"] != state["plan"]["artifact_id"]
            or audit["identity"]["config_hash"] != state["resident_config_hash"]):
        raise ValueError("source audit belongs to a different run identity")
    artifacts = LocalArtifactStore(run / "artifacts", use_persistent_validation_cache=False)
    artifacts.validate(state["plan"]["artifact_id"])
    plan_payload = json.loads((artifacts.path_for(state["plan"]["artifact_id"]) / "plan.json").read_text())
    blocks = [entry["block"]["index"] for entry in plan_payload["blocks"]]
    if not blocks or blocks != list(range(len(blocks))) or audit["completed_blocks"] != blocks:
        raise ValueError("source audit and plan must cover the same contiguous block inventory")
    source_hashes = {}
    for block in plan_payload["blocks"]:
        for layer in block["layers"]:
            key = f"{block['block']['index']}:{layer['layer']['path']}"
            source_hashes[key] = [layer["source_weight"]["content_hash"]]
        for group in block["shared_input_groups"]:
            source_hashes[f"{block['block']['index']}:{group['name']}"] = [m["weight"]["content_hash"]
                                                                       for m in group["members"]]
    if set(source_hashes) != {f"{block}:{owner}" for block in blocks for owner in OWNERS}:
        raise ValueError("source must have the five supported physical owners in each block")
    profile = None if args.kl_profile is None else load_kl_budget_profile(args.kl_profile)
    model_digest = hash_file(args.model)
    validate_model_hash(source_hashes, model_digest)
    code_paths = [Path(__file__), Path(__file__).with_name("probe_real_block_tabu.py")]
    code_paths += [Path("src/nanoquant") / p for p in (
        "domain/factorization.py", "domain/scale_fit.py", "domain/linear_math.py", "domain/planning.py",
        "domain/residual_factorization.py", "domain/seeds.py", "domain/calibration_math.py")]
    identity = {"schema_version": 1, "protocol": "source-weight-staged-admm-scale-fit-bf16-v1",
                "run_output": str(run), "manifest_sha256": hash_file(run / "manifest.json"),
                "journal_sha256": hash_file(run / "state/journal.jsonl"), "audit_sha256": hash_file(args.audit),
                "source_plan_id": state["plan"]["artifact_id"], "model_sha256": model_digest,
                "unit_ids": sorted(source_hashes),
                "model": str(args.model.resolve()), "seed": args.seed, "device": args.device,
                "fractions": list(FRACTIONS), "admm": asdict(parameters), "scale_fit": scale_config,
                "code_sha256": {str(path): hash_file(path) for path in code_paths},
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "kl_profile_key": None if profile is None else profile.profile_key,
                "scope": "pre-activation-tuning weight reconstruction; no measured per-rank KL; embeddings untouched",
                "representation": "independent scale triples; one-bit signs, BF16 scales; sum reconstructed in FP32",
                "prefix": "full-stage optimization prefixes shared exactly; fresh final tail fit at each rank",
                "cost": "exact logical packed factor/scale payload bits, excludes container metadata/alignment"}
    root.mkdir(parents=True, exist_ok=True)
    with RunLease(root / ".active-lease.json"):
        manifest_path = root / "measurement-manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
            raise ValueError("dataset identity changed; choose a new directory")
        atomic_write_json(manifest_path, identity)
        identity_key = hash_file(manifest_path)
        for child in ("measurements", "work"):
            (root / child).mkdir(exist_ok=True)
        records = []
        with acquire_device_lease(args.device), _legacy_cuda_numerics():
            for block in blocks:
                _, cases = _load_cases(run, args.model, (block,), OWNERS, len(blocks))
                if {case.name for case in cases} != set(OWNERS) or len(cases) != len(OWNERS):
                    raise ValueError("incomplete physical unit inventory")
                for case in cases:
                    key = f"{case.block}:{case.name}"
                    records.append(measure_unit(case, root, parameters, scale_config, args.seed,
                                                identity_key, source_hashes[key], args.device))
                    if args.max_units is not None and len(records) >= args.max_units:
                        print(f"Feasibility stop: {len(records)} complete units; dataset not published", flush=True)
                        return
                    gc.collect()
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                del cases
        dataset = publish(root, identity, records, plan_payload, recipe, profile)
        print(f"Published {len(records)} units, {sum(len(r['points']) for r in records)} points: "
              f"{dataset.reference.artifact_id}", flush=True)


if __name__ == "__main__":
    main()
