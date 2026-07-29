"""Probe transpose-aware and cross-block factor sharing on pinned model weights.

The tool is intentionally analysis-only. It uses the production ADMM and scale-fit
math, records scalar evidence after every group, and does not create runtime or
compression artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.models import BitCost
from nanoquant.domain.planning import factor_bit_cost
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.device_lease import acquire_device_lease

PINNED_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
PROJECTION_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
SUPPORTED_ARMS = (
    "attention-reciprocal",
    "adjacent-qkv",
    "adjacent-gate",
    "adjacent-up",
    "adjacent-down",
)


@dataclass(frozen=True, slots=True)
class MemberSpec:
    block: int
    projection: str
    transpose: bool = False

    @property
    def tensor_name(self) -> str:
        return f"model.layers.{self.block}.{PROJECTION_PATHS[self.projection]}.weight"

    @property
    def label(self) -> str:
        suffix = "^T" if self.transpose else ""
        return f"{self.block}:{self.projection}{suffix}"

    @property
    def calibration_path(self) -> str:
        return f"block.{self.block}.{PROJECTION_PATHS[self.projection]}"


@dataclass(frozen=True, slots=True)
class GroupSpec:
    label: str
    members: tuple[MemberSpec, ...]
    rank_adjustment: int = 0


@dataclass(frozen=True, slots=True)
class TopologySpec:
    comparison: str
    variant: str
    location: str
    groups: tuple[GroupSpec, ...]

    @property
    def key(self) -> str:
        return f"{self.comparison}|{self.location}|{self.variant}"


@dataclass(frozen=True, slots=True)
class ProbeProtocol:
    schema_version: int
    model_revision: str
    target_bpw: float
    rank_alignment: int
    scale_bits: int
    outer_iterations: int
    inner_iterations: int
    regularization: float
    penalty_schedule: str
    convergence_check_interval: int
    transpose_wide: bool
    scale_fit_passes: int
    seed: int
    device: str
    calibration_state: str | None = None
    calibration_shrinkage: float = 0.0


def maximum_rank_for_budget(
    out_features: int,
    in_features: int,
    target_bits: int,
    *,
    scale_bits: int,
    rank_alignment: int,
    fixed_bits: int = 0,
) -> int:
    """Return the largest logical rank whose packed factor cost fits."""

    if min(out_features, in_features, target_bits, scale_bits, fixed_bits) < 0 or rank_alignment <= 0:
        raise ValueError("rank budget inputs are invalid")
    maximum = min(out_features, in_features)
    accepted = 0
    for rank in range(1, maximum + 1):
        cost = factor_bit_cost(
            out_features,
            in_features,
            rank,
            scale_bits=scale_bits,
            rank_alignment=rank_alignment,
        ).total
        if cost + fixed_bits > target_bits:
            if rank_alignment == 1:
                break
            continue
        accepted = rank
    if accepted == 0:
        raise ValueError("target bit budget cannot fund rank one")
    return accepted


def attention_topologies(block: int, vo_rank_shift: int = 0) -> tuple[TopologySpec, TopologySpec]:
    if vo_rank_shift < 0:
        raise ValueError("V/O rank shift must not be negative")
    q = MemberSpec(block, "q")
    k = MemberSpec(block, "k")
    v = MemberSpec(block, "v")
    o_t = MemberSpec(block, "o", True)
    return (
        TopologySpec(
            "attention-reciprocal",
            "current-qkv-plus-o",
            str(block),
            (GroupSpec("qkv", (q, k, v)), GroupSpec("o-transpose", (o_t,))),
        ),
        TopologySpec(
            "attention-reciprocal",
            (
                "candidate-qk-plus-vo"
                if vo_rank_shift == 0
                else f"candidate-qk-plus-vo-vo-shift-{vo_rank_shift}"
            ),
            str(block),
            (
                GroupSpec("qk", (q, k), -vo_rank_shift),
                GroupSpec("v-o-transpose", (v, o_t), vo_rank_shift),
            ),
        ),
    )


def adjacent_topologies(projection: str, first: int, second: int) -> tuple[TopologySpec, TopologySpec]:
    if projection not in {"qkv", "gate", "up", "down"}:
        raise ValueError(f"unsupported adjacent projection group: {projection}")
    comparison = f"adjacent-{projection}"
    if projection == "qkv":
        first_members = tuple(MemberSpec(first, name) for name in ("q", "k", "v"))
        second_members = tuple(MemberSpec(second, name) for name in ("q", "k", "v"))
    else:
        transpose = projection == "down"
        first_members = (MemberSpec(first, projection, transpose),)
        second_members = (MemberSpec(second, projection, transpose),)
    return (
        TopologySpec(
            comparison,
            "current-separate",
            f"{first}-{second}",
            (GroupSpec(f"{first}", first_members), GroupSpec(f"{second}", second_members)),
        ),
        TopologySpec(
            comparison,
            "candidate-shared",
            f"{first}-{second}",
            (GroupSpec(f"{first}-{second}", first_members + second_members),),
        ),
    )


def requested_topologies(
    arms: tuple[str, ...],
    attention_blocks: tuple[int, ...],
    adjacent_pairs: tuple[tuple[int, int], ...],
    reciprocal_vo_rank_shifts: tuple[int, ...] = (0,),
) -> tuple[TopologySpec, ...]:
    result: list[TopologySpec] = []
    if "attention-reciprocal" in arms:
        for block in attention_blocks:
            current, _candidate = attention_topologies(block)
            result.append(current)
            for shift in reciprocal_vo_rank_shifts:
                result.append(attention_topologies(block, shift)[1])
    for arm in arms:
        if not arm.startswith("adjacent-"):
            continue
        projection = arm.removeprefix("adjacent-")
        for first, second in adjacent_pairs:
            result.extend(adjacent_topologies(projection, first, second))
    return tuple(result)


def _parse_ints(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    result = tuple(int(item.strip()) for item in value.split(","))
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("block indices must not be negative")
    return result


def _parse_pairs(value: str) -> tuple[tuple[int, int], ...]:
    if not value.strip():
        return ()
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        parts = item.strip().split("-")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("adjacent pairs must use FIRST-SECOND syntax")
        first, second = (int(part) for part in parts)
        if first < 0 or second != first + 1:
            raise argparse.ArgumentTypeError("each adjacent pair must contain consecutive non-negative blocks")
        result.append((first, second))
    return tuple(result)


def _parse_arms(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(result) - set(SUPPORTED_ARMS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported probe arms: {', '.join(unknown)}")
    return result


def _member_shape(handle: Any, member: MemberSpec) -> tuple[int, int]:
    shape = tuple(int(item) for item in handle.get_slice(member.tensor_name).get_shape())
    if len(shape) != 2:
        raise ValueError(f"member is not a matrix: {member.label}")
    return (shape[1], shape[0]) if member.transpose else (shape[0], shape[1])


def group_shape(handle: Any, group: GroupSpec) -> tuple[int, int, int]:
    shapes = tuple(_member_shape(handle, member) for member in group.members)
    inputs = {shape[1] for shape in shapes}
    if len(inputs) != 1:
        raise ValueError(f"group members do not share a factor input axis: {group.label}")
    rows = sum(shape[0] for shape in shapes)
    columns = inputs.pop()
    elements = sum(shape[0] * shape[1] for shape in shapes)
    if rows * columns != elements:
        raise ValueError(f"group is not a lossless row stack: {group.label}")
    return rows, columns, elements


def _planned_group_rank(
    handle: Any,
    group: GroupSpec,
    protocol: ProbeProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[int, int]:
    out_features, in_features, source_elements = group_shape(handle, group)
    target_bits = math.floor(source_elements * protocol.target_bpw)
    extra_scale_bits = _extra_pre_scale_bits(handle, group, profiles, protocol.scale_bits)
    base_rank = maximum_rank_for_budget(
        out_features,
        in_features,
        target_bits,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
        fixed_bits=extra_scale_bits,
    )
    rank = base_rank + group.rank_adjustment
    if rank <= 0 or rank > min(out_features, in_features):
        raise ValueError(f"rank adjustment puts {group.label} outside its matrix dimensions")
    return rank, extra_scale_bits


def _logical_seed(seed: int, group_key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{group_key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _protocol_hash(protocol: ProbeProtocol) -> str:
    payload = json.dumps(_protocol_payload(protocol), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _protocol_payload(protocol: ProbeProtocol) -> dict[str, Any]:
    payload = asdict(protocol)
    if protocol.calibration_state is None:
        # Preserve compatibility with schema-v1 unweighted probe checkpoints.
        payload.pop("calibration_state")
        payload.pop("calibration_shrinkage")
    return payload


def _load_output(path: Path, protocol: ProbeProtocol) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "protocol_hash": _protocol_hash(protocol),
            "protocol": _protocol_payload(protocol),
            "groups": {},
            "topologies": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != _protocol_hash(protocol):
        raise ValueError("existing output uses a different probe protocol")
    if not isinstance(payload.get("groups"), dict) or not isinstance(payload.get("topologies"), dict):
        raise ValueError("existing output is missing probe result maps")
    return payload


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _group_cache_key(topology: TopologySpec, group: GroupSpec, rank: int) -> str:
    members = ",".join(member.label for member in group.members)
    return f"{topology.comparison}|{topology.location}|{topology.variant}|{group.label}|r{rank}|{members}"


def load_calibration_profiles(
    state_directory: Path,
    shrinkage: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    manifest = json.loads((state_directory / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sample_count"])
    if sample_count <= 0:
        raise ValueError("calibration state sample count must be positive")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("calibration state manifest has no layers")
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with safe_open(str(state_directory / "state.safetensors"), framework="pt", device="cpu") as handle:
        for index, layer in enumerate(layers):
            path = str(layer["path"])
            input_importance = shrink_importance(
                handle.get_tensor(f"layer_{index}.inputs.total").float() / sample_count,
                shrinkage,
            )
            output_importance = shrink_importance(
                handle.get_tensor(f"layer_{index}.outputs.total").float() / sample_count,
                shrinkage,
            )
            result[path] = (input_importance, output_importance)
    return result


def _member_importances(
    member: MemberSpec,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        input_importance, output_importance = profiles[member.calibration_path]
    except KeyError as exc:
        raise ValueError(f"calibration state is missing {member.calibration_path}") from exc
    return (output_importance, input_importance) if member.transpose else (input_importance, output_importance)


def _unique_input_profile_count(
    group: GroupSpec,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> int:
    if profiles is None:
        return 1
    unique: list[torch.Tensor] = []
    for member in group.members:
        input_importance, _output_importance = _member_importances(member, profiles)
        normalized = input_importance.float() / input_importance.float().mean().clamp_min(1e-30)
        if not any(torch.allclose(normalized, existing, rtol=1e-5, atol=1e-7) for existing in unique):
            unique.append(normalized)
    return len(unique)


def _extra_pre_scale_bits(
    handle: Any,
    group: GroupSpec,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    scale_bits: int,
) -> int:
    _rows, inputs, _elements = group_shape(handle, group)
    return max(0, _unique_input_profile_count(group, profiles) - 1) * inputs * scale_bits


def _materialize_group(
    handle: Any,
    group: GroupSpec,
    device: str,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    raw_values: list[torch.Tensor] = []
    objective_values: list[torch.Tensor] = []
    normalizers: list[tuple[torch.Tensor, torch.Tensor]] = []
    rows: list[int] = []
    for member in group.members:
        value = handle.get_tensor(member.tensor_name)
        if member.transpose:
            value = value.mT.contiguous()
        raw = value.to(device)
        raw_values.append(raw)
        rows.append(value.shape[0])
        if profiles is None:
            input_scale = torch.ones(value.shape[1], device=device, dtype=torch.float32)
            output_scale = torch.ones(value.shape[0], device=device, dtype=torch.float32)
        else:
            input_importance, output_importance = _member_importances(member, profiles)
            if input_importance.numel() != value.shape[1] or output_importance.numel() != value.shape[0]:
                raise ValueError(f"calibration dimensions differ for {member.label}")
            input_scale = input_importance.to(device).float().clamp_min(1e-30).sqrt()
            output_scale = output_importance.to(device).float().clamp_min(1e-30).sqrt()
        # BF16 materialization preserves the production mixed-precision ADMM
        # state while making member-specific diagonal objectives row-stackable.
        objective_values.append(
            (raw.float() * output_scale[:, None] * input_scale[None, :]).to(raw.dtype)
        )
        normalizers.append((input_scale, output_scale))
    objective = objective_values[0] if len(objective_values) == 1 else torch.cat(objective_values, dim=0)
    raw_target = raw_values[0] if len(raw_values) == 1 else torch.cat(raw_values, dim=0)
    return objective, raw_target, tuple(rows), tuple(normalizers)


def _restore_original_space(
    prediction: torch.Tensor,
    member_rows: tuple[int, ...],
    normalizers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> torch.Tensor:
    members: list[torch.Tensor] = []
    offset = 0
    for rows, (input_scale, output_scale) in zip(member_rows, normalizers, strict=True):
        member = prediction[offset : offset + rows].float()
        members.append(member / output_scale[:, None] / input_scale[None, :])
        offset += rows
    return members[0] if len(members) == 1 else torch.cat(members, dim=0)


def _error_metrics(target: torch.Tensor, prediction: torch.Tensor, member_rows: tuple[int, ...]) -> dict[str, Any]:
    difference = target.float() - prediction.float()
    total_error = float(difference.square().sum())
    total_energy = float(target.float().square().sum())
    members: list[dict[str, float]] = []
    offset = 0
    for rows in member_rows:
        member_difference = difference[offset : offset + rows]
        member_target = target[offset : offset + rows].float()
        error = float(member_difference.square().sum())
        energy = float(member_target.square().sum())
        members.append(
            {
                "error_energy": error,
                "target_energy": energy,
                "normalized_rmse": math.sqrt(error / max(energy, 1e-30)),
            }
        )
        offset += rows
    return {
        "error_energy": total_error,
        "target_energy": total_energy,
        "normalized_rmse": math.sqrt(total_error / max(total_energy, 1e-30)),
        "members": members,
    }


def execute_group(
    handle: Any,
    topology: TopologySpec,
    group: GroupSpec,
    protocol: ProbeProtocol,
    profiles: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> dict[str, Any]:
    out_features, in_features, source_elements = group_shape(handle, group)
    target_bits = math.floor(source_elements * protocol.target_bpw)
    rank, extra_scale_bits = _planned_group_rank(handle, group, protocol, profiles)
    cost = factor_bit_cost(
        out_features,
        in_features,
        rank,
        scale_bits=protocol.scale_bits,
        rank_alignment=protocol.rank_alignment,
    ) + BitCost(scale_bits=extra_scale_bits)
    group_key = _group_cache_key(topology, group, rank)
    target, raw_target, member_rows, normalizers = _materialize_group(
        handle,
        group,
        protocol.device,
        profiles,
    )
    input_importance = torch.ones(in_features, device=protocol.device, dtype=torch.float32)
    output_importance = torch.ones(out_features, device=protocol.device, dtype=torch.float32)
    generator = torch.Generator(device=protocol.device).manual_seed(_logical_seed(protocol.seed, group_key))
    parameters = AdmmParameters(
        outer_iterations=protocol.outer_iterations,
        inner_iterations=protocol.inner_iterations,
        regularization=protocol.regularization,
        penalty_schedule=protocol.penalty_schedule,
        convergence_check_interval=protocol.convergence_check_interval,
        transpose_wide=protocol.transpose_wide,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(protocol.device)
        torch.cuda.synchronize(protocol.device)
    started = time.perf_counter()
    factorized = factorize_admm_with_parameters(
        target,
        input_importance,
        output_importance,
        rank,
        generator,
        parameters,
    )
    raw_metrics = _error_metrics(target, factorized.reconstruction, member_rows)
    fitted = fit_scales(
        target,
        factorized.left_binary,
        factorized.right_binary,
        factorized.scale_pre,
        factorized.scale_mid,
        factorized.scale_post,
        input_importance,
        output_importance,
        alternating_passes=protocol.scale_fit_passes,
    )
    if protocol.device.startswith("cuda"):
        torch.cuda.synchronize(protocol.device)
    wall_seconds = time.perf_counter() - started
    fitted_metrics = _error_metrics(target, fitted.reconstruction, member_rows)
    original_prediction = _restore_original_space(fitted.reconstruction, member_rows, normalizers)
    original_metrics = _error_metrics(raw_target, original_prediction, member_rows)
    peak_bytes = int(torch.cuda.max_memory_allocated(protocol.device)) if protocol.device.startswith("cuda") else 0
    iterations_completed = factorized.iterations_completed
    scale_fit_accepted = fitted.accepted
    scale_fit_rollback_reason = fitted.rollback_reason
    del factorized, fitted, target, raw_target, original_prediction, input_importance, output_importance
    if protocol.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "group_key": group_key,
        "label": group.label,
        "members": [asdict(member) | {"label": member.label} for member in group.members],
        "shape": [out_features, in_features],
        "member_rows": list(member_rows),
        "source_elements": source_elements,
        "target_bits": target_bits,
        "rank": rank,
        "bit_cost": asdict(cost),
        "actual_bpw": cost.total / source_elements,
        "input_scale_profiles": _unique_input_profile_count(group, profiles),
        "iterations_completed": iterations_completed,
        "raw": raw_metrics,
        "scale_fitted": fitted_metrics,
        "original_space": original_metrics,
        "scale_fit_accepted": scale_fit_accepted,
        "scale_fit_rollback_reason": scale_fit_rollback_reason,
        "wall_seconds": wall_seconds,
        "peak_device_bytes": peak_bytes,
    }


def summarize_topology(topology: TopologySpec, group_results: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    error = sum(float(group["scale_fitted"]["error_energy"]) for group in group_results)
    energy = sum(float(group["scale_fitted"]["target_energy"]) for group in group_results)
    original_metrics = tuple(group.get("original_space", group["scale_fitted"]) for group in group_results)
    original_error = sum(float(metrics["error_energy"]) for metrics in original_metrics)
    original_energy = sum(float(metrics["target_energy"]) for metrics in original_metrics)
    source_elements = sum(int(group["source_elements"]) for group in group_results)
    target_bits = sum(int(group["target_bits"]) for group in group_results)
    actual_bits = sum(int(group["bit_cost"]["binary_factor_bits"]) for group in group_results)
    actual_bits += sum(int(group["bit_cost"]["scale_bits"]) for group in group_results)
    actual_bits += sum(int(group["bit_cost"]["padding_bits"]) for group in group_results)
    return {
        "comparison": topology.comparison,
        "variant": topology.variant,
        "location": topology.location,
        "groups": [group["group_key"] for group in group_results],
        "error_energy": error,
        "target_energy": energy,
        "normalized_rmse": math.sqrt(error / max(energy, 1e-30)),
        "original_error_energy": original_error,
        "original_target_energy": original_energy,
        "original_normalized_rmse": math.sqrt(original_error / max(original_energy, 1e-30)),
        "source_elements": source_elements,
        "target_bits": target_bits,
        "actual_bits": actual_bits,
        "actual_bpw": actual_bits / source_elements,
        "unused_target_bits": target_bits - actual_bits,
        "wall_seconds": sum(float(group["wall_seconds"]) for group in group_results),
        "peak_device_bytes": max(int(group["peak_device_bytes"]) for group in group_results),
    }


def _comparison_lines(topologies: dict[str, Any]) -> tuple[str, ...]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in topologies.values():
        grouped.setdefault((str(result["comparison"]), str(result["location"])), []).append(result)
    lines: list[str] = []
    for (comparison, location), variants in sorted(grouped.items()):
        ordered = sorted(variants, key=lambda item: str(item["variant"]))
        if len(ordered) != 2:
            continue
        baseline = next((item for item in ordered if str(item["variant"]).startswith("current-")), None)
        candidate = next((item for item in ordered if str(item["variant"]).startswith("candidate-")), None)
        if baseline is None or candidate is None:
            continue
        delta = (float(candidate["normalized_rmse"]) / float(baseline["normalized_rmse"]) - 1.0) * 100
        lines.append(
            f"{comparison} {location}: {baseline['variant']}={baseline['normalized_rmse']:.6f}, "
            f"{candidate['variant']}={candidate['normalized_rmse']:.6f}, delta={delta:+.2f}%, "
            f"bpw={baseline['actual_bpw']:.6f}/{candidate['actual_bpw']:.6f}"
        )
    return tuple(lines)


def run(args: argparse.Namespace) -> int:
    calibration_state = None if args.calibration_state is None else str(args.calibration_state.resolve())
    protocol = ProbeProtocol(
        1,
        args.model_revision,
        args.target_bpw,
        args.rank_alignment,
        args.scale_bits,
        args.outer_iterations,
        args.inner_iterations,
        args.regularization,
        args.penalty_schedule,
        args.convergence_check_interval,
        True,
        args.scale_fit_passes,
        args.seed,
        args.device,
        calibration_state,
        args.calibration_shrinkage,
    )
    output = _load_output(args.output, protocol)
    topologies = requested_topologies(
        args.arms,
        args.attention_blocks,
        args.adjacent_pairs,
        args.reciprocal_vo_rank_shifts,
    )
    if not topologies:
        raise ValueError("the selected arms and block arguments produce no probe topologies")
    profiles = (
        None
        if args.calibration_state is None
        else load_calibration_profiles(args.calibration_state, args.calibration_shrinkage)
    )
    lease_context = acquire_device_lease(args.device) if args.device.startswith("cuda") else nullcontext()
    with lease_context, safe_open(str(args.model), framework="pt", device="cpu") as handle:
        for topology in topologies:
            group_results: list[dict[str, Any]] = []
            for group in topology.groups:
                out_features, in_features, _source_elements = group_shape(handle, group)
                rank, _extra_scale_bits = _planned_group_rank(handle, group, protocol, profiles)
                cache_key = _group_cache_key(topology, group, rank)
                cached = output["groups"].get(cache_key)
                if cached is None:
                    print(
                        f"running {topology.key} group={group.label} shape={out_features}x{in_features} rank={rank}",
                        flush=True,
                    )
                    cached = execute_group(handle, topology, group, protocol, profiles)
                    output["groups"][cache_key] = cached
                    _write_output(args.output, output)
                    print(
                        f"completed {group.label}: rmse={cached['scale_fitted']['normalized_rmse']:.6f} "
                        f"bpw={cached['actual_bpw']:.6f} wall={cached['wall_seconds']:.1f}s",
                        flush=True,
                    )
                else:
                    print(f"reusing {cache_key}", flush=True)
                group_results.append(cached)
            output["topologies"][topology.key] = summarize_topology(topology, tuple(group_results))
            _write_output(args.output, output)
    for line in _comparison_lines(output["topologies"]):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pinned model.safetensors path")
    parser.add_argument("--output", type=Path, required=True, help="Resumable JSON result path")
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--arms", type=_parse_arms, default=("attention-reciprocal",))
    parser.add_argument("--attention-blocks", type=_parse_ints, default=(16,))
    parser.add_argument("--adjacent-pairs", type=_parse_pairs, default=())
    parser.add_argument("--target-bpw", type=float, default=1.0)
    parser.add_argument("--rank-alignment", type=int, default=1)
    parser.add_argument("--scale-bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=400)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--calibration-state", type=Path)
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--reciprocal-vo-rank-shifts", type=_parse_ints, default=(0,))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
