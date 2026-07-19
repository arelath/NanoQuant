"""Validate the immutable artifact graph and progress journal for a resident run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nanoquant.config.codec import ConfigDecodeError, from_dict
from nanoquant.domain.models import ArtifactRef, ArtifactTypes, BlockId, LayerId
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    artifact_type: str | None


@dataclass(frozen=True, slots=True)
class _LayerCommitPosition:
    identity: CommitIdentity
    layer: LayerId


@dataclass(frozen=True, slots=True)
class _TensorShape:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError("tensor shape dimensions must be positive")


@dataclass(frozen=True, slots=True)
class _SourceWeight:
    spec: _TensorShape


@dataclass(frozen=True, slots=True)
class _LayerPlanMetrics:
    source_weight: _SourceWeight


@dataclass(frozen=True, slots=True)
class _FrozenLayerMetrics:
    rank: int

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("frozen rank must be positive")


@dataclass(frozen=True, slots=True)
class _LayerMetrics:
    actual_bit_cost: dict[str, int]
    frozen_state: _FrozenLayerMetrics
    layer: LayerId
    plan: _LayerPlanMetrics

    def __post_init__(self) -> None:
        if not self.actual_bit_cost or any(
            not isinstance(name, str) or not name or type(value) is not int or value < 0
            for name, value in self.actual_bit_cost.items()
        ):
            raise ValueError("actual bit costs must be named non-negative integers")


@dataclass(frozen=True, slots=True)
class _GroupPlanMetrics:
    in_features: int
    out_features: int


@dataclass(frozen=True, slots=True)
class _GroupMetrics:
    actual_bit_cost: dict[str, int]
    frozen_state: _FrozenLayerMetrics
    block: BlockId
    name: str
    plan: _GroupPlanMetrics


@dataclass(frozen=True, slots=True)
class _BlockLossMetrics:
    final_frozen_pre_kd: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.final_frozen_pre_kd):
            raise ValueError("final frozen loss must be finite")


@dataclass(frozen=True, slots=True)
class _BlockMetrics:
    identity: CommitIdentity
    block: BlockId
    layers: tuple[_LayerMetrics, ...]
    shared_input_groups: tuple[_GroupMetrics, ...]
    losses: _BlockLossMetrics
    wall_seconds: float
    peak_gpu_bytes: int
    peak_host_bytes: int
    activation_generation: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if not self.layers and not self.shared_input_groups:
            raise ValueError("committed block must contain quantized units")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0:
            raise ValueError("block wall time must be finite and non-negative")
        if self.peak_gpu_bytes < 0 or self.peak_host_bytes < 0:
            raise ValueError("block memory peaks must be non-negative")


@dataclass(frozen=True, slots=True)
class RunValidationResult:
    run_output: str
    identity: dict[str, str]
    journal_records: int
    active_journal_records: int
    inactive_journal_records: int
    journal_identity_count: int
    layer_records: int
    block_records: int
    completed_blocks: tuple[int, ...]
    complete: bool
    artifacts_validated: int
    artifact_bytes: int
    artifacts_by_type: dict[str, int]
    retired_activation_generations: tuple[str, ...]
    committed_layer_count: int
    rank_sum: int
    quantized_parameters: int
    bit_cost_by_category: dict[str, int]
    effective_bpw: float
    block_wall_seconds: float
    peak_gpu_bytes: int
    peak_host_bytes: int
    final_frozen_pre_kd_losses: tuple[float, ...]


def _references(value: object) -> tuple[ArtifactReference, ...]:
    found: list[ArtifactReference] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            artifact_id = item.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id.startswith("sha256-"):
                artifact_type = item.get("artifact_type")
                found.append(
                    ArtifactReference(
                        artifact_id,
                        artifact_type if isinstance(artifact_type, str) else None,
                    )
                )
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(found)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact member: {path}") from exc


def _project_layer_metrics(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigDecodeError("block.layers", "expected an object")
    frozen = value["frozen_state"]
    plan = value["plan"]
    if not isinstance(frozen, dict) or not isinstance(plan, dict):
        raise ConfigDecodeError("block.layers", "expected nested metric objects")
    source = plan["source_weight"]
    if not isinstance(source, dict):
        raise ConfigDecodeError("block.layers.plan.source_weight", "expected an object")
    spec = source["spec"]
    if not isinstance(spec, dict):
        raise ConfigDecodeError("block.layers.plan.source_weight.spec", "expected an object")
    return {
        "actual_bit_cost": value["actual_bit_cost"],
        "frozen_state": {"rank": frozen["rank"]},
        "layer": value["layer"],
        "plan": {"source_weight": {"spec": {"shape": spec["shape"]}}},
    }


def _project_group_metrics(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigDecodeError("block.shared_input_groups", "expected an object")
    frozen = value["frozen_state"]
    plan = value["plan"]
    if not isinstance(frozen, dict) or not isinstance(plan, dict):
        raise ConfigDecodeError("block.shared_input_groups", "expected nested metric objects")
    members = plan["members"]
    if not isinstance(members, list) or not members:
        raise ConfigDecodeError("block.shared_input_groups.plan.members", "expected a non-empty list")
    try:
        input_widths = {member["in_features"] for member in members}
        output_width = sum(member["out_features"] for member in members)
    except (KeyError, TypeError) as exc:
        raise ConfigDecodeError(
            "block.shared_input_groups.plan.members",
            "expected member feature dimensions",
        ) from exc
    if len(input_widths) != 1:
        raise ConfigDecodeError(
            "block.shared_input_groups.plan.members",
            "members must share one input width",
        )
    return {
        "actual_bit_cost": value["actual_bit_cost"],
        "frozen_state": {"rank": frozen["rank"]},
        "block": value["block"],
        "name": value["name"],
        "plan": {
            "in_features": input_widths.pop(),
            "out_features": output_width,
        },
    }


def _decode_block_metrics(payload: dict[str, Any], artifact_id: str) -> _BlockMetrics:
    try:
        losses = payload["losses"]
        if not isinstance(losses, dict):
            raise ConfigDecodeError("block.losses", "expected an object")
        projected = {
            "identity": payload["identity"],
            "block": payload["block"],
            "layers": [_project_layer_metrics(value) for value in payload["layers"]],
            "shared_input_groups": [_project_group_metrics(value) for value in payload.get("shared_input_groups", ())],
            "losses": {"final_frozen_pre_kd": losses["final_frozen_pre_kd"]},
            "wall_seconds": payload["wall_seconds"],
            "peak_gpu_bytes": payload["peak_gpu_bytes"],
            "peak_host_bytes": payload["peak_host_bytes"],
        }
        if payload.get("activation_generation") is not None:
            projected["activation_generation"] = payload["activation_generation"]
        return from_dict(_BlockMetrics, projected, path="block")
    except (ConfigDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"committed block metrics are malformed: {artifact_id}") from exc


def _decode_layer_position(payload: dict[str, Any], artifact_id: str) -> _LayerCommitPosition:
    try:
        result = payload["result"]
        if not isinstance(result, dict):
            raise ConfigDecodeError("layer.result", "expected an object")
        return from_dict(
            _LayerCommitPosition,
            {"identity": payload["identity"], "layer": result["layer"]},
            path="layer",
        )
    except (ConfigDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"committed layer identity is malformed: {artifact_id}") from exc


def _journal_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"resident journal is unavailable: {path}") from exc
    records: list[dict[str, Any]] = []
    for sequence, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid journal JSON at sequence {sequence}") from exc
        if not isinstance(value, dict) or value.get("sequence") != sequence:
            raise ValueError(f"journal sequence is not contiguous at record {sequence}")
        if value.get("kind") not in {"layer", "group", "block"}:
            raise ValueError(f"unsupported journal record kind at sequence {sequence}")
        records.append(value)
    if not records:
        raise ValueError("resident journal contains no committed records")
    return records


def _commit_payload(
    record: dict[str, Any],
    store: LocalArtifactStore,
) -> tuple[dict[str, Any], ArtifactReference | None, _BlockMetrics | None]:
    artifact_id = str(record["artifact_id"])
    kind = str(record["kind"])
    expected_type = ArtifactTypes.SHARED_INPUT_GROUP_RESULT if kind == "group" else f"{kind}-result"
    descriptor = store.validate(artifact_id)
    if descriptor.artifact_type != expected_type:
        raise ValueError(f"journal {kind} artifact has type {descriptor.artifact_type!r}: {artifact_id}")
    filename = "shared-input-group-result.json" if kind == "group" else f"{kind}-result.json"
    payload = _read_json(store.path_for(artifact_id) / filename)
    if not isinstance(payload, dict):
        raise ValueError(f"committed {kind} payload is not an object: {artifact_id}")
    if kind == "layer":
        position = _decode_layer_position(payload, artifact_id)
        if position.identity != from_dict(CommitIdentity, record["identity"], path="journal.identity"):
            raise ValueError(f"journal identity does not match committed layer: {artifact_id}")
        if position.layer.block.index != record.get("block") or position.layer.path != record.get("layer"):
            raise ValueError(f"journal position does not match committed layer: {artifact_id}")
        return payload, None, None
    if kind == "group":
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"committed shared-input group is malformed: {artifact_id}")
        identity = from_dict(CommitIdentity, payload["identity"], path="group.identity")
        if identity != from_dict(CommitIdentity, record["identity"], path="journal.identity"):
            raise ValueError(f"journal identity does not match committed group: {artifact_id}")
        if result.get("block", {}).get("index") != record.get("block") or result.get("name") != record.get("layer"):
            raise ValueError(f"journal position does not match committed group: {artifact_id}")
        return payload, None, None
    metrics = _decode_block_metrics(payload, artifact_id)
    if metrics.identity != from_dict(CommitIdentity, record["identity"], path="journal.identity"):
        raise ValueError(f"journal identity does not match committed block: {artifact_id}")
    if metrics.block.index != record.get("block") or record.get("layer") is not None:
        raise ValueError(f"journal position does not match committed block: {artifact_id}")
    activation_reference = None
    if metrics.activation_generation is not None:
        activation_reference = ArtifactReference(
            metrics.activation_generation.artifact_id,
            metrics.activation_generation.artifact_type,
        )
    return payload, activation_reference, metrics


def _committed_metrics(
    commit_payloads: list[tuple[dict[str, Any], dict[str, Any], _BlockMetrics | None]],
) -> tuple[int, int, int, dict[str, int], float, float, int, int, tuple[float, ...]]:
    ranks = 0
    parameters = 0
    bit_costs: Counter[str] = Counter()
    wall_seconds = 0.0
    peak_gpu_bytes = 0
    peak_host_bytes = 0
    losses: list[float] = []
    seen_layers: set[tuple[int, str]] = set()
    for record, _payload, metrics in commit_payloads:
        if metrics is None:
            continue
        block = int(record["block"])
        for layer_metrics in metrics.layers:
            layer_block = layer_metrics.layer.block.index
            layer = layer_metrics.layer.path
            key = (layer_block, layer)
            if layer_block != block or key in seen_layers:
                raise ValueError(f"committed layer metrics are invalid: {record['artifact_id']}")
            seen_layers.add(key)
            ranks += layer_metrics.frozen_state.rank
            parameters += math.prod(layer_metrics.plan.source_weight.spec.shape)
            bit_costs.update(layer_metrics.actual_bit_cost)
        for group_metrics in metrics.shared_input_groups:
            key = (group_metrics.block.index, group_metrics.name)
            if group_metrics.block.index != block or key in seen_layers:
                raise ValueError(f"committed group metrics are invalid: {record['artifact_id']}")
            seen_layers.add(key)
            ranks += group_metrics.frozen_state.rank
            parameters += group_metrics.plan.in_features * group_metrics.plan.out_features
            bit_costs.update(group_metrics.actual_bit_cost)
        losses.append(metrics.losses.final_frozen_pre_kd)
        wall_seconds += metrics.wall_seconds
        peak_gpu_bytes = max(peak_gpu_bytes, metrics.peak_gpu_bytes)
        peak_host_bytes = max(peak_host_bytes, metrics.peak_host_bytes)
    total_bits = sum(bit_costs.values())
    return (
        len(seen_layers),
        ranks,
        parameters,
        dict(sorted(bit_costs.items())),
        total_bits / parameters if parameters else 0.0,
        wall_seconds,
        peak_gpu_bytes,
        peak_host_bytes,
        tuple(losses),
    )


def validate_resident_run(
    run_output: str | Path,
    *,
    expected_blocks: int = 26,
    require_complete: bool = False,
) -> RunValidationResult:
    if expected_blocks <= 0:
        raise ValueError("expected block count must be positive")
    root = Path(run_output)
    artifact_root = root / "artifacts"
    if not artifact_root.is_dir():
        raise ValueError(f"resident artifact store is unavailable: {artifact_root}")
    store = LocalArtifactStore(
        artifact_root,
        temporary_root=artifact_root,
        use_persistent_validation_cache=False,
    )
    all_records = _journal_records(root / "state" / "journal.jsonl")
    identities = {json.dumps(record.get("identity"), sort_keys=True) for record in all_records}
    identity_value = all_records[-1].get("identity")
    if not isinstance(identity_value, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in identity_value.items()
    ):
        raise ValueError("resident journal identity is invalid")
    active_identity = json.dumps(identity_value, sort_keys=True)
    records = [
        record for record in all_records if json.dumps(record.get("identity"), sort_keys=True) == active_identity
    ]

    commit_payloads: list[tuple[dict[str, Any], dict[str, Any], _BlockMetrics | None]] = []
    activation_by_block: dict[int, ArtifactReference] = {}
    for record in records:
        payload, activation, metrics = _commit_payload(record, store)
        commit_payloads.append((record, payload, metrics))
        if activation is not None:
            activation_by_block[int(record["block"])] = activation

    block_records = [record for record in records if record["kind"] == "block"]
    completed_blocks = tuple(sorted(int(record["block"]) for record in block_records))
    if len(set(completed_blocks)) != len(completed_blocks):
        raise ValueError("resident journal contains duplicate block commits")
    if completed_blocks and completed_blocks != tuple(range(completed_blocks[-1] + 1)):
        raise ValueError("resident block commits are not a contiguous zero-based prefix")
    complete = completed_blocks == tuple(range(expected_blocks))
    if require_complete and not complete:
        raise ValueError(f"resident run is incomplete: {len(completed_blocks)}/{expected_blocks} blocks committed")

    latest_activation = activation_by_block.get(completed_blocks[-1]) if completed_blocks else None
    pending: deque[ArtifactReference] = deque(
        ArtifactReference(
            str(record["artifact_id"]),
            ArtifactTypes.SHARED_INPUT_GROUP_RESULT if record["kind"] == "group" else f"{record['kind']}-result",
        )
        for record in records
    )
    plan_hash = identity_value.get("plan_hash")
    if isinstance(plan_hash, str) and plan_hash.startswith("sha256-") and len(plan_hash) == 71:
        pending.append(ArtifactReference(plan_hash, ArtifactTypes.QUANTIZATION_PLAN))
    for _record, payload, _metrics in commit_payloads:
        pending.extend(_references(payload))
    validated: dict[str, tuple[str, int]] = {}
    expected_types: dict[str, str] = {}
    retired: set[str] = set()
    while pending:
        reference = pending.popleft()
        if reference.artifact_type is not None:
            previous = expected_types.setdefault(reference.artifact_id, reference.artifact_type)
            if previous != reference.artifact_type:
                raise ValueError(f"conflicting artifact types for {reference.artifact_id}")
        if reference.artifact_id in validated or reference.artifact_id in retired:
            continue
        path = store.path_for(reference.artifact_id)
        if not path.exists():
            if reference.artifact_type == ArtifactTypes.ACTIVATION_GENERATION and (
                latest_activation is None or reference.artifact_id != latest_activation.artifact_id
            ):
                retired.add(reference.artifact_id)
                continue
            raise ArtifactCorruptionError(f"ART001 referenced artifact is unavailable: {reference.artifact_id}")
        descriptor = store.validate(reference.artifact_id)
        expected_type = expected_types.get(reference.artifact_id)
        if expected_type is not None and descriptor.artifact_type != expected_type:
            raise ValueError(
                f"artifact type mismatch for {reference.artifact_id}: "
                f"expected {expected_type}, got {descriptor.artifact_type}"
            )
        artifact_bytes = sum(item.bytes for item in descriptor.files)
        validated[reference.artifact_id] = (descriptor.artifact_type, artifact_bytes)
        artifact_root = store.path_for(reference.artifact_id)
        for member in descriptor.files:
            if member.path.endswith(".json"):
                pending.extend(_references(_read_json(artifact_root / member.path)))

    by_type = Counter(value[0] for value in validated.values())
    (
        committed_layer_count,
        rank_sum,
        quantized_parameters,
        bit_cost_by_category,
        effective_bpw,
        block_wall_seconds,
        peak_gpu_bytes,
        peak_host_bytes,
        final_frozen_pre_kd_losses,
    ) = _committed_metrics(commit_payloads)
    return RunValidationResult(
        str(root.resolve()),
        {str(key): str(value) for key, value in identity_value.items()},
        len(all_records),
        len(records),
        len(all_records) - len(records),
        len(identities),
        sum(record["kind"] in {"layer", "group"} for record in records),
        len(block_records),
        completed_blocks,
        complete,
        len(validated),
        sum(value[1] for value in validated.values()),
        dict(sorted(by_type.items())),
        tuple(sorted(retired)),
        committed_layer_count,
        rank_sum,
        quantized_parameters,
        bit_cost_by_category,
        effective_bpw,
        block_wall_seconds,
        peak_gpu_bytes,
        peak_host_bytes,
        final_frozen_pre_kd_losses,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_resident_run(
        args.run_output,
        expected_blocks=args.expected_blocks,
        require_complete=args.require_complete,
    )
    rendered = json.dumps(asdict(result), sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
