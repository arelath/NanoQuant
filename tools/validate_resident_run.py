"""Validate the immutable artifact graph and progress journal for a resident run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    artifact_type: str | None


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
        if value.get("kind") not in {"layer", "block"}:
            raise ValueError(f"unsupported journal record kind at sequence {sequence}")
        records.append(value)
    if not records:
        raise ValueError("resident journal contains no committed records")
    return records


def _commit_payload(
    record: dict[str, Any],
    store: LocalArtifactStore,
) -> tuple[object, ArtifactReference | None]:
    artifact_id = str(record["artifact_id"])
    kind = str(record["kind"])
    expected_type = f"{kind}-result"
    descriptor = store.validate(artifact_id)
    if descriptor.artifact_type != expected_type:
        raise ValueError(
            f"journal {kind} artifact has type {descriptor.artifact_type!r}: {artifact_id}"
        )
    filename = f"{kind}-result.json"
    payload = _read_json(store.path_for(artifact_id) / filename)
    if not isinstance(payload, dict):
        raise ValueError(f"committed {kind} payload is not an object: {artifact_id}")
    if payload.get("identity") != record.get("identity"):
        raise ValueError(f"journal identity does not match committed {kind}: {artifact_id}")
    result = payload.get("result") if kind == "layer" else payload
    if not isinstance(result, dict):
        raise ValueError(f"committed {kind} result is not an object: {artifact_id}")
    if kind == "layer":
        layer = result.get("layer")
        if not isinstance(layer, dict):
            raise ValueError(f"committed layer identity is missing: {artifact_id}")
        block_value = layer.get("block")
        block = block_value.get("index") if isinstance(block_value, dict) else None
        if block != record.get("block") or layer.get("path") != record.get("layer"):
            raise ValueError(f"journal position does not match committed layer: {artifact_id}")
        return payload, None
    block_value = payload.get("block")
    block = block_value.get("index") if isinstance(block_value, dict) else None
    if block != record.get("block") or record.get("layer") is not None:
        raise ValueError(f"journal position does not match committed block: {artifact_id}")
    activation = payload.get("activation_generation")
    activation_reference = None
    if isinstance(activation, dict) and isinstance(activation.get("artifact_id"), str):
        activation_reference = ArtifactReference(
            activation["artifact_id"],
            activation.get("artifact_type") if isinstance(activation.get("artifact_type"), str) else None,
        )
    return payload, activation_reference


def _committed_metrics(
    commit_payloads: list[tuple[dict[str, Any], object]],
) -> tuple[int, int, int, dict[str, int], float, float, int, int, tuple[float, ...]]:
    ranks = 0
    parameters = 0
    bit_costs: Counter[str] = Counter()
    wall_seconds = 0.0
    peak_gpu_bytes = 0
    peak_host_bytes = 0
    losses: list[float] = []
    seen_layers: set[tuple[int, str]] = set()
    for record, payload_value in commit_payloads:
        if record["kind"] != "block":
            continue
        if not isinstance(payload_value, dict):
            raise ValueError(f"committed block payload is not an object: {record['artifact_id']}")
        block = int(record["block"])
        try:
            layers = payload_value["layers"]
            loss = float(payload_value["losses"]["final_frozen_pre_kd"])
            block_wall = float(payload_value["wall_seconds"])
            block_peak_gpu = int(payload_value["peak_gpu_bytes"])
            block_peak_host = int(payload_value["peak_host_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"committed block metrics are malformed: {record['artifact_id']}") from exc
        if (
            not isinstance(layers, list)
            or not layers
            or not math.isfinite(loss)
            or not math.isfinite(block_wall)
            or block_wall < 0
            or block_peak_gpu < 0
            or block_peak_host < 0
        ):
            raise ValueError(f"committed block metrics are invalid: {record['artifact_id']}")
        for layer_value in layers:
            try:
                layer_block = int(layer_value["layer"]["block"]["index"])
                layer = layer_value["layer"]["path"]
                rank = layer_value["frozen_state"]["rank"]
                shape = layer_value["plan"]["source_weight"]["spec"]["shape"]
                layer_bit_cost = layer_value["actual_bit_cost"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"committed layer metrics are malformed: {record['artifact_id']}") from exc
            if (
                layer_block != block
                or not isinstance(layer, str)
                or not layer
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank <= 0
                or not isinstance(shape, list)
                or not shape
                or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape)
                or not isinstance(layer_bit_cost, dict)
                or not layer_bit_cost
                or not all(
                    isinstance(name, str)
                    and name
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for name, value in layer_bit_cost.items()
                )
                or (layer_block, layer) in seen_layers
            ):
                raise ValueError(f"committed layer metrics are invalid: {record['artifact_id']}")
            seen_layers.add((layer_block, layer))
            ranks += rank
            parameters += math.prod(shape)
            bit_costs.update(layer_bit_cost)
        losses.append(loss)
        wall_seconds += block_wall
        peak_gpu_bytes = max(peak_gpu_bytes, block_peak_gpu)
        peak_host_bytes = max(peak_host_bytes, block_peak_host)
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
        record
        for record in all_records
        if json.dumps(record.get("identity"), sort_keys=True) == active_identity
    ]

    commit_payloads: list[tuple[dict[str, Any], object]] = []
    activation_by_block: dict[int, ArtifactReference] = {}
    for record in records:
        payload, activation = _commit_payload(record, store)
        commit_payloads.append((record, payload))
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
        raise ValueError(
            f"resident run is incomplete: {len(completed_blocks)}/{expected_blocks} blocks committed"
        )

    latest_activation = activation_by_block.get(completed_blocks[-1]) if completed_blocks else None
    pending: deque[ArtifactReference] = deque(
        ArtifactReference(str(record["artifact_id"]), f"{record['kind']}-result")
        for record in records
    )
    plan_hash = identity_value.get("plan_hash")
    if isinstance(plan_hash, str) and plan_hash.startswith("sha256-") and len(plan_hash) == 71:
        pending.append(ArtifactReference(plan_hash, "quantization-plan"))
    for _record, payload in commit_payloads:
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
            if (
                reference.artifact_type == "activation-generation"
                and (latest_activation is None or reference.artifact_id != latest_activation.artifact_id)
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
        sum(record["kind"] == "layer" for record in records),
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
