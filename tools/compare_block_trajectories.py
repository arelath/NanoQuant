"""Compare an authoritative resident block-loss trajectory with legacy logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanoquant.domain.models import ArtifactTypes
from nanoquant.infrastructure.artifacts import LocalArtifactStore

_LEGACY_BLOCK_LOSS = re.compile(
    r"Post-block scale refit summary:.*?->\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
)


@dataclass(frozen=True, slots=True)
class RewriteTrajectory:
    identity: dict[str, str]
    losses: tuple[float, ...]
    layer_budgets: tuple[RewriteLayerBudget, ...]


@dataclass(frozen=True, slots=True)
class RewriteLayerBudget:
    block: int
    layer: str
    rank: int
    actual_bits: int
    source_parameters: int


@dataclass(frozen=True, slots=True)
class LegacyLayerBudget:
    rank: int
    binary_factor_bits: int


def _identity_key(value: object) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        raise ValueError("journal block record has no identity")
    try:
        return (str(value["config_hash"]), str(value["model_hash"]), str(value["plan_hash"]))
    except KeyError as exc:
        raise ValueError("journal block identity is incomplete") from exc


def load_rewrite_trajectory(run_output: str | Path) -> RewriteTrajectory:
    root = Path(run_output)
    artifacts = LocalArtifactStore(root / "artifacts", use_persistent_validation_cache=False)
    journal = root / "state" / "journal.jsonl"
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid journal JSON at line {line_number}") from exc
        if isinstance(value, dict) and "identity" in value:
            records.append(value)
    if not records:
        raise ValueError("resident journal contains no identity-bearing commits")
    active_key = _identity_key(records[-1].get("identity"))
    active_records = [
        record
        for record in records
        if record.get("kind") == "block" and _identity_key(record.get("identity")) == active_key
    ]
    if not active_records:
        raise ValueError("active journal identity contains no committed blocks")
    by_block: dict[int, dict[str, Any]] = {}
    for record in active_records:
        block = int(record["block"])
        if block in by_block:
            raise ValueError(f"active journal identity contains duplicate block {block}")
        by_block[block] = record
    expected = list(range(len(by_block)))
    if sorted(by_block) != expected:
        raise ValueError(f"active committed blocks are not a contiguous prefix: {sorted(by_block)}")
    losses = []
    layer_budgets: list[RewriteLayerBudget] = []
    seen_layers: set[tuple[int, str]] = set()
    for block in expected:
        artifact_id = str(by_block[block]["artifact_id"])
        descriptor = artifacts.validate(artifact_id)
        if descriptor.artifact_type != ArtifactTypes.BLOCK_RESULT:
            raise ValueError(
                f"journal block {block} references {descriptor.artifact_type}, "
                f"not {ArtifactTypes.BLOCK_RESULT}"
            )
        artifact_root = artifacts.path_for(artifact_id)
        payload: Any = json.loads((artifact_root / "block-result.json").read_text(encoding="utf-8"))
        try:
            payload_block = int(payload["block"]["index"])
            payload_identity = _identity_key(payload["identity"])
            loss = float(payload["losses"]["final_frozen_pre_kd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed block result for journal block {block}") from exc
        if payload_block != block or payload_identity != active_key or not math.isfinite(loss):
            raise ValueError(f"invalid block result for journal block {block}")
        layers = payload.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"block result contains no layer budgets for journal block {block}")
        for value in layers:
            try:
                layer_block = int(value["layer"]["block"]["index"])
                layer = value["layer"]["path"]
                rank = value["frozen_state"]["rank"]
                bit_cost = value["actual_bit_cost"]
                shape = value["plan"]["source_weight"]["spec"]["shape"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"malformed layer budget for journal block {block}") from exc
            if (
                not isinstance(layer, str)
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or not isinstance(bit_cost, dict)
                or not all(isinstance(item, int) and not isinstance(item, bool) for item in bit_cost.values())
                or not isinstance(shape, list)
                or not shape
                or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape)
            ):
                raise ValueError(f"malformed layer budget for journal block {block}")
            actual_bits = sum(bit_cost.values())
            source_parameters = math.prod(shape)
            key = (layer_block, layer)
            if (
                layer_block != block
                or not layer
                or rank <= 0
                or actual_bits < 0
                or key in seen_layers
                or not all(item >= 0 for item in bit_cost.values())
            ):
                raise ValueError(f"invalid layer budget for journal block {block}")
            seen_layers.add(key)
            layer_budgets.append(RewriteLayerBudget(block, layer, rank, actual_bits, source_parameters))
        losses.append(loss)
    return RewriteTrajectory(
        {"config_hash": active_key[0], "model_hash": active_key[1], "plan_hash": active_key[2]},
        tuple(losses),
        tuple(layer_budgets),
    )


def load_legacy_trajectory(path: str | Path) -> tuple[float, ...]:
    source = Path(path).read_text(encoding="utf-8")
    losses = tuple(float(match.group(1)) for match in _LEGACY_BLOCK_LOSS.finditer(source))
    if not losses:
        raise ValueError(f"legacy log contains no post-block scale-refit summaries: {path}")
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError(f"legacy trajectory contains a non-finite loss: {path}")
    return losses


def load_legacy_rank_csv(path: str | Path) -> dict[tuple[int, str], LegacyLayerBudget]:
    budgets: dict[tuple[int, str], LegacyLayerBudget] = {}
    try:
        with Path(path).open(encoding="utf-8", newline="") as source:
            for line_number, row in enumerate(csv.DictReader(source), start=2):
                try:
                    block = int(row["block"]) - 1
                    layer = row["layer"]
                    budget = LegacyLayerBudget(int(row["rank"]), int(row["binary_bits"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid legacy rank CSV row {line_number}: {path}") from exc
                key = (block, layer)
                if block < 0 or not layer or budget.rank <= 0 or budget.binary_factor_bits < 0 or key in budgets:
                    raise ValueError(f"invalid legacy rank CSV row {line_number}: {path}")
                budgets[key] = budget
    except OSError as exc:
        raise ValueError(f"legacy rank CSV is unavailable: {path}") from exc
    if not budgets:
        raise ValueError(f"legacy rank CSV contains no layer budgets: {path}")
    return budgets


def compare_rank_allocations(
    rewrite: RewriteTrajectory,
    baselines: tuple[tuple[str, Path, dict[tuple[int, str], LegacyLayerBudget]], ...],
) -> list[dict[str, object]]:
    names = [name for name, _path, _budgets in baselines]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("legacy rank baseline names must be non-empty and unique")
    rewrite_by_key = {(item.block, item.layer): item for item in rewrite.layer_budgets}
    results: list[dict[str, object]] = []
    for name, path, all_legacy in baselines:
        legacy = {key: value for key, value in all_legacy.items() if key[0] < len(rewrite.losses)}
        paired = sorted(rewrite_by_key.keys() & legacy.keys())
        mismatches = [
            {
                "block": key[0],
                "layer": key[1],
                "rewrite_rank": rewrite_by_key[key].rank,
                "legacy_rank": legacy[key].rank,
                "rank_delta": rewrite_by_key[key].rank - legacy[key].rank,
            }
            for key in paired
            if rewrite_by_key[key].rank != legacy[key].rank
        ]
        rewrite_parameters = sum(value.source_parameters for value in rewrite_by_key.values())
        paired_parameters = sum(rewrite_by_key[key].source_parameters for key in paired)
        rewrite_actual_bits = sum(value.actual_bits for value in rewrite_by_key.values())
        legacy_rank_dependent_bits = sum(legacy[key].binary_factor_bits for key in paired)
        results.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "rewrite_layer_count": len(rewrite_by_key),
                "legacy_prefix_layer_count": len(legacy),
                "paired_layer_count": len(paired),
                "rank_mismatch_count": len(mismatches),
                "rewrite_rank_sum": sum(value.rank for value in rewrite_by_key.values()),
                "legacy_rank_sum": sum(value.rank for value in legacy.values()),
                "rewrite_source_parameters": rewrite_parameters,
                "paired_source_parameters": paired_parameters,
                "rewrite_actual_bits": rewrite_actual_bits,
                "rewrite_effective_bpw": rewrite_actual_bits / rewrite_parameters,
                "legacy_rank_dependent_bits": legacy_rank_dependent_bits,
                "legacy_rank_dependent_bpw": (
                    None if not paired_parameters else legacy_rank_dependent_bits / paired_parameters
                ),
                "missing_in_legacy": [
                    {"block": key[0], "layer": key[1]} for key in sorted(rewrite_by_key.keys() - legacy.keys())
                ],
                "missing_in_rewrite": [
                    {"block": key[0], "layer": key[1]} for key in sorted(legacy.keys() - rewrite_by_key.keys())
                ],
                "rank_mismatches": mismatches,
            }
        )
    return results


def compare_trajectories(
    rewrite: RewriteTrajectory,
    baselines: tuple[tuple[str, Path, tuple[float, ...]], ...],
) -> dict[str, object]:
    if not baselines:
        raise ValueError("at least one legacy baseline is required")
    baseline_names = [name for name, _path, _losses in baselines]
    if any(not name for name in baseline_names) or len(set(baseline_names)) != len(baseline_names):
        raise ValueError("legacy baseline names must be non-empty and unique")
    rows = []
    for block, rewrite_loss in enumerate(rewrite.losses):
        comparisons: dict[str, object] = {}
        for name, _path, losses in baselines:
            if block >= len(losses):
                comparisons[name] = None
                continue
            baseline_loss = losses[block]
            comparisons[name] = {
                "loss": baseline_loss,
                "absolute_delta": rewrite_loss - baseline_loss,
                "percent_delta": None if baseline_loss == 0 else 100.0 * (rewrite_loss / baseline_loss - 1.0),
            }
        rows.append({"block": block, "rewrite_loss": rewrite_loss, "baselines": comparisons})
    baseline_payloads = []
    for name, path, losses in baselines:
        paired = min(len(rewrite.losses), len(losses))
        deltas = [100.0 * (rewrite.losses[index] / losses[index] - 1.0) for index in range(paired) if losses[index]]
        baseline_payloads.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "block_count": len(losses),
                "paired_block_count": paired,
                "rewrite_lower_count": sum(rewrite.losses[index] < losses[index] for index in range(paired)),
                "mean_absolute_percent_delta": (
                    None if not deltas else sum(abs(delta) for delta in deltas) / len(deltas)
                ),
                "maximum_absolute_percent_delta": None if not deltas else max(abs(delta) for delta in deltas),
            }
        )
    return {
        "schema_version": 1,
        "rewrite_identity": rewrite.identity,
        "rewrite_block_count": len(rewrite.losses),
        "baselines": baseline_payloads,
        "blocks": rows,
    }


def render_markdown(comparison: dict[str, object]) -> str:
    baselines = comparison["baselines"]
    blocks = comparison["blocks"]
    if not isinstance(baselines, list) or not isinstance(blocks, list):
        raise TypeError("malformed trajectory comparison")
    names = [str(value["name"]) for value in baselines if isinstance(value, dict)]
    columns = ["Block", "Rewrite"]
    for name in names:
        columns.extend((name, f"Delta vs {name}"))
    lines = [
        "# Block-loss trajectory comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" for _column in columns) + "|",
    ]
    for value in blocks:
        if not isinstance(value, dict):
            continue
        cells = [str(value["block"]), f"{float(value['rewrite_loss']):.6g}"]
        compared = value["baselines"]
        if not isinstance(compared, dict):
            raise TypeError("malformed block comparison")
        for name in names:
            baseline = compared.get(name)
            if not isinstance(baseline, dict):
                cells.extend(("n/a", "n/a"))
                continue
            delta = baseline["percent_delta"]
            cells.extend(
                (
                    f"{float(baseline['loss']):.6g}",
                    "n/a" if delta is None else f"{float(delta):+.2f}%",
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(("", "## Summary", ""))
    for value in baselines:
        if not isinstance(value, dict):
            continue
        mean_delta = value["mean_absolute_percent_delta"]
        rendered_delta = "n/a" if mean_delta is None else f"{float(mean_delta):.2f}%"
        lines.append(
            f"- `{value['name']}`: rewrite lower at {value['rewrite_lower_count']}/{value['paired_block_count']} "
            f"paired boundaries; mean absolute delta {rendered_delta}."
        )
    rank_baselines = comparison.get("rank_baselines", [])
    if isinstance(rank_baselines, list) and rank_baselines:
        lines.extend(("", "## Rank allocation", ""))
        for value in rank_baselines:
            if not isinstance(value, dict):
                continue
            legacy_bpw = value["legacy_rank_dependent_bpw"]
            legacy_bpw_text = "n/a" if legacy_bpw is None else f"{float(legacy_bpw):.6f}"
            lines.append(
                f"- `{value['name']}`: {value['paired_layer_count']} paired layers; "
                f"{value['rank_mismatch_count']} rank mismatches; rewrite/legacy rank sums "
                f"{value['rewrite_rank_sum']}/{value['legacy_rank_sum']}. Rewrite effective BPW "
                f"{float(value['rewrite_effective_bpw']):.6f}; legacy rank-dependent BPW {legacy_bpw_text}; "
                f"missing in legacy/rewrite "
                f"{len(value['missing_in_legacy'])}/{len(value['missing_in_rewrite'])}."
            )
    lines.append("")
    return "\n".join(lines)


def _baseline(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("baseline must use NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--baseline", type=_baseline, action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--rank-baseline", type=_baseline, action="append", metavar="NAME=CSV")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    rewrite = load_rewrite_trajectory(args.run_output)
    baselines = tuple((name, path, load_legacy_trajectory(path)) for name, path in args.baseline)
    comparison = compare_trajectories(rewrite, baselines)
    rank_baselines = tuple(
        (name, path, load_legacy_rank_csv(path)) for name, path in (args.rank_baseline or [])
    )
    if rank_baselines:
        comparison["rank_baselines"] = compare_rank_allocations(rewrite, rank_baselines)
    rendered = (
        json.dumps(comparison, sort_keys=True, indent=2)
        if args.format == "json"
        else render_markdown(comparison)
    )
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
