"""Compare an authoritative resident block-loss trajectory with legacy logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LEGACY_BLOCK_LOSS = re.compile(
    r"Post-block scale refit summary:.*?->\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
)


@dataclass(frozen=True, slots=True)
class RewriteTrajectory:
    identity: dict[str, str]
    losses: tuple[float, ...]


def _identity_key(value: object) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        raise ValueError("journal block record has no identity")
    try:
        return (str(value["config_hash"]), str(value["model_hash"]), str(value["plan_hash"]))
    except KeyError as exc:
        raise ValueError("journal block identity is incomplete") from exc


def load_rewrite_trajectory(run_output: str | Path) -> RewriteTrajectory:
    root = Path(run_output)
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
    for block in expected:
        artifact_id = str(by_block[block]["artifact_id"])
        if not artifact_id.startswith("sha256-") or len(artifact_id) != 71:
            raise ValueError(f"invalid block artifact id for journal block {block}")
        artifact_root = root / "artifacts" / artifact_id[7:9] / artifact_id
        payload: Any = json.loads((artifact_root / "block-result.json").read_text(encoding="utf-8"))
        try:
            payload_block = int(payload["block"]["index"])
            loss = float(payload["losses"]["final_frozen_pre_kd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed block result for journal block {block}") from exc
        if payload_block != block or not math.isfinite(loss):
            raise ValueError(f"invalid block result for journal block {block}")
        losses.append(loss)
    return RewriteTrajectory(
        {"config_hash": active_key[0], "model_hash": active_key[1], "plan_hash": active_key[2]},
        tuple(losses),
    )


def load_legacy_trajectory(path: str | Path) -> tuple[float, ...]:
    source = Path(path).read_text(encoding="utf-8")
    losses = tuple(float(match.group(1)) for match in _LEGACY_BLOCK_LOSS.finditer(source))
    if not losses:
        raise ValueError(f"legacy log contains no post-block scale-refit summaries: {path}")
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError(f"legacy trajectory contains a non-finite loss: {path}")
    return losses


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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    rewrite = load_rewrite_trajectory(args.run_output)
    baselines = tuple((name, path, load_legacy_trajectory(path)) for name, path in args.baseline)
    comparison = compare_trajectories(rewrite, baselines)
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
