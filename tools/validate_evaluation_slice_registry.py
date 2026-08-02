"""Validate the permanent evaluation-slice retirement ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    return parser


def _interval(entry: dict[str, Any]) -> tuple[int, int]:
    offset = int(entry["offset"])
    samples = int(entry["samples"])
    sequence_length = int(entry["sequence_length"])
    if offset < 0 or samples <= 0 or sequence_length < 2:
        raise ValueError(f"slice {entry.get('id')} has invalid dimensions")
    stride = (
        sequence_length - 1
        if entry.get("dataset") == "Salesforce/wikitext:wikitext-2-raw-v1"
        else sequence_length
    )
    expected = (offset * stride, (offset + samples) * stride)
    observed = (int(entry["token_start"]), int(entry["token_end"]))
    if observed != expected:
        raise ValueError(
            f"slice {entry.get('id')} token interval differs from its dimensions"
        )
    return observed


def validate_registry(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("evaluation slice registry schema is invalid")
    entries = payload.get("slices")
    if not isinstance(entries, list) or not entries:
        raise ValueError("evaluation slice registry must contain slices")
    if not isinstance(payload.get("policy"), str) or "permanently retired" not in str(
        payload["policy"]
    ):
        raise ValueError("evaluation slice registry lacks a permanent-retirement policy")

    identities: set[str] = set()
    grouped: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    status_counts = {"reserved": 0, "retired": 0}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("evaluation slice entry is invalid")
        entry = raw_entry
        identity = entry.get("id")
        dataset = entry.get("dataset")
        split = entry.get("split")
        status = entry.get("status")
        if (
            not isinstance(identity, str)
            or not identity
            or identity in identities
            or not isinstance(dataset, str)
            or not dataset
            or not isinstance(split, str)
            or not split
            or status not in status_counts
            or not isinstance(entry.get("consumer"), str)
            or not entry["consumer"]
            or not isinstance(entry.get("token_hash"), str)
            or not entry["token_hash"]
        ):
            raise ValueError("evaluation slice identity or lifecycle metadata is invalid")
        identities.add(identity)
        status_counts[str(status)] += 1
        start, end = _interval(entry)
        grouped.setdefault((dataset, split), []).append((start, end, identity))

    for (dataset, split), intervals in grouped.items():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current[0] < previous[1]:
                raise ValueError(
                    f"evaluation slices {previous[2]} and {current[2]} overlap "
                    f"in {dataset} {split}"
                )
    return {
        "schema_version": 1,
        "slice_count": len(entries),
        "reserved_count": status_counts["reserved"],
        "retired_count": status_counts["retired"],
        "dataset_split_count": len(grouped),
    }


def run(args: argparse.Namespace) -> int:
    result = validate_registry(json.loads(args.registry.read_text(encoding="utf-8")))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
