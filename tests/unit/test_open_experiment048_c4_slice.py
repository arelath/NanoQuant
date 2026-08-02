from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoquant.config.codec import semantic_hash
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from tools.open_experiment048_c4_slice import open_slice


def _entry(identity: str, offset: int, consumer: str) -> dict[str, object]:
    return {
        "id": identity,
        "dataset": "allenai/c4",
        "split": "validation",
        "offset": offset,
        "samples": 48,
        "sequence_length": 512,
        "token_start": offset * 512,
        "token_end": (offset + 48) * 512,
        "token_hash": "sha256:" + str(offset).zfill(64),
        "status": "reserved",
        "consumer": consumer,
    }


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    registry = tmp_path / "registry.json"
    snapshot = {
        "schema_version": 1,
        "policy": "Every opened interval is permanently retired.",
        "slices": [
            _entry("selection", 344, "experiment-048-correction-selection"),
            _entry("confirmation", 392, "experiment-048-final-confirmation"),
        ],
    }
    atomic_write_json(registry, snapshot)
    protocol = {
        "experiment": 48,
        "slices": {
            "selection": snapshot["slices"][0],
            "confirmation": snapshot["slices"][1],
            "registry_snapshot": snapshot,
            "registry_snapshot_hash": semantic_hash(snapshot),
        },
        "bound_files": {
            "slice_registry": {
                "path": str(registry.resolve()),
                "sha256": "sha256:" + hash_file(registry),
            }
        },
    }
    receipt = tmp_path / "campaign.json"
    atomic_write_json(
        receipt,
        {
            "schema_version": 1,
            "status": "ready_for_selection_evaluation",
            "protocol_hash": semantic_hash(protocol),
            "protocol": protocol,
        },
    )
    return receipt, registry


def test_open_slice_retires_before_use_and_is_idempotent(tmp_path: Path) -> None:
    receipt, registry = _campaign(tmp_path)

    first = open_slice(receipt, registry, "selection")
    repeated = open_slice(receipt, registry, "selection")
    confirmation = open_slice(receipt, registry, "confirmation")

    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert [entry["status"] for entry in payload["slices"]] == ["retired", "retired"]
    assert payload["slices"][0]["retirement"]["role"] == "selection"
    assert payload["slices"][1]["retirement"]["role"] == "confirmation"
    assert repeated == first
    assert confirmation["status"] == "opened-and-permanently-retired"


def test_open_slice_rejects_unrelated_registry_change(tmp_path: Path) -> None:
    receipt, registry = _campaign(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["policy"] += " Tampered."
    atomic_write_json(registry, payload)

    with pytest.raises(ValueError, match="outside receipt-authorized"):
        open_slice(receipt, registry, "selection")


def test_open_slice_rejects_tampered_campaign_protocol(tmp_path: Path) -> None:
    receipt, registry = _campaign(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["protocol"]["experiment"] = 49
    atomic_write_json(receipt, payload)

    with pytest.raises(ValueError, match="receipt identity"):
        open_slice(receipt, registry, "selection")
