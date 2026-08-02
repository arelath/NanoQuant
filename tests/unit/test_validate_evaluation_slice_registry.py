from __future__ import annotations

from copy import deepcopy

import pytest

from tools.validate_evaluation_slice_registry import validate_registry


def _registry() -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy": "Every opened interval is permanently retired.",
        "slices": [
            {
                "id": "wiki-old",
                "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
                "split": "validation",
                "offset": 0,
                "samples": 2,
                "sequence_length": 8,
                "token_start": 0,
                "token_end": 14,
                "token_hash": "sha256:old",
                "status": "retired",
                "consumer": "old-experiment",
            },
            {
                "id": "wiki-new",
                "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
                "split": "validation",
                "offset": 2,
                "samples": 2,
                "sequence_length": 8,
                "token_start": 14,
                "token_end": 28,
                "token_hash": "sha256:new",
                "status": "reserved",
                "consumer": "new-experiment",
            },
        ],
    }


def test_accepts_disjoint_reserved_and_retired_inventory() -> None:
    result = validate_registry(_registry())
    assert result == {
        "schema_version": 1,
        "slice_count": 2,
        "reserved_count": 1,
        "retired_count": 1,
        "dataset_split_count": 1,
    }


def test_rejects_overlap_with_permanently_retired_slice() -> None:
    payload = _registry()
    payload["slices"][1].update(  # type: ignore[index, union-attr]
        offset=1,
        token_start=7,
        token_end=21,
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_registry(payload)


def test_rejects_duplicate_identity_and_released_status() -> None:
    duplicate = _registry()
    duplicate["slices"][1]["id"] = "wiki-old"  # type: ignore[index]
    with pytest.raises(ValueError, match="lifecycle metadata"):
        validate_registry(duplicate)

    released = deepcopy(_registry())
    released["slices"][0]["status"] = "released"  # type: ignore[index]
    with pytest.raises(ValueError, match="lifecycle metadata"):
        validate_registry(released)


def test_rejects_interval_that_does_not_match_slice_dimensions() -> None:
    payload = _registry()
    payload["slices"][0]["token_end"] = 16  # type: ignore[index]
    with pytest.raises(ValueError, match="differs from its dimensions"):
        validate_registry(payload)
