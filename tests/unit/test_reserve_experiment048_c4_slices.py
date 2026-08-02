from __future__ import annotations

import pytest

from tools.reserve_experiment048_c4_slices import _entry, _with_reservations


def _registry() -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy": "Every used interval is permanently retired.",
        "slices": [
            _entry("old", 296, "sha256:old", "old-experiment"),
        ],
    }


def test_reservation_appends_two_disjoint_role_bound_slices() -> None:
    base = _registry()
    selection = _entry(
        "selection",
        344,
        "sha256:selection",
        "experiment-048-correction-selection",
    )
    confirmation = _entry(
        "confirmation",
        392,
        "sha256:confirmation",
        "experiment-048-final-confirmation",
    )

    reserved = _with_reservations(base, (selection, confirmation))

    assert len(reserved["slices"]) == 3
    assert len(base["slices"]) == 1
    assert reserved["slices"][-2:] == [selection, confirmation]


def test_reservation_rejects_overlap_and_existing_identity() -> None:
    base = _registry()
    selection = _entry(
        "selection",
        320,
        "sha256:selection",
        "experiment-048-correction-selection",
    )
    confirmation = _entry(
        "confirmation",
        392,
        "sha256:confirmation",
        "experiment-048-final-confirmation",
    )

    with pytest.raises(ValueError, match="overlap"):
        _with_reservations(base, (selection, confirmation))
    with pytest.raises(ValueError, match="already exists"):
        _with_reservations(
            base,
            (
                _entry(
                    "old",
                    344,
                    "sha256:selection",
                    "experiment-048-correction-selection",
                ),
                confirmation,
            ),
        )
