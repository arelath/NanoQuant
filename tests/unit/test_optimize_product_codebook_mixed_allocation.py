from __future__ import annotations

import json
from pathlib import Path

from tools.optimize_product_codebook_mixed_allocation import (
    AllocationGroup,
    AllocationOption,
    _load_probe_options,
    _pareto_allocate,
)


def _option(name: str, bits: int, error: float) -> AllocationOption:
    return AllocationOption(
        name=name,
        bits=bits,
        weighted_error_energy=error,
        weighted_target_energy=10.0,
        actual_bpw=float(bits),
    )


def test_pareto_allocator_uses_budget_for_the_best_error_reduction() -> None:
    groups = (
        AllocationGroup(
            "a",
            "gate",
            0,
            (_option("small", 2, 5.0), _option("large", 3, 1.0)),
        ),
        AllocationGroup(
            "b",
            "up",
            0,
            (_option("small", 2, 4.0), _option("large", 3, 2.0)),
        ),
    )

    bits, selected, frontier = _pareto_allocate(groups, 5)

    assert bits == 5
    assert selected.choices == ("large", "small")
    assert selected.weighted_error_energy == 5.0
    assert frontier[-1] == (5, 5.0)


def test_probe_loader_removes_rate_distortion_dominated_options(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.json"
    payload = {
        "results": {
            "free_words": {
                "total_bits": 12,
                "actual_bpw": 1.2,
                "metrics": {
                    "weighted_error_energy": 2.0,
                    "weighted_target_energy": 10.0,
                },
            },
            "cheap": {
                "total_bits": 8,
                "actual_bpw": 0.8,
                "metrics": {
                    "weighted_error_energy": 4.0,
                    "weighted_target_energy": 10.0,
                },
            },
            "dominated": {
                "total_bits": 10,
                "actual_bpw": 1.0,
                "metrics": {
                    "weighted_error_energy": 5.0,
                    "weighted_target_energy": 10.0,
                },
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    options = _load_probe_options(path)

    assert [item.name for item in options] == ["cheap", "free_words"]


def test_probe_loader_retains_dominated_free_control_for_comparison(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "free_words": {
                        "total_bits": 12,
                        "actual_bpw": 1.2,
                        "metrics": {
                            "weighted_error_energy": 2.0,
                            "weighted_target_energy": 10.0,
                        },
                    },
                    "better": {
                        "total_bits": 10,
                        "actual_bpw": 1.0,
                        "metrics": {
                            "weighted_error_energy": 1.0,
                            "weighted_target_energy": 10.0,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    options = _load_probe_options(path)

    assert [item.name for item in options] == ["better", "free_words"]
