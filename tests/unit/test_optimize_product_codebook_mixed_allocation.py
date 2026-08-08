from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlBudgetProfile,
    KlBudgetProvenance,
    KlSequenceResult,
)
from nanoquant.config.codec import to_dict
from tools.optimize_product_codebook_mixed_allocation import (
    AllocationGroup,
    AllocationOption,
    _kl_objective_calibration,
    _limit_group_free_rows,
    _limit_option_regression,
    _load_probe_options,
    _pareto_allocate,
    _parse_group_free_row_floors,
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
    assert selected.objective_value == 5.0
    assert frontier[-1] == (5, 5.0)


def test_pareto_allocator_can_prioritize_measured_kl_effect() -> None:
    groups = (
        AllocationGroup(
            "a",
            "gate",
            0,
            (_option("small", 2, 10.0), _option("large", 3, 5.0)),
        ),
        AllocationGroup(
            "b",
            "up",
            0,
            (_option("small", 2, 10.0), _option("large", 3, 2.0)),
        ),
    )

    _bits, raw, _frontier = _pareto_allocate(groups, 5)
    _bits, calibrated, _frontier = _pareto_allocate(
        groups,
        5,
        objective_multipliers={"a": 10.0, "b": 1.0},
    )

    assert raw.choices == ("small", "large")
    assert calibrated.choices == ("large", "small")


def test_kl_calibration_uses_exact_unit_anchor_over_free_error(
    tmp_path: Path,
) -> None:
    provenance = KlBudgetProvenance(
        "model",
        "revision",
        "recipe",
        "dataset",
        "slice",
        "run",
    )
    sequence = KlSequenceResult(2.0, 0.5, 10)
    profile = KlBudgetProfile(
        2,
        provenance,
        1.0,
        (
            KlBudgetArmResult(
                "unit:0:mlp.gate_proj",
                2.0,
                0.5,
                10,
                0.25,
                (sequence,),
            ),
        ),
        True,
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(to_dict(profile)), encoding="utf-8")
    groups = (
        AllocationGroup(
            "block-00:gate",
            "gate",
            0,
            (_option("candidate", 2, 5.0), _option("free_words", 3, 10.0)),
        ),
    )

    anchors, multipliers, calibration = _kl_objective_calibration(
        groups,
        path,
        model_source="model",
        model_revision="revision",
        expected_profile_key=profile.profile_key,
    )

    assert anchors == {"block-00:gate": 0.5}
    assert multipliers == {"block-00:gate": 0.05}
    assert calibration["profile_key"] == profile.profile_key
    with pytest.raises(ValueError, match="profile key"):
        _kl_objective_calibration(
            groups,
            path,
            model_source="model",
            model_revision="revision",
            expected_profile_key="sha256:stale",
        )


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


def test_option_regression_limit_retains_control_and_bounded_candidates() -> None:
    groups = (
        AllocationGroup(
            "a",
            "gate",
            0,
            (
                _option("cheap_regression", 1, 10.2),
                _option("bounded", 2, 10.1),
                _option("free_words", 3, 10.0),
            ),
        ),
    )

    limited = _limit_option_regression(groups, 0.01)

    assert [item.name for item in limited[0].options] == [
        "bounded",
        "free_words",
    ]


def test_option_regression_limit_is_optional() -> None:
    groups = (
        AllocationGroup(
            "a",
            "down",
            0,
            (_option("cheap", 1, 12.0), _option("free_words", 3, 10.0)),
        ),
    )

    assert _limit_option_regression(groups, None) is groups


def test_group_free_row_floor_filters_only_named_group() -> None:
    groups = (
        AllocationGroup(
            "block-12:gate",
            "gate",
            12,
            (
                _option("right_product_codebook_k16_free640_outliers2", 1, 2.0),
                _option("right_product_codebook_k16_free672_outliers2", 2, 1.5),
                _option("free_words", 3, 1.0),
            ),
        ),
        AllocationGroup(
            "block-12:up",
            "up",
            12,
            (_option("cheap", 1, 2.0), _option("free_words", 3, 1.0)),
        ),
    )

    floors = _parse_group_free_row_floors("block-12:gate=672")
    limited = _limit_group_free_rows(groups, floors)

    assert [option.name for option in limited[0].options] == [
        "right_product_codebook_k16_free672_outliers2",
        "free_words",
    ]
    assert limited[1] == groups[1]
