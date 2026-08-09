from __future__ import annotations

import json
from pathlib import Path

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlBudgetProfile,
    KlBudgetProvenance,
    KlSequenceResult,
)
from nanoquant.config.codec import to_dict
from tools.optimize_product_codebook_global_allocation import (
    _constrain_grouped_qkv,
    _global_kl_objective_calibration,
)
from tools.optimize_product_codebook_mixed_allocation import (
    AllocationGroup,
    AllocationOption,
)


def _option(name: str, bits: int, error: float) -> AllocationOption:
    return AllocationOption(name, bits, error, 10.0, float(bits))


def test_shared_qkv_anchor_is_conserved_across_projection_errors(tmp_path: Path) -> None:
    provenance = KlBudgetProvenance(
        "model", "revision", "recipe", "dataset", "slice", "run"
    )
    sequence = KlSequenceResult(3.0, 0.6, 5)
    profile = KlBudgetProfile(
        2,
        provenance,
        1.0,
        (
            KlBudgetArmResult(
                "unit:0:self_attn.attn_qkv",
                3.0,
                0.6,
                5,
                0.3,
                (sequence,),
            ),
        ),
        True,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(to_dict(profile)), encoding="utf-8")
    groups = (
        AllocationGroup(
            "block-00:q", "q", 0, (_option("cheap", 1, 2.0), _option("free_words", 2, 4.0))
        ),
        AllocationGroup(
            "block-00:k", "k", 0, (_option("cheap", 1, 1.0), _option("free_words", 2, 1.0))
        ),
        AllocationGroup(
            "block-00:v", "v", 0, (_option("cheap", 1, 3.0), _option("free_words", 2, 5.0))
        ),
    )

    anchors, multipliers, calibration = _global_kl_objective_calibration(
        groups,
        profile_path,
        model_source="model",
        model_revision="revision",
        expected_profile_key=profile.profile_key,
    )

    assert sum(anchors.values()) == 0.6
    assert anchors == {
        "block-00:q": 0.24,
        "block-00:k": 0.06,
        "block-00:v": 0.3,
    }
    assert len(set(multipliers.values())) == 1
    assert calibration["mode"] == "measured_unit_kl_response"


def test_grouped_qkv_constraint_keeps_only_free_controls() -> None:
    groups = tuple(
        AllocationGroup(
            f"block-00:{projection}",
            projection,
            0,
            (_option("product", 1, 1.0), _option("free_words", 2, 2.0)),
        )
        for projection in ("q", "k", "v", "o")
    )

    constrained = _constrain_grouped_qkv(groups)

    assert [tuple(option.name for option in group.options) for group in constrained] == [
        ("free_words",),
        ("free_words",),
        ("free_words",),
        ("product", "free_words"),
    ]
