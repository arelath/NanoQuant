from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from run_error_budget_campaign import (
    Campaign,
    _improvement_gate,
    _material_improvement_passed,
    _profile_key,
    _rank_inventory_for_run,
    _rank_redistribution,
    _rank_redistribution_gate,
    _reported_arm_kls,
    _require_at_or_below_budget,
    _require_candidate_sidecars,
    _select_d2_granularity,
    _select_winning_alpha,
    _select_winning_patch,
)

from nanoquant.application.kl_budget import KL_BUDGET_EVALUATOR_VERSION
from nanoquant.config.schema import KlSensitivityGranularity


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_campaign_completion_markers_fail_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate-summary.json"
    _write(candidate, {"status": "interrupted"})
    assert not Campaign._marker_complete(candidate)
    _write(candidate, {"status": "completed"})
    assert Campaign._marker_complete(candidate)

    profile = tmp_path / "artifact.json"
    _write(profile, {"complete": False})
    assert not Campaign._marker_complete(profile)
    _write(profile, {"complete": True})
    assert not Campaign._marker_complete(profile)
    _write(
        profile,
        {"complete": True, "evaluator_version": KL_BUDGET_EVALUATOR_VERSION},
    )
    assert Campaign._marker_complete(profile)

    quality = tmp_path / "quality-comparison-tuned.json"
    _write(quality, {"status": "completed"})
    assert not Campaign._marker_complete(quality)
    _write(quality, {"candidate_evaluation": {}})
    assert Campaign._marker_complete(quality)

    cuda = tmp_path / "cuda.xml"
    cuda.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    assert Campaign._marker_complete(cuda)
    cuda.write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    assert not Campaign._marker_complete(cuda)


def test_campaign_profile_key_rejects_stale_evaluator(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    receipt = profile / "artifact.json"
    _write(
        receipt,
        {
            "complete": True,
            "evaluator_version": KL_BUDGET_EVALUATOR_VERSION - 1,
            "profile_key": "sha256:key",
        },
    )
    with pytest.raises(ValueError, match="evaluator identity is stale"):
        _profile_key(profile)

    _write(
        receipt,
        {
            "complete": True,
            "evaluator_version": KL_BUDGET_EVALUATOR_VERSION,
            "profile_key": "sha256:key",
        },
    )
    assert _profile_key(profile) == "sha256:key"


def test_campaign_budget_gate_rejects_any_material_overrun(tmp_path: Path) -> None:
    _require_at_or_below_budget(tmp_path / "equal", 1.025, 1.025)
    _require_at_or_below_budget(tmp_path / "roundoff", 1.025 + 1e-13, 1.025)

    with pytest.raises(ValueError, match="exceeds the Experiment 016 budget"):
        _require_at_or_below_budget(tmp_path / "over", 1.025 + 2e-12, 1.025)


def test_campaign_candidate_sidecars_must_match_configured_arm(tmp_path: Path) -> None:
    bias = {
        "factor_owners": 90,
        "bias_owner_count": 90,
        "actual_bias_bits": 100,
        "patch_owner_count": 0,
        "actual_patch_bits": 0,
    }
    _require_candidate_sidecars(tmp_path / "bias", bias, bias=True, patch_rank=0)

    patch = {
        **bias,
        "patch_owner_count": 12,
        "actual_patch_bits": 200,
    }
    _require_candidate_sidecars(tmp_path / "patch", patch, bias=True, patch_rank=8)

    with pytest.raises(ValueError, match="bias inventory"):
        _require_candidate_sidecars(
            tmp_path / "missing-bias",
            {**bias, "bias_owner_count": 89},
            bias=True,
            patch_rank=0,
        )
    with pytest.raises(ValueError, match="patch owners"):
        _require_candidate_sidecars(
            tmp_path / "bad-patch",
            {**bias, "patch_owner_count": 1},
            bias=True,
            patch_rank=8,
        )


def test_campaign_selects_lowest_qkv_kl_alpha() -> None:
    assert _select_winning_alpha({1: 0.8, 2: 0.6, 4: 0.7}) == 2


def test_campaign_selects_exact_d2_first_then_documented_fallback() -> None:
    exact = KlSensitivityGranularity.EXACT_OR_TYPE_BLOCK
    fallback = KlSensitivityGranularity.TYPE_BLOCK

    assert _select_d2_granularity({exact: {"passed": True}}) is exact
    assert (
        _select_d2_granularity(
            {exact: {"passed": False}, fallback: {"passed": True}}
        )
        is fallback
    )
    with pytest.raises(ValueError, match="neither exact-unit nor type-by-block"):
        _select_d2_granularity(
            {exact: {"passed": False}, fallback: {"passed": False}}
        )


def test_campaign_selects_smallest_patch_rank_that_improves_over_zero() -> None:
    assert _select_winning_patch({0: 0.8, 4: 0.79, 8: 0.5, 16: 0.4}) == 4
    assert _select_winning_patch({0: 0.8, 4: 0.81, 8: 0.79, 16: 0.7}) == 8
    assert _select_winning_patch({0: 0.8, 4: 0.81, 8: 0.82, 16: 0.83}) == 0
    assert (
        _select_winning_patch(
            {0: 0.8, 4: 0.79, 8: 0.7, 16: 0.6},
            eligible_ranks={0, 8, 16},
        )
        == 8
    )

    with pytest.raises(ValueError, match="eligible ranks"):
        _select_winning_patch(
            {0: 0.8, 4: 0.79, 8: 0.7, 16: 0.6},
            eligible_ranks={4},
        )


def test_campaign_improvement_gate_reports_absolute_and_relative_kl() -> None:
    gate = _improvement_gate(2.0, 1.5)

    assert gate == {
        "before": 2.0,
        "after": 1.5,
        "delta": -0.5,
        "relative_delta": -0.25,
        "improved": True,
    }
    assert _material_improvement_passed(gate)
    assert not _material_improvement_passed(_improvement_gate(2.0, 1.99))
    assert not _material_improvement_passed({**gate, "upper_relative_delta": -0.005})
    assert _material_improvement_passed({**gate, "upper_relative_delta": -0.02})

    with pytest.raises(ValueError, match="threshold"):
        _material_improvement_passed(gate, 0)


def test_campaign_reports_only_intentionally_measured_profile_arms(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    _write(
        profile / "artifact.json",
        {
            "complete": True,
            "evaluator_version": KL_BUDGET_EVALUATOR_VERSION,
            "profile_key": "sha256:key",
        },
    )
    _write(
        profile / "kl-budget-profile.json",
        {
            "arms": [
                {"arm": "type:self_attn.attn_qkv", "kl_nats_per_token": 0.25},
                {"arm": "unit:0:mlp.up_proj", "kl_nats_per_token": 0.5},
            ]
        },
    )

    assert _reported_arm_kls(profile) == {"profile_key": "sha256:key", "qkv": 0.25}


def test_campaign_profile_uses_requested_arms_and_pinned_local_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def capture(
        _campaign: Campaign,
        _name: str,
        command: list[str],
        _marker: Path,
    ) -> None:
        commands.append(command)

    monkeypatch.setattr(Campaign, "_run", capture)
    campaign = Campaign.__new__(Campaign)
    campaign.root = tmp_path / "formal"
    campaign.args = Namespace(snapshot=tmp_path / "snapshot", device="cuda")

    campaign.profile(
        "qkv-profile",
        tmp_path / "candidate",
        arms=("type:self_attn.attn_qkv",),
    )
    campaign.quality(tmp_path / "candidate")
    marker = campaign.cuda_sidecar()

    profile_command, quality_command, cuda_command = commands
    arm_index = profile_command.index("--arm")
    cache_index = profile_command.index("--teacher-cache-root")
    assert profile_command[arm_index + 1] == "type:self_attn.attn_qkv"
    assert Path(profile_command[cache_index + 1]) == campaign.root / "teacher-cache"
    assert "--local-files-only" in profile_command
    assert "--local-files-only" in quality_command
    assert "addopts=" in cuda_command
    assert "-m" in cuda_command and "cuda" in cuda_command
    assert str(marker) in cuda_command


def test_campaign_compression_passes_selected_kl_granularity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def capture(
        _campaign: Campaign,
        _name: str,
        command: list[str],
        _marker: Path,
    ) -> None:
        commands.append(command)

    monkeypatch.setattr(Campaign, "_run", capture)
    campaign = Campaign.__new__(Campaign)
    campaign.root = tmp_path / "formal"
    campaign.args = Namespace(
        snapshot=tmp_path / "snapshot",
        calibration_source=tmp_path / "baseline",
        device="cuda",
        dry_run=True,
    )

    campaign.compress(
        "candidate",
        tmp_path / "profile",
        "sha256:key",
        bias=False,
        alpha_v=1,
        patch_rank=0,
        kl_granularity=KlSensitivityGranularity.TYPE_BLOCK,
        rank_trust_reference_run=tmp_path / "baseline",
        rank_trust_fraction=0.25,
    )

    command = commands[0]
    index = command.index("--kl-granularity")
    assert command[index + 1] == "type_block"
    trust_index = command.index("--rank-trust-fraction")
    assert command[trust_index + 1] == "0.25"
    assert "--rank-trust-reference-run" in command


def test_campaign_reports_d2_rank_and_factor_bit_redistribution() -> None:
    baseline = [
        {"unit_id": "0:mlp.up_proj", "block": 0, "name": "mlp.up_proj", "rank": 4, "factor_bits": 40},
        {
            "unit_id": "12:self_attn.attn_qkv",
            "block": 12,
            "name": "self_attn.attn_qkv",
            "rank": 6,
            "factor_bits": 60,
        },
    ]
    candidate = [
        {"unit_id": "0:mlp.up_proj", "block": 0, "name": "mlp.up_proj", "rank": 6, "factor_bits": 60},
        {
            "unit_id": "12:self_attn.attn_qkv",
            "block": 12,
            "name": "self_attn.attn_qkv",
            "rank": 4,
            "factor_bits": 40,
        },
    ]

    result = _rank_redistribution(baseline, candidate)

    assert result["mlp"]["rank_delta"] == 2
    assert result["attention"]["rank_delta"] == -2
    assert result["early_blocks_0_10"]["factor_bit_delta"] == 20
    assert result["late_blocks_11_17"]["factor_bit_delta"] == -20
    assert _rank_redistribution_gate(result) == {
        "mlp_gained": True,
        "attention_drained": True,
        "early_gained": True,
        "late_drained": True,
        "passed": True,
    }


def test_campaign_uses_persisted_rank_inventory_without_reloading_run(tmp_path: Path) -> None:
    inventory = [
        {
            "unit_id": "0:mlp.up_proj",
            "block": 0,
            "name": "mlp.up_proj",
            "rank": 6,
            "factor_bits": 60,
        }
    ]

    assert _rank_inventory_for_run(tmp_path / "missing-run", {"rank_inventory": inventory}) == inventory


@pytest.mark.parametrize("inventory", [{}, ["not-an-entry"]])
def test_campaign_rejects_malformed_persisted_rank_inventory(
    tmp_path: Path,
    inventory: object,
) -> None:
    with pytest.raises(ValueError, match="malformed rank inventory"):
        _rank_inventory_for_run(tmp_path / "candidate", {"rank_inventory": inventory})
