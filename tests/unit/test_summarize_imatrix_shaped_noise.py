import pytest
from compare_imatrix_shaped_noise import MODES
from summarize_imatrix_shaped_noise import analyze


def fixture_report():
    targets = [f"{block}:mlp.down_proj" for block in (0, 13, 6, 19)]
    report = {
        "complete": True, "embedding_verified_unchanged": True,
        "identity": {"plan": {"seeds": [1, 2], "sources": {"NanoQuant": {"targets": targets}},
                              "previously_sampled_blocks": [0, 13], "confirmation_blocks": [6, 19]}},
        "clean": {"sequences": [{}] * 4}, "cases": {},
    }
    for index, target in enumerate(targets, 1):
        kl = index * .01
        metrics = {"raw_error": .1, "weighted_error": .05, "weighted_to_raw_energy": .25}

        def probe(multiplier, kl=kl, metrics=metrics):
            return {**metrics, "kl_above_clean": multiplier * kl,
                    "score": {"sequences": [{"kl_nats_per_token": multiplier * kl * f}
                                            for f in (.8, .9, 1.1, 1.2)]}}

        report["cases"][target] = {
            "source": "NanoQuant", "target": target, "storage_type": "binary", "actual_residual": metrics,
            "importance_floored_fraction": 0, "curve_raw_extrapolated": False,
            "curve_weighted_extrapolated": False, "curve_raw": kl * 2, "curve_weighted": kl,
            "probes": {"actual": probe(1), **{f"{mode}:{seed}": probe(2 if mode == "isotropic" else 1)
                                              for mode in MODES for seed in (1, 2)}},
        }
    return report


def test_analysis_separates_confirmation_and_noise_metrics():
    result = analyze(fixture_report())
    all_results = result["summaries"]["all"]["NanoQuant"]
    assert all_results["count"] == 4
    assert all_results["predictors"]["isotropic"]["median_fold_error"] == 2
    assert all_results["predictors"]["imatrix"]["median_fold_error"] == 1
    assert result["summaries"]["confirmation_blocks"]["NanoQuant"]["count"] == 2
    assert result["max_post_cast_relative_norm_mismatch"] == 0
    assert result["common_projection_energy_ratios"]["NanoQuant"]["median_actual_weighted_to_raw_energy"] == .25


def test_analysis_rejects_incomplete_probe_inventory():
    report = fixture_report()
    del report["cases"]["0:mlp.down_proj"]["probes"]["isotropic:1"]
    with pytest.raises(ValueError, match="missing"):
        analyze(report)
