from pathlib import Path

import torch

from nanoquant.resident_workflow import ResolvedResidentInputs, resident_request_from_config
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment034_changes_only_identity_output_and_post_refit_selection_from_022(
    tmp_path: Path,
) -> None:
    baseline = load_experiment(22)
    candidate = load_experiment(34)

    selected = candidate.config.block_tuning.post_refit_covariance_refinement
    assert selected.enabled
    assert selected.block_indices == (5, 11, 24, 25)
    assert selected.shared_input_groups == ("self_attn.attn_qkv",)
    assert selected.sampling.max_tokens_per_layer == 8192
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert config_diff_paths(baseline.config, candidate.config) == {
        "block_tuning.post_refit_covariance_refinement.block_indices",
        "block_tuning.post_refit_covariance_refinement.enabled",
        "block_tuning.post_refit_covariance_refinement.shared_input_groups",
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "output.run_root",
    }

    tokens = torch.ones(
        (candidate.config.calibration.sample_count, 8),
        dtype=torch.long,
    )
    request = resident_request_from_config(
        candidate.config,
        ResolvedResidentInputs(
            snapshot=tmp_path / "snapshot",
            output=tmp_path / "output",
            token_ids=tokens,
            quality_token_ids=tokens[:1],
            launcher_path=tmp_path / "experiments" / "034.py",
            registry_root=tmp_path / "evidence",
            pad_token_id=0,
        ),
    )
    assert request.covariance_refinement is None
    assert request.post_refit_covariance_refinement == selected
