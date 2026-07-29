from pathlib import Path

import torch

from nanoquant.config.schema import ObjectiveKind
from nanoquant.resident_workflow import ResolvedResidentInputs, resident_request_from_config
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment033_changes_only_identity_output_and_covariance_refinement_from_022(
    tmp_path: Path,
) -> None:
    baseline = load_experiment(22)
    candidate = load_experiment(33)

    objective = candidate.config.calibration.objective
    assert objective.kind is ObjectiveKind.DENSE_HESSIAN
    assert objective.sampling.max_tokens_per_layer == 8192
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert config_diff_paths(baseline.config, candidate.config) == {
        "calibration.objective.kind",
        "calibration.objective.sampling.max_tokens_per_layer",
        "intent.baseline_run",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "intent.experiment_number",
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
            launcher_path=tmp_path / "experiments" / "033.py",
            registry_root=tmp_path / "evidence",
            pad_token_id=0,
        ),
    )
    assert request.covariance_refinement == objective
