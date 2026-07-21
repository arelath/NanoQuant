"""Fresh-input orchestration for the self-measured Experiment 021 D2 run."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from nanoquant.application.kl_budget import load_kl_budget_profile
from nanoquant.compression_quality_workflow import (
    CompressionQualityExperiment,
    run_compression_quality_experiment,
)
from nanoquant.config.schema import RunConfig
from nanoquant.infrastructure.commits import latest_complete_identity


class _Experiment021Definition(Protocol):
    config: RunConfig
    workflow: CompressionQualityExperiment


def run_experiment021(
    experiment: _Experiment021Definition,
    *,
    launcher_path: str | Path,
    arguments: list[str] | None = None,
) -> int:
    """Resolve and verify same-campaign KL inputs before starting compression."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kl-profile",
        type=Path,
        required=True,
        help="fresh evaluator-v3 profile generated from this campaign's matched control run",
    )
    parser.add_argument(
        "--kl-control-run",
        type=Path,
        required=True,
        help="matched control run inside evidence/021 from which --kl-profile was generated",
    )
    parsed = parser.parse_args(arguments)
    profile_path = parsed.kl_profile.resolve()
    control_run = parsed.kl_control_run.resolve()
    repository = Path(launcher_path).resolve().parent.parent
    campaign_root = (repository / "evidence" / "021").resolve()
    for label, path in (("profile", profile_path), ("control run", control_run)):
        try:
            path.relative_to(campaign_root)
        except ValueError as exc:
            raise ValueError(
                f"Experiment 021 requires its {label} inside its own campaign root: {campaign_root}"
            ) from exc
    profile = load_kl_budget_profile(
        profile_path / "kl-budget-profile.json" if profile_path.is_dir() else profile_path
    )
    records = [
        json.loads(line)
        for line in (control_run / "state" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    control_identity, _control_blocks = latest_complete_identity(records, 18)
    expected_source_identity = (
        f"{control_identity.config_hash}|{control_identity.model_hash}|{control_identity.plan_hash}"
    )
    if not (
        profile.provenance.source_run_identity == expected_source_identity
        or profile.provenance.source_run_identity.startswith(expected_source_identity + "|")
    ):
        raise ValueError("KL profile provenance does not belong to the supplied Experiment 021 control run")
    runtime_config = replace(
        experiment.config,
        allocation=replace(
            experiment.config.allocation,
            kl_profile_artifact=str(profile_path),
            kl_profile_key=profile.profile_key,
        ),
    )
    return run_compression_quality_experiment(
        runtime_config,
        experiment.workflow,
        launcher_path=launcher_path,
    )


__all__ = ["run_experiment021"]
