"""Experiment 070: spend Experiment 060's residual-column budget on independent sign-factor stages."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_callable_main,
)
from recipes._delta import retained_config

from nanoquant.config.schema import OutlierSelector, ResidualFactorizationConfig
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

REFERENCE = retained_config(
    Path(__file__).parent / "recipes" / "references" / "experiment060.json",
    "sha256:3a48eafd5f6060c0e7617bec0ec94187310bee4990c7ea0c96cf147c05b2058a",
)
CONFIG = replace(
    REFERENCE,
    allocation=replace(
        REFERENCE.allocation,
        bounds=replace(REFERENCE.allocation.bounds, overcomplete_rank_ceiling_fraction=None),
        retry=replace(REFERENCE.allocation.retry, outlier_count_increment_at_rank_cap=0),
        kl_profile_artifact="runtime-kl-profile-required",
        kl_profile_key="runtime-kl-profile-key-required",
    ),
    factorization=replace(REFERENCE.factorization, residual_stages=ResidualFactorizationConfig(enabled=True)),
    outliers=replace(REFERENCE.outliers, selector=OutlierSelector.NONE, fraction=0.0),
)
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(wikitext_samples=48, sequence_length=512, tuned_operating_point=True)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=70,
        name="independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it",
        purpose=(
            "Match the retained Experiment 060 recipe while replacing charged INT8 residual columns "
            "with independently scaled binary residual stages under the same represented-bit budget."
        ),
        hypothesis=(
            "Independent residual stages can outperform Experiment 060 when their combined rank "
            "is budget limited and may exceed the matrix dimension."
        ),
        baseline=BaselineRef.experiment(ExperimentRef(60, "best-methods-2bpw-compress-and-benchmark-gemma-3-1b-it")),
        tags=(
            "gemma-3-1b-it",
            "compression",
            "quality",
            "2-bpw",
            "d2",
            "independent-residual-stages",
            "budget-limited-rank",
            "stacked-qkv",
            "global-distillation",
            "task-limit-1000",
        ),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    interrupt_after_block_commits=1,
    task_limit=1000,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_callable_main(
    __name__,
    lambda: run_self_measured_d2_experiment(EXPERIMENT, launcher_path=__file__, profile_options=PROFILE_OPTIONS),
)
