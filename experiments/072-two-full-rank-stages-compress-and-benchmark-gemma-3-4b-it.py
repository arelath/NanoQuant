"""Experiment 072: transfer 071's two full-rank stages to pinned Gemma 3 4B IT."""

from dataclasses import replace
from pathlib import Path

from recipes import (
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
    experiment_callable_main,
)
from recipes._delta import config_delta, retained_config

from nanoquant.config.schema import FactorizedOptimizerConfig
from nanoquant.self_measured_d2_workflow import SelfMeasuredD2ProfileOptions, run_self_measured_d2_experiment

REFERENCE = retained_config(
    Path(__file__).parent / "recipes" / "references" / "experiment071.json",
    "sha256:2ae172330b1e6d7db5b1ff878a18f3c5b0a169706426a02088c8f9c9fbeb25f5",
)
MODEL_TEMPLATE = GEMMA_3_4B_COMPRESSION_TEMPLATE
CONFIG = config_delta(
    REFERENCE,
    model=MODEL_TEMPLATE.model,
    allocation=config_delta(
        REFERENCE.allocation,
        # The first 4B campaign must measure its own profile, never inherit 070's 1B profile.
        kl_profile_artifact="runtime-kl-profile-required",
        kl_profile_key="runtime-kl-profile-key-required",
    ),
    block_tuning=config_delta(
        REFERENCE.block_tuning,
        non_factorized=config_delta(
            REFERENCE.block_tuning.non_factorized,
            loop=config_delta(REFERENCE.block_tuning.non_factorized.loop,
                              batch_size=MODEL_TEMPLATE.block_tuning.non_factorized.loop.batch_size),
        ),
        factorized=config_delta(
            REFERENCE.block_tuning.factorized,
            # Both-seed held-out gate passed for these owners. Finish the saved
            # partial block 18 with parity; retain up and shared QKV on parity.
            optimizer=FactorizedOptimizerConfig(
                name="fused_adamw_scales_fp32",
                start_block=19,
                layer_patterns=("mlp.gate_proj", "mlp.down_proj", "self_attn.o_proj"),
                continuation_journal_sha256="f0b65f0c251c7acc033f6cc7d06de4d8c3e4a3dd112946830c77f88a1408e983",
            ),
            loop=config_delta(REFERENCE.block_tuning.factorized.loop,
                              batch_size=MODEL_TEMPLATE.block_tuning.factorized.loop.batch_size),
        ),
        post_block_refit=config_delta(
            REFERENCE.block_tuning.post_block_refit,
            batch_size=MODEL_TEMPLATE.block_tuning.post_block_refit.batch_size,
        ),
        microbatch_size=MODEL_TEMPLATE.block_tuning.microbatch_size,
    ),
    runtime=config_delta(REFERENCE.runtime, block_forward_batch_size=MODEL_TEMPLATE.runtime.block_forward_batch_size),
    evaluation=config_delta(REFERENCE.evaluation, inline_quality=MODEL_TEMPLATE.evaluation.inline_quality),
)
KL_TEACHER_DATASET = Path(__file__).resolve().parents[1] / "Datasets" / "gemma-3-4b-it-kl-48x512"
PROFILE_OPTIONS = SelfMeasuredD2ProfileOptions(wikitext_samples=48, sequence_length=512, tuned_operating_point=True)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=72,
        name="two-full-rank-stages-compress-and-benchmark-gemma-3-4b-it",
        purpose=("Transfer Experiment 071's two full-rank independent stages and 600x2 compiled recipe "
                 "to Gemma 3 4B IT."),
        hypothesis="Two independently scaled full-rank stages preserve quality on the larger Gemma text decoder.",
        baseline=BaselineRef.experiment(ExperimentRef(18, "compress-and-benchmark-gemma-3-4b-it")),
        tags=("gemma-3-4b-it", "compression", "quality", "two-full-rank-stages", "admm-600x2",
              "compiled-block-tuning", "kl-teacher-dataset", "stacked-qkv", "global-distillation", "task-limit-1000"),
    ),
    CONFIG,
    maximum_wddm_shared_gib=0.75,
    # Resume from the committed activation boundary. Final evaluation loads the exported weights separately.
    restore_completed_blocks=False,
    interrupt_after_block_commits=1,
    quality_backend="dense",
    task_limit=1000,
)
EXPERIMENT = replace(EXPERIMENT, workflow=replace(EXPERIMENT.workflow, local_files_only=True))

experiment_callable_main(
    __name__,
    lambda: run_self_measured_d2_experiment(
        EXPERIMENT, launcher_path=__file__, profile_options=PROFILE_OPTIONS, kl_teacher_dataset=KL_TEACHER_DATASET,
        reuse_maximum_rank_control=True,
    ),
)
