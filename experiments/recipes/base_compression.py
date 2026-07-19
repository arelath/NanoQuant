"""Visible unnumbered templates shared by NanoQuant compression experiments."""

from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ADMMConfig,
    AllocationStrategy,
    CalibrationMethod,
    DatasetSourceConfig,
    ExecutorKind,
    LayerRankBudgetConfig,
    OutlierSelector,
    RankResponseCurveConfig,
    RankResponseSegmentConfig,
    ReconstructionImportanceConfig,
    ReconstructionRankPlanningConfig,
    SharedInputFactorizationConfig,
    SharedInputGroupConfig,
    TuningEpochLossMode,
)

from ._delta import config_delta, run_config_defaults

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
GEMMA_3_270M_MODEL_REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
GEMMA_3_4B_MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"

_SCHEMA_DEFAULTS = run_config_defaults("google/gemma-3-1b-it")

BASE_COMPRESSION_TEMPLATE = config_delta(
    _SCHEMA_DEFAULTS,
    model=config_delta(
        _SCHEMA_DEFAULTS.model,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
    dataset=config_delta(
        _SCHEMA_DEFAULTS.dataset,
        sources=(
            DatasetSourceConfig(
                "HuggingFaceH4/ultrachat_200k",
                revision="8049631c405ae6576f93f445c6b8166f76f5505a",
                split="train_sft",
                weight=0.5,
            ),
            DatasetSourceConfig(
                "Salesforce/wikitext",
                revision="b08601e04326c79dfdd32d625aee71d232d685c3",
                subset="wikitext-2-raw-v1",
                weight=0.5,
            ),
        ),
        formatting="gemma-chat-plus-raw-text-v1",
    ),
    calibration=config_delta(
        _SCHEMA_DEFAULTS.calibration,
        sample_count=256,
        shrinkage=0.6,
        fallback=config_delta(
            _SCHEMA_DEFAULTS.calibration.fallback,
            on_cuda_oom=("fail",),
        ),
    ),
    allocation=config_delta(
        _SCHEMA_DEFAULTS.allocation,
        strategy=AllocationStrategy.SENSITIVITY,
        maximum_rank_layer_patterns=("self_attn.v_proj", "self_attn.k_proj"),
        layer_budget_multipliers=(LayerRankBudgetConfig("self_attn.q_proj", 1.25),),
        bounds=config_delta(
            _SCHEMA_DEFAULTS.allocation.bounds,
            floor_fraction_of_uniform=0.9,
            ceiling_fraction_of_uniform=1.1,
        ),
        # Legacy's value was two retries after the first attempt; the canonical
        # policy counts all attempts, hence three here.
        retry=config_delta(
            _SCHEMA_DEFAULTS.allocation.retry,
            thresholds=config_delta(
                _SCHEMA_DEFAULTS.allocation.retry.thresholds,
                raw_normalized_error=0.5,
            ),
            maximum_attempts=3,
            allow_above_allocator_cap=True,
        ),
    ),
    outliers=config_delta(
        _SCHEMA_DEFAULTS.outliers,
        selector=OutlierSelector.RESIDUAL,
        fraction=0.001,
        charge_to_bit_budget=False,
    ),
    block_tuning=config_delta(
        _SCHEMA_DEFAULTS.block_tuning,
        layer_order=(
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
        ),
        non_factorized=config_delta(
            _SCHEMA_DEFAULTS.block_tuning.non_factorized,
            epochs_by_layer_position=(8, 4, 3, 2, 2, 2, 2),
        ),
        post_block_refit=config_delta(
            _SCHEMA_DEFAULTS.block_tuning.post_block_refit,
            enabled=True,
            epochs=2,
            batch_size=8,
            scale_learning_rate=1e-5,
        ),
        microbatch_size=8,
        reset_seed_each_stage=True,
        restore_best_state=False,
        epoch_loss_mode=TuningEpochLossMode.LEGACY_TRAINING,
    ),
    distillation=config_delta(
        _SCHEMA_DEFAULTS.distillation,
        enabled=True,
    ),
    runtime=config_delta(
        _SCHEMA_DEFAULTS.runtime,
        executor=ExecutorKind.RESIDENT,
        compute_device="cuda",
        on_cuda_oom=("fail",),
    ),
    output=config_delta(
        _SCHEMA_DEFAULTS.output,
        artifact_root="artifacts",
    ),
)


STACKED_QKV_COMPRESSION_TEMPLATE = config_delta(
    BASE_COMPRESSION_TEMPLATE,
    allocation=config_delta(
        BASE_COMPRESSION_TEMPLATE.allocation,
        maximum_rank_layer_patterns=(),
        layer_budget_multipliers=(),
        retry=config_delta(
            BASE_COMPRESSION_TEMPLATE.allocation.retry,
            enabled=False,
        ),
    ),
    factorization=config_delta(
        BASE_COMPRESSION_TEMPLATE.factorization,
        shared_input=SharedInputFactorizationConfig(
            enabled=True,
            groups=(
                SharedInputGroupConfig(
                    "self_attn.attn_qkv",
                    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                ),
            ),
        ),
    ),
    block_tuning=config_delta(
        BASE_COMPRESSION_TEMPLATE.block_tuning,
        non_factorized=config_delta(
            BASE_COMPRESSION_TEMPLATE.block_tuning.non_factorized,
            # The group replaces V/Q/K as one physical unit. Preserve the
            # baseline's total six dense-tuning epochs across those members.
            epochs_by_layer_position=(8, 4, 3, 6, 2),
        ),
    ),
)


RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE = config_delta(
    STACKED_QKV_COMPRESSION_TEMPLATE,
    allocation=config_delta(
        STACKED_QKV_COMPRESSION_TEMPLATE.allocation,
        strategy=AllocationStrategy.RECONSTRUCTION_AWARE,
        bounds=config_delta(
            STACKED_QKV_COMPRESSION_TEMPLATE.allocation.bounds,
            floor_fraction_of_uniform=0.6,
            ceiling_fraction_of_uniform=1.4,
        ),
        reconstruction=ReconstructionRankPlanningConfig(
            enabled=True,
            probe_admm=ADMMConfig(
                outer_iterations=400,
                inner_iterations=5,
                regularization=3e-2,
                penalty_schedule="cubic",
                convergence_check_interval=100,
                transpose_wide=True,
            ),
            response_curves=(
                RankResponseCurveConfig(
                    "mlp.down_proj",
                    0.6,
                    1.4,
                    (RankResponseSegmentConfig(1.4, 6.22e-4),),
                ),
                RankResponseCurveConfig(
                    "mlp.gate_proj",
                    0.6,
                    1.4,
                    (RankResponseSegmentConfig(1.4, 6.32e-4),),
                ),
                RankResponseCurveConfig(
                    "mlp.up_proj",
                    0.6,
                    1.4,
                    (RankResponseSegmentConfig(1.4, 6.29e-4),),
                ),
                RankResponseCurveConfig(
                    "self_attn.o_proj",
                    0.6,
                    1.4,
                    (RankResponseSegmentConfig(1.4, 1.09e-3),),
                ),
                RankResponseCurveConfig(
                    "self_attn.attn_qkv",
                    0.5,
                    2.0,
                    (
                        RankResponseSegmentConfig(1.0, 1.105e-3),
                        RankResponseSegmentConfig(2.0, 9.03e-4),
                    ),
                ),
            ),
            response_profile_provenance=(
                "Docs/ImprovementSuggestions/ReconstructionHeadroom.md#8;"
                "Docs/ImprovementSuggestions/StackedFactorization.md"
            ),
            sensitivity_strength=0.25,
            protected_sensitivity_quantile=0.80,
            protected_rank_floor_fraction=1.0,
            target_protected_error_reduction_fraction=0.01,
        ),
    ),
)


GEMMA_3_270M_COMPRESSION_TEMPLATE = config_delta(
    BASE_COMPRESSION_TEMPLATE,
    model=config_delta(
        BASE_COMPRESSION_TEMPLATE.model,
        source="unsloth/gemma-3-270m-it",
        revision=GEMMA_3_270M_MODEL_REVISION,
        tokenizer_revision=GEMMA_3_270M_MODEL_REVISION,
    ),
)


GEMMA_3_270M_STACKED_QKV_COMPRESSION_TEMPLATE = config_delta(
    STACKED_QKV_COMPRESSION_TEMPLATE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
)


GEMMA_3_270M_RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE = config_delta(
    RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
)


GEMMA_3_270M_ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE = config_delta(
    GEMMA_3_270M_RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE,
    allocation=config_delta(
        GEMMA_3_270M_RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE.allocation,
        reconstruction=config_delta(
            GEMMA_3_270M_RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE.allocation.reconstruction,
            importance=ReconstructionImportanceConfig(
                layer_multipliers=(
                    LayerRankBudgetConfig("self_attn.q_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.k_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.v_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.o_proj", 1.25),
                    LayerRankBudgetConfig("mlp.down_proj", 1.25),
                ),
                protected_layer_patterns=(
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.down_proj",
                ),
                edge_block_multiplier=1.25,
                protected_edge_block_count=1,
            ),
        ),
    ),
)


_4B_TUNING = BASE_COMPRESSION_TEMPLATE.block_tuning

GEMMA_3_4B_COMPRESSION_TEMPLATE = config_delta(
    BASE_COMPRESSION_TEMPLATE,
    model=config_delta(
        BASE_COMPRESSION_TEMPLATE.model,
        source="google/gemma-3-4b-it",
        revision=GEMMA_3_4B_MODEL_REVISION,
        tokenizer_revision=GEMMA_3_4B_MODEL_REVISION,
    ),
    allocation=config_delta(
        BASE_COMPRESSION_TEMPLATE.allocation,
        retry=config_delta(
            BASE_COMPRESSION_TEMPLATE.allocation.retry,
            thresholds=config_delta(
                BASE_COMPRESSION_TEMPLATE.allocation.retry.thresholds,
                weighted_normalized_error=0.35,
                raw_normalized_error=0.40,
            ),
        ),
    ),
    block_tuning=config_delta(
        _4B_TUNING,
        non_factorized=config_delta(
            _4B_TUNING.non_factorized,
            loop=config_delta(_4B_TUNING.non_factorized.loop, batch_size=4),
        ),
        factorized=config_delta(
            _4B_TUNING.factorized,
            loop=config_delta(_4B_TUNING.factorized.loop, batch_size=1),
        ),
        post_block_refit=config_delta(
            _4B_TUNING.post_block_refit,
            batch_size=1,
        ),
        microbatch_size=1,
    ),
    runtime=config_delta(
        BASE_COMPRESSION_TEMPLATE.runtime,
        block_forward_batch_size=4,
    ),
    evaluation=config_delta(
        BASE_COMPRESSION_TEMPLATE.evaluation,
        inline_quality=False,
    ),
    observability=config_delta(
        BASE_COMPRESSION_TEMPLATE.observability,
        record_resource_interval_seconds=1.0,
    ),
    profiling=config_delta(
        BASE_COMPRESSION_TEMPLATE.profiling,
        cuda_timing=True,
        memory_counters=True,
        emit_span_events=True,
    ),
)


LARGE_MODEL_COMPRESSION_TEMPLATE = config_delta(
    BASE_COMPRESSION_TEMPLATE,
    distillation=config_delta(
        BASE_COMPRESSION_TEMPLATE.distillation,
        enabled=False,
    ),
    calibration=config_delta(
        BASE_COMPRESSION_TEMPLATE.calibration,
        method=CalibrationMethod.FORWARD_ONLY,
    ),
    runtime=config_delta(
        BASE_COMPRESSION_TEMPLATE.runtime,
        executor=ExecutorKind.CPU_OFFLOAD,
        activations=config_delta(
            BASE_COMPRESSION_TEMPLATE.runtime.activations,
            gpu_cache=ActivationGpuCacheMode.AUTO,
            gpu_reserve_gib=4.0,
        ),
    ),
    evaluation=config_delta(
        BASE_COMPRESSION_TEMPLATE.evaluation,
        inline_quality=False,
    ),
)


__all__ = [
    "BASE_COMPRESSION_TEMPLATE",
    "GEMMA_3_270M_COMPRESSION_TEMPLATE",
    "GEMMA_3_270M_MODEL_REVISION",
    "GEMMA_3_270M_RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE",
    "GEMMA_3_270M_STACKED_QKV_COMPRESSION_TEMPLATE",
    "GEMMA_3_4B_COMPRESSION_TEMPLATE",
    "GEMMA_3_4B_MODEL_REVISION",
    "LARGE_MODEL_COMPRESSION_TEMPLATE",
    "MODEL_REVISION",
    "RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE",
    "STACKED_QKV_COMPRESSION_TEMPLATE",
]
