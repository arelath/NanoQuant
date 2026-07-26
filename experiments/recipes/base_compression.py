"""Visible unnumbered templates shared by NanoQuant compression experiments."""

from dataclasses import replace
from pathlib import Path

from nanoquant.config.codec import apply_overrides
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ADMMConfig,
    AllocationStrategy,
    BehaviorSliceConfig,
    CalibrationMethod,
    DatasetSourceConfig,
    EvaluationTier,
    ExecutorKind,
    LayerRankBudgetConfig,
    MemoryPolicyMode,
    MemoryPolicyProfile,
    OutlierSelector,
    RankResponseCurveConfig,
    RankResponseSegmentConfig,
    ReasoningMode,
    ReconstructionImportanceConfig,
    ReconstructionRankPlanningConfig,
    RunConfig,
    SharedInputFactorizationConfig,
    SharedInputGroupConfig,
    TeacherTraceGenerationConfig,
    TuningEpochLossMode,
)

from ._delta import config_delta, run_config_defaults
from ._interactive_catalog import load_interactive_recommended_models

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
GEMMA_3_270M_MODEL_REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
GEMMA_3_4B_MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
LLAMA_3_2_1B_INSTRUCT_MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
LLAMA_3_2_3B_INSTRUCT_MODEL_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
META_LLAMA_3_8B_INSTRUCT_MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
QWEN_3_0_6B_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN_3_8B_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

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
            sensitivity_strength=0.75,
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


ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE = config_delta(
    RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE,
    allocation=config_delta(
        RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE.allocation,
        reconstruction=config_delta(
            RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE.allocation.reconstruction,
            importance=ReconstructionImportanceConfig(
                layer_multipliers=(
                    LayerRankBudgetConfig("self_attn.q_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.k_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.v_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.o_proj", 1.25),
                    LayerRankBudgetConfig("mlp.down_proj", 1.50),
                ),
                protected_layer_patterns=(
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.down_proj",
                ),
                edge_block_multiplier=1.30,
                protected_edge_block_count=1,
            ),
        ),
    ),
)


GEMMA_3_4B_COMPRESSION_TEMPLATE = apply_overrides(
    BASE_COMPRESSION_TEMPLATE,
    {
        "model.source": "google/gemma-3-4b-it",
        "model.revision": GEMMA_3_4B_MODEL_REVISION,
        "model.tokenizer_revision": GEMMA_3_4B_MODEL_REVISION,
        "allocation.retry.thresholds.weighted_normalized_error": 0.35,
        "allocation.retry.thresholds.raw_normalized_error": 0.40,
        "block_tuning.non_factorized.loop.batch_size": 4,
        "block_tuning.factorized.loop.batch_size": 1,
        "block_tuning.post_block_refit.batch_size": 1,
        "block_tuning.microbatch_size": 1,
        "runtime.block_forward_batch_size": 4,
        "evaluation.inline_quality": False,
        "observability.record_resource_interval_seconds": 1.0,
        "profiling.cuda_timing": True,
        "profiling.memory_counters": True,
        "profiling.emit_span_events": True,
    },
)


LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE = apply_overrides(
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    {
        "model.source": "meta-llama/Llama-3.2-1B-Instruct",
        "model.revision": LLAMA_3_2_1B_INSTRUCT_MODEL_REVISION,
        "model.tokenizer_revision": LLAMA_3_2_1B_INSTRUCT_MODEL_REVISION,
    },
)


_LLAMA_ARCHITECTURE_POLICY = config_delta(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    allocation=config_delta(
        ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE.allocation,
        reconstruction=config_delta(
            ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE.allocation.reconstruction,
            sensitivity_strength=0.5,
        ),
    ),
)


LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE = config_delta(
    LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE,
    allocation=_LLAMA_ARCHITECTURE_POLICY.allocation,
    factorization=_LLAMA_ARCHITECTURE_POLICY.factorization,
    block_tuning=config_delta(
        LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE.block_tuning,
        non_factorized=config_delta(
            LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE.block_tuning.non_factorized,
            epochs_by_layer_position=(
                _LLAMA_ARCHITECTURE_POLICY.block_tuning.non_factorized.epochs_by_layer_position
            ),
        ),
    ),
)


LLAMA_3_2_3B_INSTRUCT_COMPRESSION_TEMPLATE = apply_overrides(
    LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE,
    {
        "model.source": "meta-llama/Llama-3.2-3B-Instruct",
        "model.revision": LLAMA_3_2_3B_INSTRUCT_MODEL_REVISION,
        "model.tokenizer_revision": LLAMA_3_2_3B_INSTRUCT_MODEL_REVISION,
    },
)


META_LLAMA_3_8B_INSTRUCT_COMPRESSION_TEMPLATE = config_delta(
    LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE,
    model=config_delta(
        LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.model,
        source="meta-llama/Meta-Llama-3-8B-Instruct",
        revision=META_LLAMA_3_8B_INSTRUCT_MODEL_REVISION,
        tokenizer_revision=META_LLAMA_3_8B_INSTRUCT_MODEL_REVISION,
    ),
    block_tuning=config_delta(
        LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.block_tuning,
        # Use one reviewed logical batch across each tuning stage. The adaptive
        # memory plan still selects a physical microbatch from 1..32, so the
        # optimizer semantics are hardware-independent while CUDA occupancy can
        # grow on larger devices.
        non_factorized=config_delta(
            LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.block_tuning.non_factorized,
            loop=config_delta(
                LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.block_tuning.non_factorized.loop,
                batch_size=32,
            ),
        ),
        factorized=config_delta(
            LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.block_tuning.factorized,
            loop=config_delta(
                LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.block_tuning.factorized.loop,
                batch_size=32,
            ),
        ),
        post_block_refit=config_delta(
            LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.block_tuning.post_block_refit,
            batch_size=32,
        ),
        microbatch_size=None,
    ),
    runtime=config_delta(
        LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.runtime,
        memory_policy=config_delta(
            LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.runtime.memory_policy,
            mode=MemoryPolicyMode.ADAPTIVE,
            profile=MemoryPolicyProfile.THROUGHPUT,
        ),
        activations=config_delta(
            LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE.runtime.activations,
            gpu_cache=ActivationGpuCacheMode.AUTO,
        ),
        on_cuda_oom=("reduce_batch_size", "move_activations_down_one_tier", "fail"),
    ),
)


QWEN_3_0_6B_COMPRESSION_TEMPLATE = apply_overrides(
    META_LLAMA_3_8B_INSTRUCT_COMPRESSION_TEMPLATE,
    {
        "model.source": "Qwen/Qwen3-0.6B",
        "model.revision": QWEN_3_0_6B_MODEL_REVISION,
        "model.tokenizer_revision": QWEN_3_0_6B_MODEL_REVISION,
    },
)


QWEN_3_8B_COMPRESSION_TEMPLATE = apply_overrides(
    QWEN_3_0_6B_COMPRESSION_TEMPLATE,
    {
        "model.source": "Qwen/Qwen3-8B",
        "model.revision": QWEN_3_8B_MODEL_REVISION,
        "model.tokenizer_revision": QWEN_3_8B_MODEL_REVISION,
    },
)

OPENR1_MATH_220K_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"

_QWEN_DUAL_MODE_SLICES = (
    BehaviorSliceConfig(
        "raw",
        ReasoningMode.RAW,
        DatasetSourceConfig(
            "Salesforce/wikitext",
            revision="b08601e04326c79dfdd32d625aee71d232d685c3",
            subset="wikitext-2-raw-v1",
        ),
        "raw_text",
        0.25,
        minimum_valid_tokens=256 // 2 * 2048,
    ),
    BehaviorSliceConfig(
        "non-thinking",
        ReasoningMode.NON_THINKING,
        DatasetSourceConfig(
            "HuggingFaceH4/ultrachat_200k",
            revision="8049631c405ae6576f93f445c6b8166f76f5505a",
            split="train_sft",
        ),
        "ultrachat_messages",
        0.25,
        minimum_valid_tokens=256 // 2 * 2048,
    ),
    BehaviorSliceConfig(
        "thinking",
        ReasoningMode.THINKING,
        DatasetSourceConfig(
            "open-r1/OpenR1-Math-220k",
            revision=OPENR1_MATH_220K_REVISION,
        ),
        "openr1_generations",
        0.50,
    ),
)

QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE = config_delta(
    QWEN_3_0_6B_COMPRESSION_TEMPLATE,
    dataset=config_delta(
        QWEN_3_0_6B_COMPRESSION_TEMPLATE.dataset,
        sources=tuple(item.source for item in _QWEN_DUAL_MODE_SLICES),
        formatting="qwen3-dual-mode-behavior-v1",
        behavior_slices=_QWEN_DUAL_MODE_SLICES,
    ),
    calibration=config_delta(
        QWEN_3_0_6B_COMPRESSION_TEMPLATE.calibration,
        sample_count=528,
    ),
    evaluation=config_delta(
        QWEN_3_0_6B_COMPRESSION_TEMPLATE.evaluation,
        reasoning_modes=(ReasoningMode.THINKING, ReasoningMode.NON_THINKING),
    ),
)

QWEN_3_8B_DUAL_MODE_COMPRESSION_TEMPLATE = config_delta(
    QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE,
    model=QWEN_3_8B_COMPRESSION_TEMPLATE.model,
    evaluation=config_delta(
        QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE.evaluation,
        default_tier=EvaluationTier.FULL,
    ),
)

_QWEN_TEACHER_TRACE_SLICES = (
    _QWEN_DUAL_MODE_SLICES[0],
    replace(
        _QWEN_DUAL_MODE_SLICES[1],
        teacher_trace_generation=TeacherTraceGenerationConfig(),
    ),
    BehaviorSliceConfig(
        "thinking",
        ReasoningMode.THINKING,
        DatasetSourceConfig(
            "HuggingFaceH4/ultrachat_200k",
            revision="8049631c405ae6576f93f445c6b8166f76f5505a",
            split="train_sft",
        ),
        "ultrachat_messages",
        0.50,
        minimum_valid_tokens=256 // 2 * 2048,
        teacher_trace_generation=TeacherTraceGenerationConfig(),
    ),
)

QWEN_3_0_6B_TEACHER_TRACE_COMPRESSION_TEMPLATE = config_delta(
    QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE,
    dataset=config_delta(
        QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE.dataset,
        sources=tuple(item.source for item in _QWEN_TEACHER_TRACE_SLICES),
        formatting="qwen3-ultrachat-teacher-traces-v1",
        behavior_slices=_QWEN_TEACHER_TRACE_SLICES,
    ),
)

QWEN_3_8B_TEACHER_TRACE_COMPRESSION_TEMPLATE = config_delta(
    QWEN_3_0_6B_TEACHER_TRACE_COMPRESSION_TEMPLATE,
    model=QWEN_3_8B_COMPRESSION_TEMPLATE.model,
    evaluation=QWEN_3_8B_DUAL_MODE_COMPRESSION_TEMPLATE.evaluation,
)


_INTERACTIVE_GEMMA_3_1B_TEMPLATE = apply_overrides(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    {
        "allocation.reconstruction.sensitivity_strength": 0.5,
    },
)

_INTERACTIVE_GEMMA_3_270M_TEMPLATE = replace(
    ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
)

_INTERACTIVE_GEMMA_3_4B_TEMPLATE = replace(
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    allocation=_INTERACTIVE_GEMMA_3_1B_TEMPLATE.allocation,
    factorization=_INTERACTIVE_GEMMA_3_1B_TEMPLATE.factorization,
    block_tuning=replace(
        GEMMA_3_4B_COMPRESSION_TEMPLATE.block_tuning,
        non_factorized=replace(
            GEMMA_3_4B_COMPRESSION_TEMPLATE.block_tuning.non_factorized,
            epochs_by_layer_position=(
                _INTERACTIVE_GEMMA_3_1B_TEMPLATE.block_tuning.non_factorized.epochs_by_layer_position
            ),
        ),
    ),
)

GEMMA_3_12B_MODEL_REVISION = "9478e665381f42974aa06177b019352fb6291876"
_INTERACTIVE_GEMMA_3_12B_TEMPLATE = apply_overrides(
    BASE_COMPRESSION_TEMPLATE,
    {
        "model.source": "unsloth/gemma-3-12b-it",
        "model.revision": GEMMA_3_12B_MODEL_REVISION,
        "model.tokenizer_source": "unsloth/gemma-3-12b-it",
        "model.tokenizer_revision": GEMMA_3_12B_MODEL_REVISION,
        "distillation.enabled": False,
        "calibration.method": "forward_only",
        "runtime.executor": "cpu_offload",
        "runtime.activations.gpu_cache": "auto",
        "runtime.activations.gpu_reserve_gib": 4.0,
        "evaluation.inline_quality": False,
        "block_tuning.microbatch_size": 1,
        "runtime.block_forward_batch_size": 1,
    },
)

_INTERACTIVE_QWEN_REVISIONS = {
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "Qwen/Qwen3.6-27B": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
}


def _interactive_qwen_template(source: str, revision: str) -> RunConfig:
    return apply_overrides(
        QWEN_3_8B_TEACHER_TRACE_COMPRESSION_TEMPLATE,
        {
            "model.source": source,
            "model.revision": revision,
            "model.tokenizer_source": None,
            "model.tokenizer_revision": revision,
        },
    )


_INTERACTIVE_QWEN_TEMPLATES = {
    source: _interactive_qwen_template(source, revision)
    for source, revision in _INTERACTIVE_QWEN_REVISIONS.items()
}

_INTERACTIVE_TEMPLATES = {
    "Qwen/Qwen3-0.6B": QWEN_3_0_6B_TEACHER_TRACE_COMPRESSION_TEMPLATE,
    "Qwen/Qwen3-8B": QWEN_3_8B_TEACHER_TRACE_COMPRESSION_TEMPLATE,
    **_INTERACTIVE_QWEN_TEMPLATES,
    "unsloth/gemma-3-270m-it": _INTERACTIVE_GEMMA_3_270M_TEMPLATE,
    "google/gemma-3-1b-it": _INTERACTIVE_GEMMA_3_1B_TEMPLATE,
    "google/gemma-3-4b-it": _INTERACTIVE_GEMMA_3_4B_TEMPLATE,
    "unsloth/gemma-3-12b-it": _INTERACTIVE_GEMMA_3_12B_TEMPLATE,
    "meta-llama/Meta-Llama-3-8B-Instruct": META_LLAMA_3_8B_INSTRUCT_COMPRESSION_TEMPLATE,
    "meta-llama/Llama-3.2-1B-Instruct": LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE,
    "meta-llama/Llama-3.2-3B-Instruct": LLAMA_3_2_3B_INSTRUCT_COMPRESSION_TEMPLATE,
}

INTERACTIVE_RECOMMENDED_MODELS = load_interactive_recommended_models(
    Path(__file__).with_name("interactive_recommended_models.yaml"),
    _INTERACTIVE_TEMPLATES,
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
    "ARCHITECTURE_PROTECTED_RECONSTRUCTION_COMPRESSION_TEMPLATE",
    "BASE_COMPRESSION_TEMPLATE",
    "GEMMA_3_270M_COMPRESSION_TEMPLATE",
    "GEMMA_3_270M_MODEL_REVISION",
    "GEMMA_3_270M_STACKED_QKV_COMPRESSION_TEMPLATE",
    "GEMMA_3_12B_MODEL_REVISION",
    "GEMMA_3_4B_COMPRESSION_TEMPLATE",
    "GEMMA_3_4B_MODEL_REVISION",
    "INTERACTIVE_RECOMMENDED_MODELS",
    "LARGE_MODEL_COMPRESSION_TEMPLATE",
    "LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE",
    "LLAMA_3_2_1B_INSTRUCT_MODEL_REVISION",
    "LLAMA_3_2_3B_INSTRUCT_COMPRESSION_TEMPLATE",
    "LLAMA_3_2_3B_INSTRUCT_MODEL_REVISION",
    "LLAMA_ARCHITECTURE_PROTECTED_COMPRESSION_TEMPLATE",
    "META_LLAMA_3_8B_INSTRUCT_COMPRESSION_TEMPLATE",
    "META_LLAMA_3_8B_INSTRUCT_MODEL_REVISION",
    "MODEL_REVISION",
    "QWEN_3_0_6B_COMPRESSION_TEMPLATE",
    "QWEN_3_0_6B_DUAL_MODE_COMPRESSION_TEMPLATE",
    "QWEN_3_0_6B_TEACHER_TRACE_COMPRESSION_TEMPLATE",
    "QWEN_3_0_6B_MODEL_REVISION",
    "QWEN_3_8B_COMPRESSION_TEMPLATE",
    "QWEN_3_8B_DUAL_MODE_COMPRESSION_TEMPLATE",
    "QWEN_3_8B_TEACHER_TRACE_COMPRESSION_TEMPLATE",
    "QWEN_3_8B_MODEL_REVISION",
    "RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE",
    "STACKED_QKV_COMPRESSION_TEMPLATE",
]
