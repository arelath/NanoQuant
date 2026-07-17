"""Visible base recipe shared by all complete NanoQuant compression experiments."""

from pathlib import Path

from nanoquant.compression_export_workflow import (
    CompressionExportRecipe,
    HuggingFaceUploadConfig,
)
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    AllocationStrategy,
    CalibrationMethod,
    DatasetSourceConfig,
    ExecutorKind,
    LayerRankBudgetConfig,
    OutlierSelector,
    TuningEpochLossMode,
)

from ._delta import config_delta, run_config_defaults

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

_SCHEMA_DEFAULTS = run_config_defaults("google/gemma-3-1b-it")

BASE_COMPRESSION_CONFIG = config_delta(
    _SCHEMA_DEFAULTS,
    model=config_delta(
        _SCHEMA_DEFAULTS.model,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
    intent=config_delta(
        _SCHEMA_DEFAULTS.intent,
        name="base-compression-gemma-3",
        purpose="Compress, tune, validate, and export a complete NanoQuant model.",
        hypothesis="The shared compression pipeline produces bounded, resumable, deployable GGUF artifacts.",
        tags=("base-compression", "gemma-3", "gguf"),
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
        prepared_artifact="sha256-ad1f609729f86db7598eed5c703c55aacbb9cb024cab816ca7b300d574b7a4c8",
        prepared_root="evidence/m3/experiment018-calibration",
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


LARGE_MODEL_COMPRESSION_CONFIG = config_delta(
    BASE_COMPRESSION_CONFIG,
    intent=config_delta(
        BASE_COMPRESSION_CONFIG.intent,
        name="base-large-model-compression",
        purpose="Compress and evaluate a model that cannot safely keep its BF16 shell on CUDA.",
        hypothesis="Host-resident source blocks and bounded activation caching keep device use block-local.",
        tags=("base-compression", "large-model", "cpu-offload", "packed-quality"),
    ),
    distillation=config_delta(
        BASE_COMPRESSION_CONFIG.distillation,
        enabled=False,
    ),
    calibration=config_delta(
        BASE_COMPRESSION_CONFIG.calibration,
        method=CalibrationMethod.FORWARD_ONLY,
    ),
    runtime=config_delta(
        BASE_COMPRESSION_CONFIG.runtime,
        executor=ExecutorKind.CPU_OFFLOAD,
        activations=config_delta(
            BASE_COMPRESSION_CONFIG.runtime.activations,
            gpu_cache=ActivationGpuCacheMode.AUTO,
        ),
    ),
    evaluation=config_delta(
        BASE_COMPRESSION_CONFIG.evaluation,
        inline_quality=False,
    ),
)


def compression_export_recipe(
    experiment_number: int,
    model_slug: str,
    *,
    token_embedding_type: str = "q8_0",
    huggingface: HuggingFaceUploadConfig | None = None,
) -> CompressionExportRecipe:
    """Return the mandatory deployment outputs for a numbered compression experiment."""

    if experiment_number < 0 or experiment_number > 999:
        raise ValueError("compression experiment number must be between 0 and 999")
    if not model_slug or Path(model_slug).name != model_slug:
        raise ValueError("compression model slug must be one safe path component")
    root = Path(f"outputs/{experiment_number:03d}-{model_slug}")
    return CompressionExportRecipe(
        logical_output=root / "logical",
        packed_output=root / "packed",
        checkpoint_output=root / "llamacpp-checkpoint",
        gguf_output=root / f"{model_slug}-nanoquant.gguf",
        llama_cpp_root=Path(r"D:\dev\research\llama.cpp"),
        runtime_family="gemma3",
        token_embedding_type=token_embedding_type,
        huggingface=huggingface,
    )


__all__ = [
    "BASE_COMPRESSION_CONFIG",
    "LARGE_MODEL_COMPRESSION_CONFIG",
    "MODEL_REVISION",
    "HuggingFaceUploadConfig",
    "compression_export_recipe",
]
