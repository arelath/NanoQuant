"""The sole canonical run-configuration schema.

Defaults in this module are normative. Entry points decode or construct these
frozen values and never maintain a second set of defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StringEnum(str, Enum):
    """String-valued enum with stable serialized values."""


class DType(StringEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    INT8 = "int8"


class ExecutorKind(StringEnum):
    AUTO = "auto"
    RESIDENT = "resident"
    CPU_OFFLOAD = "cpu_offload"
    STREAMING = "streaming"
    DISTRIBUTED = "distributed"


class ActivationStoreKind(StringEnum):
    AUTO = "auto"
    CUDA = "cuda"
    PINNED_RAM = "pinned_ram"
    RAM = "ram"
    MMAP = "mmap"


class ActivationGpuCacheMode(StringEnum):
    OFF = "off"
    INPUTS = "inputs"
    BOTH = "both"
    AUTO = "auto"


class CalibrationMethod(StringEnum):
    ONLINE_FISHER = "online_fisher"
    TWO_PHASE_FISHER = "two_phase_fisher"
    FORWARD_ONLY = "forward_only"
    DBF = "dbf"
    NONE = "none"


class ObjectiveKind(StringEnum):
    DIAGONAL = "diagonal"
    BLOCK_DIAGONAL = "block_diagonal"
    LOW_RANK_DIAGONAL = "low_rank_diagonal"
    DENSE_HESSIAN = "dense_hessian"


class AllocationStrategy(StringEnum):
    UNIFORM = "uniform"
    SENSITIVITY = "sensitivity"
    UTILITY_PROFILE = "utility_profile"


class OutlierSelector(StringEnum):
    NONE = "none"
    FISHER = "fisher"
    RESIDUAL = "residual"


class DistillationLoss(StringEnum):
    TOP_K = "top_k"
    FULL_KL = "full_kl"


class EvaluationTier(StringEnum):
    SMOKE = "smoke"
    QUICK = "quick"
    STANDARD = "standard"
    FULL = "full"


class ProfilingLevel(StringEnum):
    OFF = "off"
    MACRO = "macro"
    MICRO = "micro"
    TRACE = "trace"


class TuningEpochLossMode(StringEnum):
    FULL_EVALUATION = "full_evaluation"
    LEGACY_TRAINING = "legacy_training"


class ActivationRetention(StringEnum):
    ROLLING = "rolling"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class IntentConfig:
    experiment_number: int | None = None
    name: str = "unnamed-run"
    purpose: str = ""
    hypothesis: str = ""
    baseline_run: str | None = None
    owner: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    source: str
    revision: str | None = None
    tokenizer_source: str | None = None
    tokenizer_revision: str | None = None
    sequence_length: int = 2048
    load_dtype: DType = DType.BFLOAT16
    trust_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class DatasetSourceConfig:
    name: str
    revision: str | None = None
    split: str = "train"
    subset: str | None = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    sources: tuple[DatasetSourceConfig, ...] = (DatasetSourceConfig(name="wikitext2"),)
    formatting: str = "model_default"
    shuffle: bool = True
    selection_seed: int = 0
    cache_tokenized: bool = True


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    seed: int = 0
    deterministic: bool = True
    allow_nondeterministic_kernels: bool = False


@dataclass(frozen=True, slots=True)
class HessianSamplingConfig:
    max_tokens_per_layer: int = 4096
    max_sequences: int | None = None
    batch_size: int = 1
    reuse_sibling_inputs: bool = False


@dataclass(frozen=True, slots=True)
class HessianRegularizationConfig:
    diagonal_damp_fraction: float = 0.01
    identity_shrinkage: float = 0.0
    diagonal_blend: float = 0.0
    jitter_attempts: int = 5


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    kind: ObjectiveKind = ObjectiveKind.DIAGONAL
    block_size: int | None = None
    low_rank: int | None = None
    sampling: HessianSamplingConfig = field(default_factory=HessianSamplingConfig)
    regularization: HessianRegularizationConfig = field(default_factory=HessianRegularizationConfig)


@dataclass(frozen=True, slots=True)
class CalibrationFallbackConfig:
    on_cuda_oom: tuple[str, ...] = ("cpu_offload", "forward_only", "fail")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    method: CalibrationMethod = CalibrationMethod.ONLINE_FISHER
    sample_count: int = 128
    batch_size: int = 1
    shrinkage: float = 0.4
    accumulation_dtype: DType = DType.FLOAT32
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    fallback: CalibrationFallbackConfig = field(default_factory=CalibrationFallbackConfig)


@dataclass(frozen=True, slots=True)
class RankBoundsConfig:
    multiple: int = 32
    floor_fraction_of_uniform: float = 0.80
    ceiling_fraction_of_uniform: float = 1.15
    edge_block_boost: float = 0.15


@dataclass(frozen=True, slots=True)
class RetryThresholdConfig:
    weighted_normalized_error: float | None = 0.50
    raw_normalized_error: float | None = None


@dataclass(frozen=True, slots=True)
class RankRetryConfig:
    enabled: bool = True
    thresholds: RetryThresholdConfig = field(default_factory=RetryThresholdConfig)
    rank_increase_fraction: float = 0.25
    maximum_attempts: int = 2
    extra_bit_budget_fraction: float = 0.02
    allow_above_allocator_cap: bool = False


@dataclass(frozen=True, slots=True)
class LayerRankBudgetConfig:
    pattern: str
    multiplier: float


@dataclass(frozen=True, slots=True)
class RankAllocationConfig:
    target_bpw: float = 1.0
    strategy: AllocationStrategy = AllocationStrategy.UNIFORM
    sensitivity_alpha: float = 0.5
    utility_profile_artifact: str | None = None
    maximum_rank_layer_patterns: tuple[str, ...] = ()
    layer_budget_multipliers: tuple[LayerRankBudgetConfig, ...] = ()
    bounds: RankBoundsConfig = field(default_factory=RankBoundsConfig)
    retry: RankRetryConfig = field(default_factory=RankRetryConfig)


@dataclass(frozen=True, slots=True)
class ADMMConfig:
    outer_iterations: int = 800
    inner_iterations: int = 5
    regularization: float = 3e-2
    penalty_schedule: str = "cubic"
    convergence_check_interval: int = 100
    early_stop_tolerance: float | None = None
    transpose_wide: bool = False


@dataclass(frozen=True, slots=True)
class ScaleFitConfig:
    enabled: bool = True
    alternating_passes: int = 2
    epsilon: float = 1e-8
    chunk_rows: int = 512
    rollback_on_regression: bool = True


@dataclass(frozen=True, slots=True)
class FactorizationConfig:
    implementation: str = "nanoquant_admm"
    compute_dtype: DType = DType.BFLOAT16
    solve_dtype: DType = DType.FLOAT32
    admm: ADMMConfig = field(default_factory=ADMMConfig)
    scale_fit: ScaleFitConfig = field(default_factory=ScaleFitConfig)


@dataclass(frozen=True, slots=True)
class ResidualProbeConfig:
    iterations: int = 80
    chunk_rows: int = 512


@dataclass(frozen=True, slots=True)
class OutlierConfig:
    selector: OutlierSelector = OutlierSelector.NONE
    fraction: float = 0.0
    storage_dtype: DType = DType.BFLOAT16
    layer_patterns: tuple[str, ...] = ("*",)
    charge_to_bit_budget: bool = True
    count_multiple: int = 1
    removed_column_importance: str = "zero"
    residual_probe: ResidualProbeConfig = field(default_factory=ResidualProbeConfig)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0


@dataclass(frozen=True, slots=True)
class TuningLoopConfig:
    enabled: bool = True
    epochs: int = 8
    batch_size: int = 8
    early_stop_relative_tolerance: float | None = None


@dataclass(frozen=True, slots=True)
class NonFactorizedTuningConfig:
    loop: TuningLoopConfig = field(default_factory=TuningLoopConfig)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(learning_rate=1e-4))
    epochs_by_layer_position: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class FactorizedLearningRates:
    binary: float = 1e-5
    scale: float = 1e-5
    outlier: float | None = None
    bias: float = 1e-5


@dataclass(frozen=True, slots=True)
class FactorizedTuningConfig:
    loop: TuningLoopConfig = field(default_factory=TuningLoopConfig)
    learning_rates: FactorizedLearningRates = field(default_factory=FactorizedLearningRates)
    skip_if_relative_loss_jump_below: float | None = None


@dataclass(frozen=True, slots=True)
class PostBlockRefitConfig:
    enabled: bool = False
    epochs: int = 0
    batch_size: int | None = None
    scale_learning_rate: float | None = None
    outlier_learning_rate: float | None = None
    bias_learning_rate: float | None = None


@dataclass(frozen=True, slots=True)
class BlockTuningConfig:
    layer_order: tuple[str, ...] = ()
    non_factorized: NonFactorizedTuningConfig = field(default_factory=NonFactorizedTuningConfig)
    factorized: FactorizedTuningConfig = field(default_factory=FactorizedTuningConfig)
    post_block_refit: PostBlockRefitConfig = field(default_factory=PostBlockRefitConfig)
    microbatch_size: int | None = None
    reset_seed_each_stage: bool = False
    restore_best_state: bool = True
    epoch_loss_mode: TuningEpochLossMode = TuningEpochLossMode.FULL_EVALUATION


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    enabled: bool = False
    loss: DistillationLoss = DistillationLoss.TOP_K
    epochs: int = 8
    batch_size: int = 1
    learning_rate: float = 1e-5
    temperature: float = 1.0
    top_k: int = 64
    vocabulary_chunk_size: int = 8192
    token_chunk_size: int = 128
    maximum_tokens_per_batch: int | None = 512
    gradient_checkpointing: bool = True
    weight_decay: float = 0.0
    optimizer_version: str = "legacy-optimi-adamw-v1"
    sampling_version: str = "legacy-python-device-rng-v1"
    teacher_targets_artifact: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceLimitsConfig:
    gpu_memory_gib: float | None = None
    cpu_memory_gib: float | None = None
    pinned_memory_gib: float = 1.0
    temporary_disk_gib: float | None = None
    workspace_memory_gib: float | None = None


@dataclass(frozen=True, slots=True)
class ActivationStorageConfig:
    kind: ActivationStoreKind = ActivationStoreKind.AUTO
    directory: str | None = None
    batch_size: int = 8
    prefetch_batches: int = 1
    gpu_cache: ActivationGpuCacheMode = ActivationGpuCacheMode.OFF
    gpu_reserve_gib: float = 1.0


@dataclass(frozen=True, slots=True)
class SourceStreamingConfig:
    prefetch_blocks: int = 0
    verify_tensor_hashes: bool = True
    prefer_memory_mapping: bool = True


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    enabled: bool = True
    commit_granularity: str = "layer"
    keep_attempt_artifacts: bool = False
    verify_on_resume: bool = True
    activation_retention: ActivationRetention = ActivationRetention.ROLLING


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    executor: ExecutorKind = ExecutorKind.AUTO
    compute_device: str = "cuda:0"
    block_forward_batch_size: int = 8
    resources: ResourceLimitsConfig = field(default_factory=ResourceLimitsConfig)
    activations: ActivationStorageConfig = field(default_factory=ActivationStorageConfig)
    source_streaming: SourceStreamingConfig = field(default_factory=SourceStreamingConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    on_cuda_oom: tuple[str, ...] = ("reduce_batch_size", "move_activations_down_one_tier", "fail")


@dataclass(frozen=True, slots=True)
class EmbeddingStorageConfig:
    bits: int = 8
    group_size: int | None = None


@dataclass(frozen=True, slots=True)
class PackingConfig:
    logical_format: str = "nanoquant-v1"
    backend_layouts: tuple[str, ...] = ("cuda-binary-v1",)
    shard_size_gib: float = 2.0
    align_blocks_to_shards: bool = True
    embeddings: EmbeddingStorageConfig = field(default_factory=EmbeddingStorageConfig)


@dataclass(frozen=True, slots=True)
class MetricGateConfig:
    metric: str
    maximum_regression: float | None = None
    minimum_improvement: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    default_tier: EvaluationTier = EvaluationTier.QUICK
    suites: tuple[str, ...] = ("artifact-parity-v1", "ppl-wikitext2-v1", "decode-v1")
    baseline_run: str | None = None
    gates: tuple[MetricGateConfig, ...] = ()
    few_shot: int = 0
    sample_limit: int | None = None
    inline_quality: bool = True
    inline_quality_samples: int = 1
    inline_quality_tokens: int = 8


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    console_level: str = "info"
    event_level: str = "info"
    record_resource_interval_seconds: float = 5.0
    record_admm_steps: bool = False
    record_weight_reconstruction_table: bool = True
    record_block_loss_snapshots: bool = True
    block_snapshot_samples: int = 4
    block_snapshot_tokens: int = 512
    loss_denominator_floor: float = 1e-8
    capture_cuda_trace: bool = False
    diagnostic_fixture_policy: str = "on_failure"


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    level: ProfilingLevel = ProfilingLevel.MACRO
    cuda_timing: bool = False
    cuda_sample_every: int = 16
    memory_counters: bool = False
    raw_samples_per_phase: int = 64
    trace_blocks: tuple[int, ...] = ()
    trace_layers: tuple[str, ...] = ()
    emit_span_events: bool = False


@dataclass(frozen=True, slots=True)
class OutputConfig:
    run_root: str = "runs"
    artifact_root: str = ".nanoquant/artifacts"
    temporary_root: str | None = None
    report_formats: tuple[str, ...] = ("markdown", "json")
    retain_temporary_artifacts: bool = False


@dataclass(frozen=True, slots=True)
class RunConfig:
    model: ModelConfig
    schema_version: int = 1
    intent: IntentConfig = field(default_factory=IntentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    allocation: RankAllocationConfig = field(default_factory=RankAllocationConfig)
    factorization: FactorizationConfig = field(default_factory=FactorizationConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    block_tuning: BlockTuningConfig = field(default_factory=BlockTuningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
