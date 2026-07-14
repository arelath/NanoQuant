# Hierarchical Configuration Reference

This document is the concrete configuration design that the rewrite should implement. The shorter [configuration and run-identity document](03-configuration-and-runs.md) explains lifecycle and hashing; this document defines the actual nested types, ownership of defaults, decoding, validation, and migration from current fields.

The code below is normative design pseudocode. Names may receive minor implementation corrections, but the hierarchy and ownership boundaries require an architecture decision to change.

## 1. Canonical rule

There is exactly one configuration object used by the application:

```python
RunConfig
```

YAML is decoded into `RunConfig`. The CLI produces a sparse set of path/value overrides and applies them to `RunConfig`. Python callers construct or load `RunConfig`. None of those entry points declares its own numerical defaults.

The following are deliberately **not** configuration:

- discovered model shapes;
- selected ranks by layer;
- retry bits already spent;
- active block/layer/attempt;
- peak memory and elapsed time;
- cache contents and artifact paths discovered at runtime;
- selected backend after capability probing;
- evaluation results.

Those values live in `RunPlan`, `RunState`, stage artifacts, or results.

## 2. Shared scalar types and enums

```python
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional


class StringEnum(str, Enum):
    """Python 3.9-compatible string enum base."""


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
```

Enums remove stringly typed comparisons scattered through the pipeline. Serialized values remain readable strings.

## 3. Intent, model, data, and reproducibility

```python
@dataclass(frozen=True, slots=True)
class IntentConfig:
    experiment_number: Optional[int] = None
    name: str = "unnamed-run"
    purpose: str = ""
    hypothesis: str = ""
    baseline_run: Optional[str] = None
    owner: Optional[str] = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelConfig:
    # Required user input; there is intentionally no default model.
    source: str
    revision: Optional[str] = None
    tokenizer_source: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    sequence_length: int = 2048
    load_dtype: DType = DType.BFLOAT16
    trust_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class DatasetSourceConfig:
    name: str
    revision: Optional[str] = None
    split: str = "train"
    subset: Optional[str] = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    sources: tuple[DatasetSourceConfig, ...] = (
        DatasetSourceConfig(name="wikitext2"),
    )
    formatting: str = "model_default"
    shuffle: bool = True
    selection_seed: int = 0
    cache_tokenized: bool = True


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    seed: int = 0
    deterministic: bool = True
    allow_nondeterministic_kernels: bool = False
```

The `model.source` is required instead of silently defaulting to a large remote model. Revisions may be omitted only in the input recipe; source resolution pins them in the resolved `RunConfig` before computation begins.

## 4. Calibration and reconstruction objective

```python
@dataclass(frozen=True, slots=True)
class HessianSamplingConfig:
    max_tokens_per_layer: int = 4096
    max_sequences: Optional[int] = None
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
    block_size: Optional[int] = None
    low_rank: Optional[int] = None
    sampling: HessianSamplingConfig = field(default_factory=HessianSamplingConfig)
    regularization: HessianRegularizationConfig = field(
        default_factory=HessianRegularizationConfig
    )


@dataclass(frozen=True, slots=True)
class CalibrationFallbackConfig:
    # Ordered, finite, and visible. An empty tuple means fail immediately.
    on_cuda_oom: tuple[str, ...] = (
        "cpu_offload",
        "forward_only",
        "fail",
    )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    method: CalibrationMethod = CalibrationMethod.ONLINE_FISHER
    sample_count: int = 128
    shrinkage: float = 0.4
    accumulation_dtype: DType = DType.FLOAT32
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    fallback: CalibrationFallbackConfig = field(
        default_factory=CalibrationFallbackConfig
    )
```

The old `hessian_whitening` boolean is replaced by `calibration.objective.kind`. Dense, block-diagonal, and low-rank approximations can now share one interface without accumulating more booleans.

## 5. Rank allocation, retry, and bit budget

```python
@dataclass(frozen=True, slots=True)
class RankBoundsConfig:
    multiple: int = 32
    floor_fraction_of_uniform: float = 0.80
    ceiling_fraction_of_uniform: float = 1.15
    edge_block_boost: float = 0.15


@dataclass(frozen=True, slots=True)
class RetryThresholdConfig:
    weighted_normalized_error: Optional[float] = 0.50
    raw_normalized_error: Optional[float] = None


@dataclass(frozen=True, slots=True)
class RankRetryConfig:
    enabled: bool = True
    thresholds: RetryThresholdConfig = field(
        default_factory=RetryThresholdConfig
    )
    rank_increase_fraction: float = 0.25
    maximum_attempts: int = 2
    extra_bit_budget_fraction: float = 0.02
    allow_above_allocator_cap: bool = False


@dataclass(frozen=True, slots=True)
class RankAllocationConfig:
    target_bpw: float = 1.0
    strategy: AllocationStrategy = AllocationStrategy.UNIFORM
    sensitivity_alpha: float = 0.5
    utility_profile_artifact: Optional[str] = None
    bounds: RankBoundsConfig = field(default_factory=RankBoundsConfig)
    retry: RankRetryConfig = field(default_factory=RankRetryConfig)
```

`None` means a retry threshold is disabled. The rewrite does not overload numeric zero with boolean semantics.

The output of allocation is not stored back into this object:

```python
@dataclass(frozen=True, slots=True)
class QuantizationPlan:
    config_hash: str
    calibration_artifact: str
    layers: tuple[LayerPlan, ...]
    total_planned_bits: int
```

Likewise, `retry_bits_spent` belongs to block/run progress state, not `RankRetryConfig`.

## 6. Factorization and scale fitting

```python
@dataclass(frozen=True, slots=True)
class ADMMConfig:
    outer_iterations: int = 800
    inner_iterations: int = 5
    regularization: float = 3e-2
    penalty_schedule: str = "cubic"
    convergence_check_interval: int = 100
    early_stop_tolerance: Optional[float] = None
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
```

`transpose_wide` preserves an exact replay mode for the legacy source's wide-matrix solve. It is disabled by
default because the native orientation is the policy validated by the full Gemma trajectory; changing it
invalidates resident factorization commits. Diagnostic verbosity is intentionally absent. ADMM iteration events
are controlled by observability settings, not by mathematical configuration.

## 7. Salient outliers

```python
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
    residual_probe: ResidualProbeConfig = field(
        default_factory=ResidualProbeConfig
    )
```

Setting `selector: none` is the canonical disabled state. Validation rejects a positive fraction with a disabled selector instead of guessing intent.

## 8. Block tuning and distillation

```python
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
    early_stop_relative_tolerance: Optional[float] = None


@dataclass(frozen=True, slots=True)
class NonFactorizedTuningConfig:
    loop: TuningLoopConfig = field(default_factory=TuningLoopConfig)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(learning_rate=1e-4)
    )
    # Empty means use loop.epochs for every position.
    epochs_by_layer_position: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class FactorizedLearningRates:
    binary: float = 1e-5
    scale: float = 1e-5
    outlier: Optional[float] = None
    bias: float = 1e-5


@dataclass(frozen=True, slots=True)
class FactorizedTuningConfig:
    loop: TuningLoopConfig = field(default_factory=TuningLoopConfig)
    learning_rates: FactorizedLearningRates = field(
        default_factory=FactorizedLearningRates
    )
    skip_if_relative_loss_jump_below: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PostBlockRefitConfig:
    enabled: bool = False
    epochs: int = 0
    batch_size: Optional[int] = None
    scale_learning_rate: Optional[float] = None
    outlier_learning_rate: Optional[float] = None
    bias_learning_rate: Optional[float] = None


@dataclass(frozen=True, slots=True)
class BlockTuningConfig:
    layer_order: tuple[str, ...] = ()  # empty means adapter default
    non_factorized: NonFactorizedTuningConfig = field(
        default_factory=NonFactorizedTuningConfig
    )
    factorized: FactorizedTuningConfig = field(
        default_factory=FactorizedTuningConfig
    )
    post_block_refit: PostBlockRefitConfig = field(
        default_factory=PostBlockRefitConfig
    )


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
    maximum_tokens_per_batch: Optional[int] = 512
    gradient_checkpointing: bool = True
    teacher_targets_artifact: Optional[str] = None
```

Separate nested tuning types prevent unrelated learning rates and batch sizes from becoming a flat list of similarly named fields.

## 9. Runtime, storage, and resume

```python
@dataclass(frozen=True, slots=True)
class ResourceLimitsConfig:
    gpu_memory_gib: Optional[float] = None
    cpu_memory_gib: Optional[float] = None
    pinned_memory_gib: float = 1.0
    temporary_disk_gib: Optional[float] = None
    workspace_memory_gib: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ActivationStorageConfig:
    kind: ActivationStoreKind = ActivationStoreKind.AUTO
    directory: Optional[str] = None
    batch_size: int = 8
    prefetch_batches: int = 1


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


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    executor: ExecutorKind = ExecutorKind.AUTO
    compute_device: str = "cuda:0"
    block_forward_batch_size: int = 8
    resources: ResourceLimitsConfig = field(
        default_factory=ResourceLimitsConfig
    )
    activations: ActivationStorageConfig = field(
        default_factory=ActivationStorageConfig
    )
    source_streaming: SourceStreamingConfig = field(
        default_factory=SourceStreamingConfig
    )
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    on_cuda_oom: tuple[str, ...] = (
        "reduce_batch_size",
        "move_activations_down_one_tier",
        "fail",
    )
```

The actual executor selected under `auto`, actual activation-store path, observed free memory, current batch fallback, and current checkpoint are recorded in `RunPlan`/`RunState`, not by mutating `RuntimeConfig`.

## 10. Packing, evaluation, observability, and output

```python
@dataclass(frozen=True, slots=True)
class EmbeddingStorageConfig:
    bits: int = 8
    group_size: Optional[int] = None


@dataclass(frozen=True, slots=True)
class PackingConfig:
    logical_format: str = "nanoquant-v1"
    backend_layouts: tuple[str, ...] = ("cuda-binary-v1",)
    shard_size_gib: float = 2.0
    align_blocks_to_shards: bool = True
    embeddings: EmbeddingStorageConfig = field(
        default_factory=EmbeddingStorageConfig
    )


@dataclass(frozen=True, slots=True)
class MetricGateConfig:
    metric: str
    maximum_regression: Optional[float] = None
    minimum_improvement: Optional[float] = None


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    default_tier: EvaluationTier = EvaluationTier.QUICK
    suites: tuple[str, ...] = (
        "artifact-parity-v1",
        "ppl-wikitext2-v1",
        "decode-v1",
    )
    baseline_run: Optional[str] = None
    gates: tuple[MetricGateConfig, ...] = ()
    few_shot: int = 0
    sample_limit: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    console_level: str = "info"
    event_level: str = "info"
    record_resource_interval_seconds: float = 5.0
    record_admm_steps: bool = False
    record_weight_reconstruction_table: bool = True
    record_block_loss_snapshots: bool = True
    loss_denominator_floor: float = 1e-8
    capture_cuda_trace: bool = False
    diagnostic_fixture_policy: str = "on_failure"


@dataclass(frozen=True, slots=True)
class OutputConfig:
    run_root: str = "runs"
    artifact_root: str = ".nanoquant/artifacts"
    temporary_root: Optional[str] = None
    report_formats: tuple[str, ...] = ("markdown", "json")
    retain_temporary_artifacts: bool = False
```

Weight-error CSV and Markdown paths disappear from algorithm config. Structured layer metrics are always artifacts/events; report selection belongs to `OutputConfig`.

## 11. Complete `RunConfig`

```python
@dataclass(frozen=True, slots=True)
class RunConfig:
    # model is the only required constructor argument.
    model: ModelConfig

    schema_version: int = 1
    intent: IntentConfig = field(default_factory=IntentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    allocation: RankAllocationConfig = field(
        default_factory=RankAllocationConfig
    )
    factorization: FactorizationConfig = field(
        default_factory=FactorizationConfig
    )
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    block_tuning: BlockTuningConfig = field(
        default_factory=BlockTuningConfig
    )
    distillation: DistillationConfig = field(
        default_factory=DistillationConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = field(
        default_factory=ObservabilityConfig
    )
    output: OutputConfig = field(default_factory=OutputConfig)
```

Minimal Python usage:

```python
config = RunConfig(
    model=ModelConfig(source="google/gemma-3-4b-it"),
)
```

Targeted Python customization uses immutable replacement:

```python
config = replace(
    config,
    calibration=replace(config.calibration, sample_count=256),
    allocation=replace(config.allocation, target_bpw=0.9),
    runtime=replace(config.runtime, executor=ExecutorKind.STREAMING),
)
```

Application services receive only a validated `RunConfig`:

```python
result = QuantizeApplication(components).run(config)
```

They do not accept parallel `**kwargs` that can override it.

## 12. Equivalent YAML

The same object can be represented as:

```yaml
schema_version: 1

intent:
  experiment_number: 19
  name: gemma-3-4b-phase1
  purpose: Establish the phase-1 diagonal-objective baseline.
  hypothesis: Residual outliers improve difficult attention projections at equal BPW.
  baseline_run: run_01J_BASELINE
  tags: [gemma3, phase1, residual-outliers]

model:
  source: google/gemma-3-4b-it
  revision: 0123456789abcdef
  sequence_length: 2048
  load_dtype: bfloat16

dataset:
  sources:
    - name: ultrachat_200k
      revision: abcdef0123456789
      weight: 0.75
    - name: wikitext2
      revision: 9876543210fedcba
      weight: 0.25
  selection_seed: 0

reproducibility:
  seed: 0
  deterministic: true

calibration:
  method: online_fisher
  sample_count: 256
  shrinkage: 0.6
  objective:
    kind: diagonal

allocation:
  target_bpw: 1.0
  strategy: sensitivity
  sensitivity_alpha: 0.5
  bounds:
    multiple: 32
    floor_fraction_of_uniform: 0.9
    ceiling_fraction_of_uniform: 1.1
    edge_block_boost: 0.15
  retry:
    enabled: true
    thresholds:
      weighted_normalized_error: 0.5
      raw_normalized_error: 0.5
    rank_increase_fraction: 0.25
    maximum_attempts: 2
    extra_bit_budget_fraction: 0.02
    allow_above_allocator_cap: true

factorization:
  implementation: nanoquant_admm
  admm:
    outer_iterations: 800
    inner_iterations: 5
    regularization: 0.03
    penalty_schedule: cubic
    transpose_wide: false
  scale_fit:
    enabled: true
    alternating_passes: 2
    chunk_rows: 512

outliers:
  selector: residual
  fraction: 0.001
  storage_dtype: bfloat16
  layer_patterns: ["*"]
  charge_to_bit_budget: true
  residual_probe:
    iterations: 80
    chunk_rows: 512

block_tuning:
  layer_order:
    - mlp.gate_proj
    - mlp.up_proj
    - mlp.down_proj
    - self_attn.v_proj
    - self_attn.o_proj
    - self_attn.q_proj
    - self_attn.k_proj
  non_factorized:
    loop:
      enabled: true
      epochs: 8
      batch_size: 8
      early_stop_relative_tolerance: 0.001
    optimizer:
      name: adamw
      learning_rate: 0.0001
    epochs_by_layer_position: [8, 4, 3, 2, 2, 2, 2]
  factorized:
    loop:
      enabled: true
      epochs: 8
      batch_size: 8
    learning_rates:
      binary: 0.00001
      scale: 0.00001
      bias: 0.00001
  post_block_refit:
    enabled: false

distillation:
  enabled: false
  loss: top_k
  top_k: 64

runtime:
  executor: auto
  compute_device: cuda:0
  block_forward_batch_size: 8
  resources:
    gpu_memory_gib: 44
    cpu_memory_gib: 64
    pinned_memory_gib: 1
    temporary_disk_gib: 500
  activations:
    kind: auto
    batch_size: 8
    prefetch_batches: 1
  checkpoints:
    enabled: true
    commit_granularity: layer

packing:
  logical_format: nanoquant-v1
  backend_layouts: [cuda-binary-v1]
  embeddings:
    bits: 8

evaluation:
  default_tier: quick
  suites:
    - artifact-parity-v1
    - ppl-wikitext2-v1
    - decode-v1

observability:
  console_level: info
  event_level: info
  record_admm_steps: false
  record_weight_reconstruction_table: true
  record_block_loss_snapshots: true
  loss_denominator_floor: 1.0e-8
  diagnostic_fixture_policy: on_failure

output:
  run_root: runs
  artifact_root: .nanoquant/artifacts
  report_formats: [markdown, json]
```

Omitted values come from the dataclass fields above. The resolved recipe writes every value, including defaults.

## 13. One default source

Defaults exist only on the canonical nested dataclasses. Entry points behave as follows:

### YAML

```python
raw_mapping = yaml.safe_load(path.read_text())
config = decode_dataclass(RunConfig, raw_mapping)
```

`decode_dataclass` recursively reads field types/defaults from the canonical schema. It rejects unknown keys with a path and nearest-name suggestion.

### CLI

Common convenience flags default to `None`, never to algorithm values:

```text
nanoquant quantize recipe.yaml \
  --model google/gemma-3-4b-it \
  --set calibration.sample_count=256 \
  --set factorization.admm.outer_iterations=1000
```

The CLI produces this sparse patch:

```python
{
    "model.source": "google/gemma-3-4b-it",
    "calibration.sample_count": 256,
    "factorization.admm.outer_iterations": 1000,
}
```

It applies the patch through schema-aware immutable replacement. It does not construct a second CLI configuration dataclass.

### Python

Python passes `RunConfig` directly or calls the same `load_recipe`/`apply_overrides` functions as the CLI.

### Generated documentation

CLI help and a configuration reference table are generated from field metadata on these types, so default and help text cannot drift.

## 14. Resolution without a second config model

Input and resolved recipes use the same `RunConfig` type:

```python
input_config = load_recipe("recipe.yaml")
validate(input_config, phase="pre_resolution")

resolution = source_resolver.resolve(input_config.model)
resolved_config = replace(
    input_config,
    model=replace(
        input_config.model,
        revision=resolution.model_revision,
        tokenizer_source=resolution.tokenizer_source,
        tokenizer_revision=resolution.tokenizer_revision,
    ),
)
resolved_config = executor_resolver.replace_auto_choices(resolved_config, host)
validate(resolved_config, phase="resolved")
```

`SourceResolution`, `HostInventory`, and `RunPlan` are separate result types, not alternate bags of configuration defaults.

The run stores both:

```text
recipe.input.yaml       # what the user supplied, including omissions
recipe.resolved.yaml    # complete RunConfig with pinned inputs and selected auto choices
plan.json               # derived shapes, ranks, bytes, executor actions, estimates
```

## 15. Validation

Local invariants can be checked while decoding, but cross-field validation is centralized and returns all discoverable problems at once:

```python
@dataclass(frozen=True, slots=True)
class ConfigProblem:
    code: str
    path: str
    message: str
    remediation: Optional[str] = None


def validate(config: RunConfig, phase: str) -> tuple[ConfigProblem, ...]:
    ...
```

Examples:

```text
NQ-CFG-001 model.sequence_length must be positive
NQ-CFG-010 dataset.sources weights must sum to a positive value
NQ-CFG-020 calibration.sample_count must be positive unless method=none
NQ-CFG-021 objective.low_rank is required when kind=low_rank_diagonal
NQ-CFG-022 objective.block_size is invalid when kind=diagonal
NQ-CFG-030 allocation.target_bpw cannot pay mandatory representation overhead
NQ-CFG-031 retry.maximum_attempts must be >= 1 when retry.enabled=true
NQ-CFG-040 outliers.fraction must be 0 when selector=none
NQ-CFG-041 int8 outlier training requires a supported trainable master policy
NQ-CFG-050 post_block_refit.epochs must be > 0 when enabled=true
NQ-CFG-060 full_kl distillation is incompatible with the selected single-GPU streaming plan
NQ-CFG-070 mmap activation storage requires a directory or temporary_root
NQ-CFG-071 planned temporary bytes exceed runtime.resources.temporary_disk_gib
NQ-CFG-080 requested packed layout does not support the planned model shapes
```

Validation phases:

- `pre_resolution`: syntax and context-free invariants; no downloads or weight allocation;
- `resolved`: pinned sources, no unresolved `auto` values needed for execution;
- `planned`: shape-, budget-, backend-, and host-aware constraints.

No computation stage starts until its required validation phase passes.

## 16. Runtime state is separate

```python
@dataclass(frozen=True, slots=True)
class ProgressCursor:
    stage: str
    block: Optional[int] = None
    layer: Optional[str] = None
    attempt: Optional[int] = None


@dataclass(frozen=True, slots=True)
class BudgetState:
    planned_binary_bits: int = 0
    accepted_extra_retry_bits: int = 0


@dataclass(frozen=True, slots=True)
class RunState:
    run_id: str
    config_hash: str
    status: str
    progress: ProgressCursor
    budget: BudgetState
    committed_artifacts: tuple[str, ...]
```

State updates create journal/commit records. The application never performs operations such as:

```python
# Prohibited
config["_rank_retry_extra_bits"] += extra
config["_weight_error_rows"].append(row)
config["selected_executor"] = "streaming"
```

## 17. Current-field migration map

The migration tool maps old flat names to canonical paths. Representative current fields are listed below; fields that become artifacts or diagnostics are explicitly identified.

| Current field | New canonical path or disposition |
| --- | --- |
| `model_id` | `model.source` |
| `seqlen` | `model.sequence_length` |
| `device_map` | `runtime.executor` plus `runtime.compute_device`; no longer a model-domain field |
| `bits` | `allocation.target_bpw` |
| `seed` | `reproducibility.seed` and dataset selection seed where intentionally shared |
| `num_calib_samples` | `calibration.sample_count` |
| `calib_dataset` | `dataset.sources` |
| `calib_shrinkage` | `calibration.shrinkage` |
| `calib_strategy` | `calibration.method` |
| `calib_oom_fallback` | `calibration.fallback.on_cuda_oom` |
| `hessian_whitening` | `calibration.objective.kind` |
| `hessian_max_tokens` | `calibration.objective.sampling.max_tokens_per_layer` |
| `hessian_max_sequences` | `calibration.objective.sampling.max_sequences` |
| `hessian_batch_size` | `calibration.objective.sampling.batch_size` |
| `hessian_reuse_siblings` | `calibration.objective.sampling.reuse_sibling_inputs` |
| `hessian_damp_percent` | `calibration.objective.regularization.diagonal_damp_fraction` |
| `hessian_shrinkage` | `calibration.objective.regularization.identity_shrinkage` |
| `hessian_diagonal_blend` | `calibration.objective.regularization.diagonal_blend` |
| `block_forward_batch_size` | `runtime.block_forward_batch_size` |
| `block_activation_device` | `runtime.activations.kind` |
| `pin_cpu_activations` | resolved from `runtime.activations.kind`; not an independent boolean |
| `pin_cpu_activation_max_gib` | `runtime.resources.pinned_memory_gib` |
| `loss_pct_floor` | `observability.loss_denominator_floor` |
| `eval_block_ppl` | an explicit block-level evaluator suite/promotion policy; not a quantization boolean |
| `quant_layer_order` | `block_tuning.layer_order` |
| `cleanup_per_layer` | removed; resource scopes/executor policy own cleanup |
| `rank_allocation_strategy` | `allocation.strategy` |
| `rank_sensitivity_alpha` | `allocation.sensitivity_alpha` |
| `rank_edge_boost` | `allocation.bounds.edge_block_boost` |
| `rank_floor_frac` | `allocation.bounds.floor_fraction_of_uniform` |
| `rank_ceil_frac` | `allocation.bounds.ceiling_fraction_of_uniform` |
| `rank_retry_norm_error_threshold` | `allocation.retry.thresholds.weighted_normalized_error` |
| `rank_retry_raw_norm_error_threshold` | `allocation.retry.thresholds.raw_normalized_error` |
| `rank_retry_allow_above_cap` | `allocation.retry.allow_above_allocator_cap` |
| `rank_retry_bump_frac` | `allocation.retry.rank_increase_fraction` |
| `rank_retry_max_attempts` | `allocation.retry.maximum_attempts` |
| `rank_retry_bits_budget_frac` | `allocation.retry.extra_bit_budget_fraction` |
| `weight_error_log_path` | removed; canonical layer-result events/artifacts are always produced |
| `weight_error_table_path` | `output.report_formats`; report path is derived from run ID |
| `rank_utility_profile_path` | `allocation.utility_profile_artifact` |
| `rank_utility_log_path` | removed; rank utility is a standard layer metric artifact |
| `outlier_frac` | `outliers.fraction` |
| `outlier_dtype` | `outliers.storage_dtype` |
| `outlier_layers` | `outliers.layer_patterns` |
| `outlier_metric` | `outliers.selector` |
| `outlier_budget_compensate` | `outliers.charge_to_bit_budget` |
| `outlier_count_multiple` | `outliers.count_multiple` |
| `outlier_i_norm_mode` | `outliers.removed_column_importance` |
| `outlier_residual_probe_iters` | `outliers.residual_probe.iterations` |
| `outlier_residual_chunk_rows` | `outliers.residual_probe.chunk_rows` |
| `embed_tokens_weight_bits` | `packing.embeddings.bits` |
| `tune_nonfact` | `block_tuning.non_factorized.loop.enabled` |
| `nonfact_lr` | `block_tuning.non_factorized.optimizer.learning_rate` |
| `nonfact_batch_size` | `block_tuning.non_factorized.loop.batch_size` |
| `nonfact_epochs` | `block_tuning.non_factorized.loop.epochs` |
| `nonfact_early_stop_rel_tol` | `block_tuning.non_factorized.loop.early_stop_relative_tolerance` |
| `nonfact_epoch_schedule` | `block_tuning.non_factorized.epochs_by_layer_position` |
| `admm_type` | `factorization.implementation` |
| `admm_outer_iters` | `factorization.admm.outer_iterations` |
| `admm_inner_iters` | `factorization.admm.inner_iterations` |
| `admm_reg` | `factorization.admm.regularization` |
| `admm_penalty_scheduler` | `factorization.admm.penalty_schedule` |
| `admm_print_steps` | `observability.record_admm_steps` |
| `ls_scale_fit` | `factorization.scale_fit.enabled` |
| `ls_scale_fit_iters` | `factorization.scale_fit.alternating_passes` |
| `ls_scale_fit_eps` | `factorization.scale_fit.epsilon` |
| `ls_scale_fit_chunk_rows` | `factorization.scale_fit.chunk_rows` |
| `tune_fact` | `block_tuning.factorized.loop.enabled` |
| `fact_binary_lr` | `block_tuning.factorized.learning_rates.binary` |
| `fact_scale_lr` | `block_tuning.factorized.learning_rates.scale` |
| `fact_outlier_lr` | `block_tuning.factorized.learning_rates.outlier` |
| `fact_bias_lr` | `block_tuning.factorized.learning_rates.bias` |
| `fact_batch_size` | `block_tuning.factorized.loop.batch_size` |
| `fact_epochs` | `block_tuning.factorized.loop.epochs` |
| `fact_early_stop_rel_tol` | `block_tuning.factorized.loop.early_stop_relative_tolerance` |
| `fact_skip_jump_frac` | `block_tuning.factorized.skip_if_relative_loss_jump_below` |
| `tune_eval_summaries` | removed as an opt-in; canonical layer/block snapshots are always recorded when `observability.record_block_loss_snapshots=true` (the default) |
| `post_block_scale_epochs` | `block_tuning.post_block_refit.enabled/epochs` |
| `post_block_scale_lr` | `block_tuning.post_block_refit.scale_learning_rate` |
| `post_block_outlier_lr` | `block_tuning.post_block_refit.outlier_learning_rate` |
| `post_block_bias_lr` | `block_tuning.post_block_refit.bias_learning_rate` |
| `post_block_scale_batch_size` | `block_tuning.post_block_refit.batch_size` |
| `tune_model` | `distillation.enabled` |
| `model_kd_lr` | `distillation.learning_rate` |
| `model_kd_batch_size` | `distillation.batch_size` |
| `model_kd_epochs` | `distillation.epochs` |
| `model_kd_gradient_checkpointing` | `distillation.gradient_checkpointing` |
| `model_kd_loss` | `distillation.loss` |
| `model_kd_temperature` | `distillation.temperature` |
| `model_kd_topk` | `distillation.top_k` |
| `model_kd_vocab_chunk_size` | `distillation.vocabulary_chunk_size` |
| `model_kd_token_chunk_size` | `distillation.token_chunk_size` |
| `model_kd_max_tokens_per_batch` | `distillation.maximum_tokens_per_batch` |
| `ppl_task`, `zeroshot_task` | named `evaluation.suites` entries |
| evaluation `batch_size` | evaluator suite configuration, not quantization config |
| `num_fewshot` | `evaluation.few_shot` |
| `limit` | `evaluation.sample_limit` |
| `_rank_retry_total_bits` | `QuantizationPlan.total_planned_bits` |
| `_rank_retry_extra_bits` | `RunState.budget.accepted_extra_retry_bits` |
| `_weight_error_rows` | `LayerResult` artifacts and structured events |

The migration command reports every mapped, removed, defaulted, or ambiguous field. It never silently drops an unknown legacy field.

## 18. CLI and API signatures

The application boundary stays small:

```python
def load_recipe(path: str) -> RunConfig: ...
def apply_overrides(config: RunConfig, values: Mapping[str, Any]) -> RunConfig: ...
def resolve_config(config: RunConfig, host: HostInventory) -> RunConfig: ...
def validate_config(config: RunConfig, phase: str) -> tuple[ConfigProblem, ...]: ...
def plan_run(config: RunConfig) -> RunPlan: ...
def quantize(config: RunConfig, *, resume_run_id: Optional[str] = None) -> RunResult: ...
```

There is no alternate `NanoQuantConfig`, `NanoQuantConfigDataclass`, `QuantArguments`, or `TuneArguments` in the rewrite.

## 19. Tests required for the configuration model

- every default appears exactly once in schema introspection;
- minimal YAML and minimal Python construction resolve identically;
- YAML, CLI override, and Python `replace` produce byte-identical canonical serialization;
- unknown and misspelled fields fail with the full path;
- tuples/enums/nested dataclasses round trip through YAML and JSON;
- frozen objects cannot be mutated by runtime code;
- cross-field validation returns all independent problems;
- input and resolved recipes use the same `RunConfig` decoder;
- stage semantic hashes ignore intent/report formatting and react to numerical changes;
- migration covers every legacy field and fails on unmapped input;
- generated CLI help and config documentation match field defaults;
- no module outside `config` constructs its own defaults for canonical fields.
