# Domain Objects and Stage Contracts

This document makes the data moving through the rewrite explicit. The [architecture](02-architecture.md) defines component boundaries; this reference defines the stable objects crossing those boundaries.

The current `factorize_and_replace` path combines source-weight access, outlier policy, Hessian handling, factorization attempts, retry budgeting, scale fitting, module replacement, logging, and report-file updates. The rewrite replaces that implicit bundle with small services connected by typed requests and results.

## 1. Contract rules

1. Every stage receives one typed request and returns one typed result.
2. A stage receives only the configuration subtree it needs.
3. Persisted stage results reference immutable tensor artifacts; they do not embed arbitrary Python objects.
4. Large tensors may be materialized inside a stage through a lease, but stage boundaries use `TensorRef`.
5. Input tensors are read-only. Mutation requires a stage-owned copy or lease explicitly marked writable.
6. Model-family objects do not cross domain boundaries; canonical block/layer/tensor identities do.
7. Runtime progress, logging rows, and cache state are not fields on mathematical objects.
8. Every public object has a schema version, validation function, and canonical serializer.
9. Optional values use `None` or an explicit variant, not magic tensor shapes or sentinel numbers.
10. Results include enough lineage to replay the operation without loading the complete model.

## 2. Identities and tensor references

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


@dataclass(frozen=True, slots=True, order=True)
class BlockId:
    index: int


@dataclass(frozen=True, slots=True, order=True)
class LayerId:
    block: BlockId
    path: str


@dataclass(frozen=True, slots=True, order=True)
class TensorId:
    layer: Optional[LayerId]
    name: str


@dataclass(frozen=True, slots=True)
class ComponentRef:
    name: str
    version: str


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_type: str
    artifact_id: str
    schema_version: int


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    layout: str = "contiguous"
    device_requirement: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TensorRef:
    artifact: ArtifactRef
    key: str
    spec: TensorSpec
    content_hash: str
```

`LayerId` is canonical and adapter-independent. An adapter translates it to a source checkpoint key or model submodule path. Filesystem paths are not identities.

`TensorRef` permits a stage result to remain small and inspectable. The executor/artifact store turns it into a scoped tensor lease:

```python
with tensor_store.read(tensor_ref, device="cuda:0") as weight:
    result = factorizer.factorize(weight, ...)
```

The lease prevents accidental retention of multi-gigabyte tensors across stage boundaries.

## 3. Source and dataset objects

```python
@dataclass(frozen=True, slots=True)
class ModelIdentity:
    source: str
    revision: str
    config_hash: str
    tokenizer_source: str
    tokenizer_revision: str
    adapter: ComponentRef


@dataclass(frozen=True, slots=True)
class SourceTensor:
    id: TensorId
    source_key: str
    shard: str
    spec: TensorSpec
    content_hash: str


@dataclass(frozen=True, slots=True)
class LayerInventory:
    layer: LayerId
    weight: SourceTensor
    bias: Optional[SourceTensor]
    in_features: int
    out_features: int


@dataclass(frozen=True, slots=True)
class BlockInventory:
    block: BlockId
    source_tensors: tuple[SourceTensor, ...]
    quantizable_layers: tuple[LayerInventory, ...]


@dataclass(frozen=True, slots=True)
class ModelInventory:
    schema_version: int
    model: ModelIdentity
    blocks: tuple[BlockInventory, ...]
    shared_tensors: tuple[SourceTensor, ...]
    total_source_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    fingerprint: str
    sources: tuple[str, ...]
    revisions: tuple[str, ...]
    tokenizer_hash: str
    formatting_version: str


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    schema_version: int
    identity: DatasetIdentity
    token_batches: ArtifactRef
    sample_count: int
    valid_token_count: int
    selection_seed: int
```

`ModelInventory` can be built from checkpoint metadata without loading all weights. It is an input to planning and large-model resource estimation.

## 4. Calibration objects

Calibration statistics are represented per layer rather than as parallel untyped dictionaries:

```python
@dataclass(frozen=True, slots=True)
class StatisticSummary:
    minimum: float
    maximum: float
    mean: float
    zero_fraction: float
    non_finite_count: int


@dataclass(frozen=True, slots=True)
class CovarianceRef:
    representation: str  # diagonal, block_diagonal, low_rank_diagonal, dense
    diagonal: TensorRef
    blocks: Optional[TensorRef] = None
    low_rank_factors: Optional[TensorRef] = None
    dense: Optional[TensorRef] = None
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class LayerCalibrationStats:
    layer: LayerId
    input_importance: TensorRef
    output_importance: TensorRef
    input_covariance: Optional[CovarianceRef]
    input_summary: StatisticSummary
    output_summary: StatisticSummary
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    schema_version: int
    producer: ComponentRef
    model: ModelIdentity
    dataset: DatasetIdentity
    method: str
    accumulation_dtype: str
    layers: tuple[LayerCalibrationStats, ...]
    total_samples: int
    total_tokens: int
```

In-memory calibration code may accumulate tensors in a typed `MutableLayerAccumulator`, but that type is private to the stage and never becomes a persisted result.

Invariants:

- exactly one `LayerCalibrationStats` per requested canonical layer;
- vector lengths match model inventory dimensions;
- covariance input dimension matches `in_features`;
- all required statistics are finite after the declared sanitization policy;
- dataset and source identities are pinned;
- summaries are recomputable from tensor artifacts within tolerance.

## 5. Objective objects

The factorizer should not receive unrelated calibration state. An objective builder converts relevant statistics into an executable objective specification:

```python
@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    schema_version: int
    layer: LayerId
    kind: str
    input_importance: TensorRef
    output_importance: TensorRef
    covariance: Optional[CovarianceRef]
    damping: float
    normalization: str
    target_weighted_norm_squared: Optional[float]
    source_calibration: ArtifactRef
```

At compute time, an infrastructure adapter materializes an implementation of:

```python
class ReconstructionObjective(Protocol):
    def weighted_error(self, target: Tensor, prediction: Tensor) -> Tensor: ...
    def normalized_error(self, target: Tensor, prediction: Tensor) -> Tensor: ...
    def transform_for_factorizer(self, weight: Tensor) -> ObjectiveWorkspace: ...
```

The persisted `ObjectiveSpec` is portable; `ReconstructionObjective` is a scoped executable view.

## 6. Planning objects

```python
@dataclass(frozen=True, slots=True)
class BitCost:
    binary_factor_bits: int
    scale_bits: int
    outlier_value_bits: int
    outlier_index_bits: int
    padding_bits: int

    @property
    def total(self) -> int:
        return (
            self.binary_factor_bits
            + self.scale_bits
            + self.outlier_value_bits
            + self.outlier_index_bits
            + self.padding_bits
        )


@dataclass(frozen=True, slots=True)
class OutlierPlan:
    selector: str
    count: int
    storage_dtype: str
    charge_to_budget: bool


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    maximum_attempts: int
    rank_increase_fraction: float
    weighted_error_threshold: Optional[float]
    raw_error_threshold: Optional[float]
    hard_rank_cap: int
    extra_bit_budget: int


@dataclass(frozen=True, slots=True)
class LayerPlan:
    schema_version: int
    layer: LayerId
    source_weight: SourceTensor
    rank: int
    rank_multiple: int
    allocator_cap: int
    objective: ObjectiveSpec
    outliers: OutlierPlan
    retry: RetryPolicy
    estimated_cost: BitCost


@dataclass(frozen=True, slots=True)
class BlockPlan:
    block: BlockId
    layer_order: tuple[LayerId, ...]
    layers: tuple[LayerPlan, ...]
    estimated_workspace_bytes: int


@dataclass(frozen=True, slots=True)
class QuantizationPlan:
    schema_version: int
    producer: ComponentRef
    model: ModelIdentity
    calibration: ArtifactRef
    blocks: tuple[BlockPlan, ...]
    target_bpw: float
    planned_cost: BitCost
```

Ranks are derived outputs. They appear here, never as runtime mutations of configuration.

## 7. Outlier-selection objects

```python
@dataclass(frozen=True, slots=True)
class OutlierSelectionRequest:
    layer: LayerId
    source_weight: TensorRef
    objective: ObjectiveSpec
    plan: OutlierPlan
    probe_rank: int
    logical_seed: int


@dataclass(frozen=True, slots=True)
class OutlierSelectionResult:
    schema_version: int
    producer: ComponentRef
    layer: LayerId
    indices: TensorRef
    values: TensorRef
    scales: Optional[TensorRef]
    residual_weight: TensorRef
    selected_score_summary: StatisticSummary
    bit_cost: BitCost
```

The selection service does not replace a model module or decide rank retry. It produces a residual target and side-path state for a factorization request.

## 8. Factorization objects

```python
@dataclass(frozen=True, slots=True)
class ScaleState:
    pre: TensorRef
    mid: Optional[TensorRef]
    post: TensorRef


@dataclass(frozen=True, slots=True)
class TrainableFactors:
    left_latent: TensorRef
    right_latent: TensorRef
    left_binary: TensorRef
    right_binary: TensorRef
    scales: ScaleState


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    objective_mode: str
    target_weighted_norm_squared: float
    latent_weighted_error: Optional[float]
    latent_weighted_normalized_error: Optional[float]
    unwhitened_weighted_error: float
    unwhitened_weighted_normalized_error: float
    export_weighted_error: float
    export_weighted_normalized_error: float
    raw_error: float
    raw_normalized_error: float


@dataclass(frozen=True, slots=True)
class ConvergenceMetrics:
    iterations_completed: int
    stopped_early: bool
    final_primal_residual: Optional[float]
    final_dual_residual: Optional[float]
    trace: Optional[ArtifactRef]


@dataclass(frozen=True, slots=True)
class FactorizationRequest:
    schema_version: int
    layer: LayerId
    source_weight: TensorRef
    residual_weight: TensorRef
    objective: ObjectiveSpec
    rank: int
    logical_seed: int
    factorizer_config_hash: str


@dataclass(frozen=True, slots=True)
class FactorizationResult:
    schema_version: int
    producer: ComponentRef
    layer: LayerId
    rank: int
    factors: TrainableFactors
    metrics: ReconstructionMetrics
    convergence: ConvergenceMetrics
    wall_seconds: float
    peak_workspace_bytes: int
```

This is the direct concrete form of the outline's `FactorizationResult`. It separates factor tensors, scales, outlier state, reconstruction quality, convergence, and resources instead of returning a dictionary/namespace with optional keys inferred by callers.

## 9. Scale-fitting and retry objects

```python
@dataclass(frozen=True, slots=True)
class ScaleFitRequest:
    layer: LayerId
    target_weight: TensorRef
    factors: TrainableFactors
    objective: ObjectiveSpec
    protected_columns: Optional[TensorRef]


@dataclass(frozen=True, slots=True)
class ScaleFitResult:
    scales: ScaleState
    before: ReconstructionMetrics
    after: ReconstructionMetrics
    accepted: bool
    rollback_reason: Optional[str]


@dataclass(frozen=True, slots=True)
class AttemptSummary:
    attempt: int
    rank: int
    result: ArtifactRef
    weighted_error: float
    raw_error: float
    bit_cost: BitCost
    retry_score: float
    accepted: bool
    decision_reason: str


@dataclass(frozen=True, slots=True)
class RetryDecisionRequest:
    layer_plan: LayerPlan
    attempts: tuple[AttemptSummary, ...]
    global_extra_bits_spent: int


@dataclass(frozen=True, slots=True)
class RetryDecision:
    action: str  # accept_best, retry
    accepted_attempt: Optional[int]
    next_rank: Optional[int]
    projected_extra_bits: int
    reason: str
```

The retry policy is a pure decision. The application loop executes it and updates `BudgetState` only after a layer commit succeeds.

## 10. Tuning and frozen layer objects

```python
@dataclass(frozen=True, slots=True)
class LossMetrics:
    loss: float
    valid_elements: int
    objective: str


@dataclass(frozen=True, slots=True)
class TuningMetrics:
    before: LossMetrics
    best: LossMetrics
    final: LossMetrics
    epochs_completed: int
    best_epoch: int
    stopped_early: bool
    trace: Optional[ArtifactRef]


@dataclass(frozen=True, slots=True)
class FrozenOutlierState:
    indices: TensorRef
    values: TensorRef
    scales: Optional[TensorRef]


@dataclass(frozen=True, slots=True)
class FrozenNanoQuantState:
    layer: LayerId
    rank: int
    left_binary: TensorRef
    right_binary: TensorRef
    scales: ScaleState
    outliers: Optional[FrozenOutlierState]
    bias: Optional[TensorRef]
    logical_format: str


@dataclass(frozen=True, slots=True)
class LayerResult:
    schema_version: int
    layer: LayerId
    plan: LayerPlan
    attempts: tuple[AttemptSummary, ...]
    accepted_attempt: int
    factorization: ArtifactRef
    scale_fit: Optional[ScaleFitResult]
    tuning: Optional[TuningMetrics]
    frozen_state: FrozenNanoQuantState
    final_reconstruction: ReconstructionMetrics
    actual_bit_cost: BitCost
    extra_retry_bits: int
    warnings: tuple[str, ...]
```

`FrozenNanoQuantState` contains no optimizer flags, mutable module registrations, or runtime packed caches.

## 11. Block objects

```python
@dataclass(frozen=True, slots=True)
class ActivationStreamRef:
    artifact: ArtifactRef
    shape: tuple[int, ...]
    dtype: str
    sample_count: int
    sequence_length: int


@dataclass(frozen=True, slots=True)
class BlockQuantizationRequest:
    schema_version: int
    model: ModelIdentity
    block_plan: BlockPlan
    source_block: ArtifactRef
    teacher_inputs: ActivationStreamRef
    compressed_inputs: ActivationStreamRef
    calibration: ArtifactRef
    block_tuning_config_hash: str
    budget_state: ArtifactRef


@dataclass(frozen=True, slots=True)
class LossComparison:
    baseline_name: str
    candidate_name: str
    baseline_loss: float
    candidate_loss: float
    absolute_delta: float
    relative_delta: Optional[float]
    denominator_floor: float


@dataclass(frozen=True, slots=True)
class BlockLossMetrics:
    source_reference: float
    block_entry_pre_quantization: float
    after_each_layer: tuple[tuple[LayerId, float], ...]
    after_post_block_refit: Optional[float]
    final_frozen_pre_kd: float
    final_vs_block_entry: LossComparison
    final_vs_source_reference: LossComparison


@dataclass(frozen=True, slots=True)
class GlobalTuningBlockMetrics:
    block: BlockId
    final_frozen_pre_kd: float
    final_post_kd: float
    post_kd_vs_pre_kd: LossComparison


@dataclass(frozen=True, slots=True)
class FrozenBlockState:
    block: BlockId
    quantized_layers: tuple[FrozenNanoQuantState, ...]
    passthrough_tensors: tuple[TensorRef, ...]
    auxiliary_parameters: tuple[tuple[str, TensorRef], ...]


@dataclass(frozen=True, slots=True)
class BlockResult:
    schema_version: int
    block: BlockId
    layers: tuple[LayerResult, ...]
    frozen_state: FrozenBlockState
    losses: BlockLossMetrics
    teacher_outputs: ActivationStreamRef
    compressed_outputs: ActivationStreamRef
    extra_bits_used: int
    wall_seconds: float
    peak_gpu_bytes: int
    peak_host_bytes: int
    warnings: tuple[str, ...]
```

This is the concrete form of the outline's `BlockResult`. It includes both next-block activation generations because they are required to resume sequential reconstruction without replaying prior blocks.

## 12. Model, packing, and evaluation objects

```python
@dataclass(frozen=True, slots=True)
class FrozenModelResult:
    schema_version: int
    model: ModelIdentity
    plan: ArtifactRef
    blocks: tuple[ArtifactRef, ...]
    shared_tensors: tuple[TensorRef, ...]
    global_tuning: Optional[ArtifactRef]
    actual_total_bits: int
    effective_bpw: float


@dataclass(frozen=True, slots=True)
class PackedLayoutRef:
    backend: ComponentRef
    layout_version: str
    tensors: ArtifactRef
    required_capabilities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PackedModelResult:
    schema_version: int
    logical_model: ArtifactRef
    layouts: tuple[PackedLayoutRef, ...]
    artifact_bytes: int
    artifact_bpw: float
    validation: ArtifactRef


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float
    unit: str
    direction: str
    sample_count: Optional[int]
    confidence_interval: Optional[tuple[float, float]]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    schema_version: int
    evaluation_id: str
    tier: str
    model_artifact: ArtifactRef
    baseline_artifact: Optional[ArtifactRef]
    metrics: tuple[MetricValue, ...]
    resource_usage: ArtifactRef
    status: str
    warnings: tuple[str, ...]
```

## 13. Stage interface

```python
RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class Stage(Generic[RequestT, ResultT], Protocol):
    name: str
    version: str

    def semantic_key(self, request: RequestT) -> str: ...
    def estimate(self, request: RequestT, host: HostInventory) -> ResourceEstimate: ...
    def execute(self, request: RequestT, context: StageContext) -> ResultT: ...
    def validate(self, result: ResultT, context: ValidationContext) -> ValidationReport: ...
```

`StageContext` provides ports, not global singletons:

```python
@dataclass(frozen=True, slots=True)
class StageContext:
    run_id: str
    executor: Executor
    artifact_store: ArtifactStore
    tensor_store: TensorStore
    events: EventSink
    cancellation: CancellationToken
```

It does not provide the complete `RunConfig`. The application constructs requests containing only relevant typed config or its canonical hash.

## 14. Stage input/output matrix

| Stage | Typed request/input | Typed result/output | Default commit |
| --- | --- | --- | --- |
| Resolve source | `ModelConfig` | `ModelInventory` | model inventory |
| Prepare dataset | `DatasetConfig`, `ModelIdentity` | `DatasetSelection` | dataset selection |
| Calibrate | `CalibrationRequest` | `CalibrationStats` | block/layer statistics |
| Build objectives | `CalibrationStats`, `ObjectiveConfig` | objective artifacts | layer objective |
| Allocate | `PlanningRequest` | `QuantizationPlan` | complete plan |
| Capture/replay block | adapter request and activation refs | source block fixture | fixture |
| Select outliers | `OutlierSelectionRequest` | `OutlierSelectionResult` | layer selection |
| Factorize | `FactorizationRequest` | `FactorizationResult` | attempt result |
| Fit scales | `ScaleFitRequest` | `ScaleFitResult` | accepted result with layer |
| Decide retry | `RetryDecisionRequest` | `RetryDecision` | event only; pure decision |
| Tune layer | `LayerTuningRequest` | `TuningMetrics` plus trainable state | accepted layer |
| Freeze layer | trainable state and selected attempt | `LayerResult` | layer |
| Quantize block | `BlockQuantizationRequest` | `BlockResult` | block and activation generation |
| Global tune | frozen model and target cache | globally tuned frozen model | epoch/model boundary |
| Pack | `FrozenModelResult`, `PackingConfig` | `PackedModelResult` | shard/layout |
| Validate model | `PackedModelResult` | `ValidationReport` | report |
| Evaluate | evaluation request | `EvaluationResult` | evaluator task |
| Benchmark | benchmark request | benchmark result | workload case |
| Render report | manifest and result refs | report artifact | report |

## 15. Splitting the current factorize-and-replace behavior

The current combined behavior maps to these explicit operations:

```text
SourceWeightLoader.load
  → ObjectiveBuilder.build
  → OutlierSelector.select
  → Factorizer.factorize
  → ReconstructionEvaluator.measure
  → ScaleFitter.fit_and_compare
  → RetryPolicy.decide
       ↘ repeat Factorizer with a new immutable rank plan
  → TrainableLayerFactory.create
  → LayerTuner.tune
  → LayerFreezer.freeze
  → BlockEditor.install_frozen_layer
  → ArtifactStore.commit LayerResult
  → EventSink emits decision/result events
```

Important ownership changes:

- only `BlockEditor` mutates the stage-owned working block;
- the factorizer never knows a module path or report file;
- retry policy never runs ADMM;
- scale fitting never edits global budget state;
- event/report code observes results and never influences numerical decisions;
- the accepted layer commit is the only point where retry expenditure becomes durable.

## 16. In-memory compute views versus persisted DTOs

Persisted objects use tensor references. Compute components need real tensors, so infrastructure creates short-lived views:

```python
@dataclass
class MaterializedFactorizationInput:
    weight: torch.Tensor
    residual_weight: torch.Tensor
    input_importance: torch.Tensor
    output_importance: torch.Tensor
    covariance: Optional[MaterializedCovariance]


@dataclass
class MaterializedTrainableFactors:
    left_latent: torch.Tensor
    right_latent: torch.Tensor
    left_binary: torch.Tensor
    right_binary: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: Optional[torch.Tensor]
    scale_post: torch.Tensor


@dataclass
class MaterializedFactorizationOutput:
    factors: MaterializedTrainableFactors
    metrics: ReconstructionMetrics
    convergence: ConvergenceMetrics
```

Rules:

- materialized views are stage-private and not serialized;
- views release device/storage leases at scope exit;
- domain math returns `MaterializedFactorizationOutput` to the stage; the stage persists its tensors and then creates the immutable `FactorizationResult` DTO;
- persisted DTOs never contain a live CUDA tensor, `nn.Module`, hook handle, optimizer, logger, file object, or callable;
- unit tests may use an in-memory tensor store to avoid filesystem overhead.

## 17. Validation ownership

Each object has structural and semantic validation:

- constructor/decoder: required fields, enum values, primitive ranges;
- artifact validator: files, hashes, tensor names/specs;
- domain validator: layer dimensions, ranks, objective compatibility, bit accounting;
- stage validator: output completeness and numerical invariants;
- pipeline validator: output identity matches stage input and plan;
- runtime validator: packed capability and numerical parity.

Validation reports are typed results. A report may contain warnings but cannot silently repair an artifact.

## 18. Testing without a Hugging Face model

Typed requests make every service independently testable:

```python
request = FactorizationRequest(
    schema_version=1,
    layer=LayerId(BlockId(0), "mlp.up_proj"),
    source_weight=in_memory_store.put(weight),
    residual_weight=in_memory_store.put(weight),
    objective=make_diagonal_objective(weight.shape),
    rank=32,
    logical_seed=1234,
    factorizer_config_hash="sha256:...",
)

result = factorization_stage.execute(request, test_context)
assert result.layer == request.layer
assert result.rank == 32
assert result.metrics.export_weighted_normalized_error >= 0
```

Similarly, a block test supplies a tiny stage-owned block plus activation references. No tokenizer, Hub wrapper, complete model, global output path, or CLI parser is involved.

## 19. Contract evolution

- adding an optional field with an unambiguous default may use a minor schema migration;
- changing tensor meaning, normalization, bit accounting, or identity requires a new schema/component version;
- readers reject unknown required variants;
- migrations create a new artifact and retain parent lineage;
- fixture/golden tests cover every supported version;
- stage cache keys include producer version and semantic input schema.

## 20. Definition of complete domain extraction

Extraction is complete when:

- no domain/application boundary passes a `dict[str, Any]`, `argparse.Namespace`, or complete model config;
- no persisted result contains a live tensor/module/optimizer;
- every expensive unit can be replayed from typed artifact references;
- current factorization/outlier/Hessian/retry/scale-fit behavior maps to independently tested components;
- block results contain enough activation and state lineage to resume at the next block;
- reports are rendered from these results and events without inspecting mutated model internals.
