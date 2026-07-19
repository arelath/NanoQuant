"""Versioned domain contracts shared across stages and artifacts."""

from __future__ import annotations

from dataclasses import dataclass


class ArtifactTypes:
    """Canonical content-addressed artifact schema identities."""

    LAYER_RESULT = "layer-result"
    SHARED_INPUT_GROUP_RESULT = "shared-input-group-result"
    BLOCK_RESULT = "block-result"
    ACTIVATION_GENERATION = "activation-generation"
    QUANTIZATION_PLAN = "quantization-plan"
    RANK_PROBE_PLAN = "rank-probe-plan"
    RANK_PROBE_RESULT = "rank-probe-result"
    RECONSTRUCTION_RANK_PROFILE = "reconstruction-rank-profile"
    MEMORY_PLAN = "memory-plan"
    EVALUATION_TASK_INPUTS = "evaluation-task-inputs"
    EVALUATION_RESULT = "evaluation-result"


@dataclass(frozen=True, slots=True, order=True)
class BlockId:
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("block index must not be negative")


@dataclass(frozen=True, slots=True, order=True)
class LayerId:
    block: BlockId
    path: str

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in self.path.split("."):
            raise ValueError("layer path must be a non-empty canonical module path")


@dataclass(frozen=True, slots=True)
class SharedInputGroupCandidate:
    """Adapter-declared logical projections that consume the same activation."""

    block: BlockId
    name: str
    members: tuple[LayerId, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith("/") or ".." in self.name.split("."):
            raise ValueError("shared-input group name must be a canonical dotted path")
        if len(self.members) < 2:
            raise ValueError("shared-input group requires at least two members")
        if len(set(self.members)) != len(self.members):
            raise ValueError("shared-input group members must be unique")
        if any(member.block != self.block for member in self.members):
            raise ValueError("shared-input group members must belong to its block")


@dataclass(frozen=True, slots=True, order=True)
class TensorId:
    layer: LayerId | None
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
    device_requirement: str | None = None

    def __post_init__(self) -> None:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must not be negative")


@dataclass(frozen=True, slots=True)
class TensorRef:
    artifact: ArtifactRef
    key: str
    spec: TensorSpec
    content_hash: str


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
    bias: SourceTensor | None
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
class CheckpointTensorMetadata:
    key: str
    shard: str
    spec: TensorSpec
    shard_hash: str | None


@dataclass(frozen=True, slots=True)
class CheckpointInventory:
    schema_version: int
    source: str
    revision: str
    config: dict[str, object]
    tokenizer_files: tuple[str, ...]
    tokenizer_hash: str
    tensors: tuple[CheckpointTensorMetadata, ...]
    total_shard_bytes: int


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


@dataclass(frozen=True, slots=True)
class StatisticSummary:
    minimum: float
    maximum: float
    mean: float
    zero_fraction: float
    non_finite_count: int


@dataclass(frozen=True, slots=True)
class CovarianceRef:
    representation: str
    diagonal: TensorRef
    blocks: TensorRef | None = None
    low_rank_factors: TensorRef | None = None
    dense: TensorRef | None = None
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class LayerCalibrationStats:
    layer: LayerId
    input_importance: TensorRef
    output_importance: TensorRef
    input_covariance: CovarianceRef | None
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


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    schema_version: int
    layer: LayerId
    kind: str
    input_importance: TensorRef
    output_importance: TensorRef
    covariance: CovarianceRef | None
    damping: float
    normalization: str
    target_weighted_norm_squared: float | None
    source_calibration: ArtifactRef


@dataclass(frozen=True, slots=True)
class BitCost:
    binary_factor_bits: int = 0
    scale_bits: int = 0
    outlier_value_bits: int = 0
    outlier_index_bits: int = 0
    padding_bits: int = 0

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.binary_factor_bits,
                self.scale_bits,
                self.outlier_value_bits,
                self.outlier_index_bits,
                self.padding_bits,
            )
        ):
            raise ValueError("bit costs must not be negative")

    @property
    def total(self) -> int:
        return sum(
            (
                self.binary_factor_bits,
                self.scale_bits,
                self.outlier_value_bits,
                self.outlier_index_bits,
                self.padding_bits,
            )
        )

    def __add__(self, other: BitCost) -> BitCost:
        return BitCost(*(left + right for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)))

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.binary_factor_bits,
            self.scale_bits,
            self.outlier_value_bits,
            self.outlier_index_bits,
            self.padding_bits,
        )


@dataclass(frozen=True, slots=True)
class OutlierPlan:
    selector: str
    count: int
    storage_dtype: str
    charge_to_budget: bool
    removed_column_importance: str = "zero"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    maximum_attempts: int
    rank_increase_fraction: float
    weighted_error_threshold: float | None
    raw_error_threshold: float | None
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
class SharedInputGroupPlan:
    schema_version: int
    block: BlockId
    name: str
    members: tuple[LayerInventory, ...]
    rank: int
    rank_multiple: int
    allocator_cap: int
    objectives: tuple[ObjectiveSpec, ...]
    outliers: OutlierPlan
    retry: RetryPolicy
    estimated_cost: BitCost

    def __post_init__(self) -> None:
        if len(self.members) < 2 or len(self.members) != len(self.objectives):
            raise ValueError("shared-input plan requires aligned member objectives")
        if any(member.layer.block != self.block for member in self.members):
            raise ValueError("shared-input plan contains a member from another block")
        if len({member.in_features for member in self.members}) != 1:
            raise ValueError("shared-input plan members must have equal input width")
        if tuple(objective.layer for objective in self.objectives) != tuple(member.layer for member in self.members):
            raise ValueError("shared-input plan objectives must follow member order")

    @property
    def unit_id(self) -> str:
        return f"{self.block.index}:{self.name}"

    @property
    def in_features(self) -> int:
        return self.members[0].in_features

    @property
    def out_features(self) -> int:
        return sum(member.out_features for member in self.members)


@dataclass(frozen=True, slots=True)
class BlockPlan:
    block: BlockId
    layer_order: tuple[LayerId, ...]
    layers: tuple[LayerPlan, ...]
    estimated_workspace_bytes: int
    shared_input_groups: tuple[SharedInputGroupPlan, ...] = ()
    unit_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ordinary = {layer.layer.path for layer in self.layers}
        grouped = [member.layer.path for group in self.shared_input_groups for member in group.members]
        if ordinary.intersection(grouped) or len(grouped) != len(set(grouped)):
            raise ValueError("block plan units must partition logical layers")
        expected = {*ordinary, *(group.name for group in self.shared_input_groups)}
        if self.unit_order and (set(self.unit_order) != expected or len(self.unit_order) != len(expected)):
            raise ValueError("block plan unit order must contain every unit exactly once")


@dataclass(frozen=True, slots=True)
class QuantizationPlan:
    schema_version: int
    producer: ComponentRef
    model: ModelIdentity
    calibration: ArtifactRef
    blocks: tuple[BlockPlan, ...]
    target_bpw: float
    planned_cost: BitCost
    reconstruction_profile: ArtifactRef | None = None
    reconstruction_decisions: tuple[ReconstructionRankDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class ReconstructionRankDecision:
    unit_id: str
    members: tuple[LayerId, ...]
    baseline_rank: int
    planned_rank: int
    baseline_squared_error: float
    predicted_squared_error: float
    sensitivity: float
    protected_members: tuple[LayerId, ...]

    def __post_init__(self) -> None:
        if not self.unit_id or not self.members:
            raise ValueError("reconstruction decision requires a unit and members")
        if self.baseline_rank <= 0 or self.planned_rank <= 0:
            raise ValueError("reconstruction decision ranks must be positive")


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
    scales: TensorRef | None
    residual_weight: TensorRef
    factor_input_importance: TensorRef
    factor_generator_state: TensorRef | None
    selected_score_summary: StatisticSummary
    bit_cost: BitCost


@dataclass(frozen=True, slots=True)
class ScaleState:
    pre: TensorRef
    mid: TensorRef | None
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
    latent_weighted_error: float | None
    latent_weighted_normalized_error: float | None
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
    final_primal_residual: float | None
    final_dual_residual: float | None
    trace: ArtifactRef | None


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
    generator_state: TensorRef | None = None


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


@dataclass(frozen=True, slots=True)
class ScaleFitRequest:
    layer: LayerId
    target_weight: TensorRef
    factors: TrainableFactors
    objective: ObjectiveSpec
    protected_columns: TensorRef | None


@dataclass(frozen=True, slots=True)
class ScaleFitResult:
    scales: ScaleState
    before: ReconstructionMetrics
    after: ReconstructionMetrics
    accepted: bool
    rollback_reason: str | None


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
    action: str
    accepted_attempt: int | None
    next_rank: int | None
    projected_extra_bits: int
    reason: str


@dataclass(frozen=True, slots=True)
class LossMetrics:
    loss: float
    valid_elements: int
    objective: str


@dataclass(frozen=True, slots=True)
class TuningMetrics:
    before: LossMetrics | None
    best: LossMetrics
    final: LossMetrics
    epochs_completed: int
    best_epoch: int
    stopped_early: bool
    trace: ArtifactRef | None


@dataclass(frozen=True, slots=True)
class FrozenOutlierState:
    indices: TensorRef
    values: TensorRef
    scales: TensorRef | None


@dataclass(frozen=True, slots=True)
class FrozenNanoQuantState:
    layer: LayerId
    rank: int
    left_binary: TensorRef
    right_binary: TensorRef
    scales: ScaleState
    outliers: FrozenOutlierState | None
    bias: TensorRef | None
    logical_format: str


@dataclass(frozen=True, slots=True)
class SharedInputMemberSlice:
    layer: LayerId
    row_start: int
    row_end: int

    def __post_init__(self) -> None:
        if self.row_start < 0 or self.row_end <= self.row_start:
            raise ValueError("shared-input member slice must be a non-empty forward interval")


@dataclass(frozen=True, slots=True)
class FrozenSharedInputGroupState:
    block: BlockId
    name: str
    members: tuple[SharedInputMemberSlice, ...]
    rank: int
    left_binary: TensorRef
    right_binary: TensorRef
    scales: ScaleState
    outliers: FrozenOutlierState | None
    bias: TensorRef | None
    logical_format: str

    def __post_init__(self) -> None:
        if len(self.members) < 2 or any(member.layer.block != self.block for member in self.members):
            raise ValueError("frozen shared-input group has invalid members")
        cursor = 0
        for member in self.members:
            if member.row_start != cursor:
                raise ValueError("frozen shared-input member slices must be contiguous")
            cursor = member.row_end


@dataclass(frozen=True, slots=True)
class LayerResult:
    schema_version: int
    layer: LayerId
    plan: LayerPlan
    attempts: tuple[AttemptSummary, ...]
    accepted_attempt: int
    factorization: ArtifactRef
    scale_fit: ScaleFitResult | None
    tuning: TuningMetrics | None
    frozen_state: FrozenNanoQuantState
    final_reconstruction: ReconstructionMetrics
    actual_bit_cost: BitCost
    extra_retry_bits: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SharedInputGroupResult:
    schema_version: int
    block: BlockId
    name: str
    plan: SharedInputGroupPlan
    attempts: tuple[AttemptSummary, ...]
    accepted_attempt: int
    factorization: ArtifactRef
    scale_fit: ScaleFitResult | None
    tuning: TuningMetrics | None
    frozen_state: FrozenSharedInputGroupState
    final_reconstruction: ReconstructionMetrics
    member_reconstruction: tuple[tuple[LayerId, ReconstructionMetrics], ...]
    actual_bit_cost: BitCost
    extra_retry_bits: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActivationStreamRef:
    artifact: ArtifactRef
    shape: tuple[int, ...]
    dtype: str
    sample_count: int
    sequence_length: int


@dataclass(frozen=True, slots=True)
class LossComparison:
    baseline_name: str
    candidate_name: str
    baseline_loss: float
    candidate_loss: float
    absolute_delta: float
    relative_delta: float | None
    denominator_floor: float


@dataclass(frozen=True, slots=True)
class BlockLossMetrics:
    source_reference: float
    block_entry_pre_quantization: float
    after_each_layer: tuple[tuple[LayerId, float], ...]
    after_post_block_refit: float | None
    final_frozen_pre_kd: float
    final_vs_block_entry: LossComparison
    final_vs_source_reference: LossComparison
    target_weighted_mean_square: float | None = None
    block_entry_normalized_error: float | None = None
    final_frozen_normalized_error: float | None = None


@dataclass(frozen=True, slots=True)
class FrozenBlockState:
    block: BlockId
    quantized_layers: tuple[FrozenNanoQuantState, ...]
    passthrough_tensors: tuple[TensorRef, ...]
    auxiliary_parameters: tuple[tuple[str, TensorRef], ...] = ()
    shared_input_groups: tuple[FrozenSharedInputGroupState, ...] = ()


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
    shared_input_groups: tuple[SharedInputGroupResult, ...] = ()


@dataclass(frozen=True, slots=True)
class GlobalTuningBlockMetrics:
    block: BlockId
    final_frozen_pre_kd: float
    final_post_kd: float
    post_kd_vs_pre_kd: LossComparison


@dataclass(frozen=True, slots=True)
class GlobalTuningResult:
    schema_version: int
    source_blocks: tuple[ArtifactRef, ...]
    tuned_blocks: tuple[FrozenBlockState, ...]
    auxiliary_parameters: tuple[tuple[str, TensorRef], ...]
    protocol_hash: str
    token_hash: str
    epoch_losses: tuple[float, ...]
    steps_completed: int
    selected_parameter_count: int
    teacher_cache_bytes: int
    wall_seconds: float
    peak_gpu_bytes: int
    peak_host_bytes: int
    block_snapshot_protocol_hash: str | None = None
    block_metrics: tuple[GlobalTuningBlockMetrics, ...] = ()


@dataclass(frozen=True, slots=True)
class FrozenModelResult:
    schema_version: int
    model: ModelIdentity
    plan: ArtifactRef
    blocks: tuple[ArtifactRef, ...]
    shared_tensors: tuple[TensorRef, ...]
    global_tuning: ArtifactRef | None
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
    sample_count: int | None
    confidence_interval: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    schema_version: int
    evaluation_id: str
    tier: str
    model_artifact: ArtifactRef
    baseline_artifact: ArtifactRef | None
    metrics: tuple[MetricValue, ...]
    resource_usage: ArtifactRef
    status: str
    warnings: tuple[str, ...]


@dataclass(slots=True)
class MaterializedFactorizationInput:
    weight: object
    residual_weight: object
    input_importance: object
    output_importance: object
    covariance: object | None


@dataclass(slots=True)
class MaterializedTrainableFactors:
    left_latent: object
    right_latent: object
    left_binary: object
    right_binary: object
    scale_pre: object
    scale_mid: object | None
    scale_post: object


@dataclass(slots=True)
class MaterializedFactorizationOutput:
    factors: MaterializedTrainableFactors
    metrics: ReconstructionMetrics
    convergence: ConvergenceMetrics
