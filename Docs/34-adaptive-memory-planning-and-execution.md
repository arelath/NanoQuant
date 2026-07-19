# Adaptive Memory Planning and Execution

Status: resident adaptive core implemented; measured promotion remains gated

Date: 2026-07-19

Audience: configuration, resident execution, resource planning, tuning, calibration, evaluation, and observability maintainers

## 1. Summary

NanoQuant should select execution settings from the memory actually available to the process and the memory expected
for each stage, instead of relying on model-specific batches and placement flags chosen by an operator. The desired
behavior is:

1. inspect GPU, host, pinned-host, and temporary-disk capacity before loading weights;
2. derive a resource envelope from observed capacity, user ceilings, explicit reserves, and estimator uncertainty;
3. estimate fixed and scalable memory for every materially different stage;
4. select the highest-throughput admissible executor, placement, cache, prefetch depth, and physical batch size;
5. measure actual stage peaks and refine later choices at safe boundaries;
6. recover from a genuine OOM with a finite, persisted, algorithm-preserving plan revision; and
7. fail clearly rather than silently paging, changing the compression algorithm, or retrying forever.

This is a two-level controller. A static preflight plan prevents predictable failures and makes an initial performance
choice. A conservative online controller handles estimator error and changing external pressure. Neither layer
mutates the canonical recipe. The resolved execution plan and every revision are persisted separately and reused on
resume.

### Implementation status (2026-07-19)

The first production slice is implemented for the resident compression workflow:

- `fixed` remains the compatibility default, while `adaptive` is an explicit memory-policy mode;
- preflight builds a process-aware GPU/host/pinned/disk envelope and a metadata-only model inventory before CUDA
  weight loading;
- the resolved plan selects resident versus CPU-offload execution, activation caching, and separate physical batch
  sizes for calibration, block forward, tuning, and post-block refit;
- minimum viable configurations are admitted before work starts, and user ceilings and reserves are enforced;
- the plan is stored as a content-addressed artifact with an active pointer and finite, journaled OOM revisions;
- a CUDA OOM reduces only the affected scalable stage when it can be identified, then evicts optional activation
  cache before failing; logical optimizer batches and every compression semantic remain unchanged;
- resident events record plan admission, resize decisions, observations, revision, utilization, and per-block planned
  memory; and
- unit and workflow tests cover plan selection, admission failure, persistence, config resolution, stable semantic
  identity across revisions, and OOM resume behavior.

The following performance promotions remain deliberately gated by the rollout and acceptance criteria below:
learned cross-run estimator profiles, within-run upward growth, source-prefetch resizing, adaptive model-level
distillation/evaluation batches, streaming-mode transitions, and enabling adaptive mode in the canonical Gemma
recipe. They require protocol-matched 1B/4B canaries rather than being enabled from synthetic fixtures alone.

## 2. Current implementation and gaps

The repository already contains many of the necessary pieces, but they are not connected into the production
resident workflow.

| Existing capability | Current location | Current limitation |
| --- | --- | --- |
| Host/GPU/disk inventory | `infrastructure/resource_planning.py::inspect_host` | Used by tests, not the resident composition root |
| Pure resource plan and activation-tier selection | `domain/resources.py`, `infrastructure/resource_planning.py` | Consumes caller-supplied aggregate estimates; no production model inventory builds them |
| Resource ceilings | `config/schema.py::ResourceLimitsConfig` | The resident workflow rejects every non-default value as "not yet enforced" |
| `auto` executor and activation resolution | `config/resolution.py` | Chooses resident whenever CUDA exists and CUDA activations for resident; it does not consider model size or free memory |
| Stage resource estimates | typed stage `estimate()` methods | Estimates are sparse, often CPU-only, and are not calibrated against observed peaks |
| GPU activation cache admission | `_cache_activation_tensor` in `resident_quantization.py` | Uses current free VRAM and a reserve, but considers only the cache tensor, not the next stage's complete peak |
| Bounded pageable-to-CUDA staging | `application/device_batches.py`, `application/tuning.py` | Buffer sizes come from configured batches rather than a resource plan |
| CUDA/host/WDDM meters and peak windows | `infrastructure/device_memory.py` | Excellent diagnostics, but measurements do not feed future sizing decisions |
| OOM forensics | `capture_oom_forensics` and the resource event sink | Captures the failure but does not resize the resident stage |
| Finite generic fallback helpers | `application/runtime_fallback.py`, `application/calibration_fallback.py` | Not composed into resident execution; runtime action names also differ from the schema defaults |
| Rolling activation retention | resident block-result v2 and `Docs/14` | Bounds durable disk, but is not incorporated into a complete GPU/host/disk admission plan |

The production recipe therefore hard-codes values such as block-forward batch 4, tuning microbatch 1, and disabled
inline quality for Gemma 3 4B. Those are sensible known-host choices, but they leave throughput unused on larger
devices and still depend on operator knowledge to avoid OOM on smaller or externally loaded devices.

Measured evidence in `Docs/18-legacy-quantization-performance-and-vram-investigation.md` also shows why a single
"VRAM used" number is inadequate:

- the true allocated high-water can occur in post-block refit rather than factorization;
- allocator reservation can remain much larger than live allocation after a high-water stage;
- completed frozen blocks can create block-by-block growth when retained on CUDA;
- full activation pinning can exhaust WDDM shared memory while CUDA allocation appears bounded; and
- a tuning microbatch of 1 can be dramatically slower than 8 even when both fit.

The new planner must model stage lifetimes and separate live tensors, reusable allocator reservation, driver-visible
free memory, pageable host memory, pinned-host memory, and durable/scratch disk.

## 3. Goals

- Use available resources aggressively enough to select the fastest safe execution plan on the current host.
- Prevent predictable CUDA, host-memory, pinned-memory, and disk OOMs before expensive work begins.
- Adapt to layer/block shape changes and external resource pressure without uncontrolled paging.
- Preserve the logical optimizer batch, learning schedule, rank plan, objective, calibration method, dtype, and all
  other algorithmic settings unless the user explicitly authorizes an algorithm-changing fallback.
- Preserve deterministic resume by persisting resolved settings and plan revisions at existing safe commit boundaries.
- Make estimates, decisions, headroom, fallbacks, and estimator error visible in structured evidence.
- Reuse the existing architecture: pure policy in `domain`, orchestration in `application`, host inspection and tensor
  sizing in `infrastructure`, and execution at the resident/streaming composition roots.

## 4. Non-goals

- This design does not automatically lower rank, change BPW, switch Hessian representation, reduce calibration
  samples, change calibration method, alter tuning epochs, or disable evaluation.
- It does not treat Windows shared GPU memory as extra CUDA capacity. WDDM spill is a failure condition, not a tier.
- It does not make arbitrary mid-kernel preemption possible. Adaptation occurs only before a stage or after rollback
  to a durable boundary.
- It does not promise identical throughput decisions across different hardware. It promises that the complete
  resolved plan is recorded and replayed unless the operator explicitly requests replanning.
- It does not replace CUDA allocator diagnostics or the resource sampler described in `Docs/17`; it consumes them.

## 5. Design principles

### 5.1 Separate semantic and execution decisions

The controller may change where and in what physical chunk size equivalent work executes. It may not silently change
what mathematical work is performed. The distinction must be explicit:

| Algorithm-preserving execution decisions | Algorithm-changing decisions |
| --- | --- |
| Resident versus CPU-offload placement when both implement the same stage | Online Fisher to forward-only calibration |
| CUDA/input/both/pageable/mmap activation placement | Dense Hessian to diagonal or low-rank objective |
| Physical tuning microbatch with the logical optimizer batch held fixed | Logical optimizer batch or scheduler step count |
| No-gradient forward/evaluation batch or token chunk size | Sample count, token selection, rank, outliers, or BPW |
| Prefetch depth, double buffering, cache eviction, completed-block offload | Dtype or numerical kernel with different approved tolerances |

Some settings currently participate in `_resident_config_hash`, including tuning microbatch and block-forward batch.
The first implementation should preserve that conservative identity rule: `auto` resolves to concrete values before
the resident semantic identity is finalized. A later parity study may prove selected batching choices
execution-only, but the memory project should not assume that result.

### 5.2 Optimize under a hard envelope, not up to reported free bytes

`torch.cuda.mem_get_info()` is driver truth at one instant, not a safe allocation target. The plan must preserve:

- a fixed operator reserve for the desktop, display driver, peer work, and CUDA context growth;
- a proportional safety margin;
- an uncertainty allowance derived from estimator error;
- space for the largest known indivisible allocation; and
- extra protection when allocator fragmentation or external pressure is unstable.

### 5.3 Model stage peaks, not run-wide sums

Tensors with disjoint lifetimes must not be added together. Conversely, tensors alive together at post-block refit,
quality evaluation, activation commit, or tuning checkpoint creation must be represented together. The plan is a
timeline of stage peaks and handoff peaks, not one aggregate component total.

### 5.4 Prefer measured high-water data over permanent conservatism

Static tensor accounting establishes a safe first attempt. Observed peak allocated/reserved, host working set,
pinned/WDDM shared usage, and disk growth correct the model. Estimator error is retained by stage signature and used
as a future uncertainty margin rather than forcing all workloads to inherit the worst global margin.

## 6. Core contracts

### 6.1 Resource envelope

Add a pure `ResourceEnvelope` value describing capacity available to NanoQuant at planning time:

```python
@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    device: str
    gpu_total_bytes: int
    gpu_free_bytes: int
    gpu_process_allocated_bytes: int
    gpu_process_reserved_bytes: int
    gpu_hard_limit_bytes: int
    gpu_reserve_bytes: int
    host_available_bytes: int
    host_process_bytes: int
    host_hard_limit_bytes: int
    pinned_host_limit_bytes: int
    temporary_disk_free_bytes: int
    temporary_disk_hard_limit_bytes: int
    observed_at: str
```

The configured `gpu_memory_gib`, `cpu_memory_gib`, and `temporary_disk_gib` are hard process/run ceilings, not claims
that the memory exists. An omitted ceiling uses detected capacity after reserves. `pinned_memory_gib` remains a
separate cap because pinned pages affect host health and WDDM behavior differently from pageable RAM.

For CUDA admission, define:

```text
reusable_pool       = max(0, reserved - allocated)
physical_headroom   = device_free + reusable_pool
policy_headroom     = max(0, gpu_hard_limit - allocated)
uncertainty         = max(minimum_uncertainty, predicted_increment * estimator_error)
safe_increment      = min(physical_headroom, policy_headroom)
                      - gpu_reserve - uncertainty
```

The reusable pool is included because PyTorch can satisfy new tensors from its reservation even though
`mem_get_info()` reports those bytes unavailable. It is discounted or ignored when recent observations indicate
fragmentation. If `reserved - allocated` is large but unusable, a safe-boundary cache release and a fresh envelope
are preferable to optimistic admission.

Host admission uses the smaller of the configured remaining allowance and OS-reported available memory, less a host
reserve and estimator uncertainty. It never counts the page file as equivalent to RAM. Temporary-disk admission
includes the rolling generation commit peak from `Docs/14`, scratch writers, checkpoint replacement, and a minimum
free-space reserve.

### 6.2 Stage memory model

Replace the single aggregate-only estimate with composable, stage-specific estimates:

```python
@dataclass(frozen=True, slots=True)
class StageMemoryModel:
    stage: str
    signature: str
    fixed_gpu_bytes: int
    gpu_bytes_per_row: int
    fixed_host_bytes: int
    host_bytes_per_row: int
    pinned_bytes_per_prefetch_slot: int
    temporary_disk_bytes: int
    largest_indivisible_gpu_allocation_bytes: int
    minimum_batch_size: int
    maximum_batch_size: int
    confidence: str
```

`signature` includes the properties that explain memory shape: operation, model family, block/layer type and tensor
shapes, sequence length, dtype, factor rank, trainable parameter bytes, optimizer implementation, gradient
checkpointing, activation placement, and relevant kernel version. It excludes run IDs and presentation fields.

Not every stage is linear in rows. A model can provide either a conservative closed-form `peak(batch)` function or
a piecewise estimate. The fixed/per-row form is the default because it supports monotone batch search and is easy to
audit. Indivisible workspaces, such as an ADMM matrix temporary or dense Hessian, remain fixed terms and can make a
stage impossible even at batch 1.

Initial models come from tensor metadata and explicit ownership accounting:

- model shell and active/completed block residency;
- inputs, targets, metadata, logits, and loss temporaries;
- gradients, current/best parameter copies, optimizer moments, Kahan state, and scheduler state;
- ADMM residual, latent, binary, reconstruction, importance, and solve workspaces;
- two host and two device staging slots in `iter_device_batches`;
- activation GPU cache inputs/targets;
- calibration graphs and accumulators;
- distillation teacher chunks, vocabulary/token chunks, and optimizer state;
- mmap mappings, pageable generations, pinned staging, checkpoint snapshots, and atomic successor generations.

Measured profiles then supply a correction factor or absolute residual for matching signatures.

### 6.3 Resolved execution plan

Persist an immutable `ResolvedMemoryPlan` artifact before model allocation. It contains:

- the envelope and its source observations;
- user ceilings/reserves and policy profile;
- executor and model/block residency policy;
- activation tier and GPU cache policy;
- per-stage physical batch/microbatch, token/vocabulary chunk, and prefetch depth;
- predicted GPU allocated/reserved, host, pinned-host, and disk peaks;
- estimator confidence and safety allowance;
- semantic versus execution-only classification for every resolved choice; and
- a monotonically increasing plan revision.

Each plan revision records its parent, reason, triggering observation/OOM, earliest safe retry boundary, and whether
semantic identity changes. `RunState` should reference the active revision rather than embedding mutable settings in
`RunConfig`.

## 7. Planning lifecycle

### 7.1 Metadata preflight

Before loading the model:

1. inventory checkpoint shards, tensor shapes, dtypes, tied weights, blocks, and largest layers through
   `ModelSource` and the adapter;
2. calculate source, packed output, durable result, rolling resume, atomic commit, and scratch disk bytes;
3. inspect GPU/host/disk capacity and apply configured hard ceilings;
4. build candidate executor/placement plans;
5. reject candidates whose fixed or indivisible working set cannot fit;
6. choose the fastest candidate using measured cost profiles when present and a deterministic preference order
   otherwise; and
7. persist and print the resolved plan before the first expensive allocation.

Candidate preference should normally be:

```text
resident with reusable CUDA activations
resident with bounded activation GPU cache
resident with pageable activations
CPU offload with pageable activations
streaming with pageable activations
streaming with mmap activations
```

Pinned RAM is a bounded staging resource, not the default home for complete multi-GiB generations. This preserves the
WDDM fix documented in `Docs/18`.

### 7.2 Post-load reconciliation

Model loading creates a CUDA context and may reveal adapter-specific storage absent from metadata. Immediately after
the shell is loaded and unused blocks are released:

1. sample actual allocation/reservation and host working set;
2. compare them with the planned baseline;
3. recompute scalable stage sizes from the remaining envelope; and
4. fail or revise placement before calibration if the fixed baseline already violates the hard limit.

This checkpoint catches tied-weight behavior, library workspaces, and driver overhead without waiting for a deep
block to OOM.

### 7.3 Stage admission

At every material stage boundary, calculate the predicted incremental peak against a fresh envelope. The admission
decision has three results:

- `admit`: predicted peak is below the target;
- `resize`: a smaller physical batch/chunk/prefetch setting fits; or
- `replan/fail`: fixed memory does not fit and a placement transition or explicit failure is required.

The controller should not poll inside hot loops. Existing block/layer/epoch/checkpoint boundaries are sufficient,
and the periodic sampler remains read-only.

## 8. Selecting high-performance sizes

### 8.1 Deterministic bounded search

For a monotone scalable setting, choose the largest candidate whose predicted peak is within the target. Use a
bounded geometric candidate set rather than probing every integer:

```text
candidates = configured maximum, then descending powers/factors to 1
selected   = largest candidate with predicted_peak(candidate) <= target
```

When no measured model exists and a cheap, side-effect-free probe is available, use exponential growth followed by
binary search. Probes must use representative tensor shapes, execute under `PeakWindow`, and release all temporary
state before the real stage. Do not probe a mutating optimizer step or a factorization attempt whose RNG/result would
become part of the algorithm. For those stages, use static accounting plus observations from the first real minimal
unit.

### 8.2 Hysteresis

Avoid oscillation and needless reallocation:

- shrink immediately when a predicted or observed hard threshold is crossed;
- grow only at a block/stage boundary after at least two matching observations show sufficient headroom;
- require the next candidate to improve estimated throughput materially (for example, at least 5%);
- never grow above the persisted configured maximum; and
- do not grow during a resumed unit unless the user requested replanning.

### 8.3 Stage-specific controls

| Stage | Primary scalable controls | Fixed-memory escape when batch 1 does not fit |
| --- | --- | --- |
| Prefix/teacher/block forward and loss snapshots | `block_forward_batch_size`, row/token chunk | Evict activation cache; offload completed blocks; change executor |
| Online Fisher calibration | physical calibration batch, checkpointing, placement | CPU offload; a method change requires explicit semantic fallback |
| Outlier/residual probes and ADMM | normally none; implementation workspace tiles if parity-proven | Offload other live state; reject if indivisible workspace exceeds limit |
| Non-factorized/factorized tuning | physical microbatch only; logical batch remains fixed | Evict target/input cache; offload completed blocks; CPU/streaming execution if implemented |
| Post-block refit | refit physical microbatch, immutable-factor fast path | Same placement actions; never silently lower logical refit batch |
| Activation propagation | forward batch, prefetch slots, input/target GPU cache | Pageable RAM, then mmap |
| Global distillation | batch, token chunk, vocabulary chunk, teacher-cache placement | Gradient checkpointing and offload; semantic settings remain fixed |
| Inline/final quality | sample batch, position/logit chunk, compact dense/factorized backend | Stream completed blocks or use packed evaluation |
| Packing/checkpoint commit | shard/write chunk, overlap depth | Disable overlap; serialize writers; require disk commit peak |

The expected payoff is asymmetric. Increasing tuning microbatch from 1 toward the logical batch can remove many
small launches and transfers. Increasing prefetch from zero to one can overlap I/O. Larger no-gradient batches can
improve GEMM utilization. By contrast, retaining complete target activations on CUDA can consume large capacity for
less benefit; the planner should rank choices by measured throughput gained per byte rather than following one fixed
flag order.

## 9. Online feedback and estimator learning

At the end of every admitted stage, record:

```text
predicted peak allocated/reserved/host/pinned/disk
observed baseline and peak for the same window
external free-memory minimum
selected batch/chunk/prefetch/cache settings
estimator absolute and relative error
wall time, tokens/rows, transfer bytes, and cache hit state
```

Update only an execution-profile cache, never source evidence or the semantic recipe. The cache is keyed by stage
signature, hardware identity, PyTorch/CUDA versions, and allocator configuration. Use an upper prediction interval
(for example, a high percentile plus a minimum byte floor), not the mean error. Stale or mismatched profiles are
ignored.

Within a run, later same-signature stages use the maximum observed positive residual. Across runs, retain a bounded
sample window and reject outliers caused by suspension or unrelated external pressure from throughput fitting, while
still retaining them for capacity safety.

If an observed peak exceeds the planned hard target without OOM, latch a pressure event and shrink the next safe
unit. The periodic sampler can detect transient WDDM shared-memory or host pressure, but it should only signal the
controller; it must not mutate execution from its background thread.

## 10. OOM recovery

OOM handling is a last safety net, not the primary sizing mechanism.

### 10.1 Required sequence

On CUDA or host OOM:

1. capture existing OOM forensics before cleanup;
2. classify the active stage, live placement, requested allocation, and whether the stage is safely retryable;
3. abandon uncommitted outputs and restore the latest valid layer/block/epoch boundary;
4. release known temporary tensors, stale staging buffers, and—at a safe boundary—unoccupied allocator caches;
5. resample the resource envelope;
6. create and persist exactly one lower-memory plan revision;
7. retry with the same logical seed and semantic inputs; and
8. stop after a finite stage/run retry count.

### 10.2 Algorithm-preserving fallback order

Choose the action with the lowest estimated performance cost that reclaims enough memory, usually:

1. reduce the active physical batch/chunk;
2. reduce prefetch/double-buffer depth;
3. evict teacher targets, then inputs, from the activation GPU cache;
4. offload completed frozen blocks not needed by the block loop;
5. release excess CUDA or pinned-host allocator cache at a safe boundary;
6. move complete activations from CUDA to pageable RAM, then mmap;
7. transition model placement from resident to CPU offload/streaming at a reconstructable boundary; and
8. fail with the minimum required bytes and the closest viable alternative.

An action is attempted only if it applies and predicts sufficient reclamation. This replaces the current
attempt-each-string-once helper behavior with resource-aware selection while retaining a hard attempt limit.

Algorithm-changing actions live in a separate, opt-in policy and create a fork or new semantic plan. They must never
be appended to the default memory fallback sequence.

### 10.3 Resume rules

The active plan revision is journaled with each unit. On ordinary resume, reuse the concrete resolved settings even
if more memory is now free; this preserves the execution trajectory. If less memory is available, admission may
create a lower-memory revision before restarting the incomplete unit. A command-line `--replan-memory` may opt into
new higher-performance settings and must report whether that changes resident identity or invalidates in-progress
tuning checkpoints.

## 11. Configuration model

Keep hard limits distinct from adaptive policy. Existing resource fields become effective:

```yaml
runtime:
  resources:
    gpu_memory_gib: null       # optional hard NanoQuant ceiling
    cpu_memory_gib: null       # optional hard NanoQuant ceiling
    pinned_memory_gib: 1.0     # hard pinned-host ceiling
    temporary_disk_gib: null   # optional per-run disk ceiling
    workspace_memory_gib: null # hard indivisible-workspace ceiling
  memory_policy:
    mode: adaptive             # fixed | adaptive
    profile: balanced          # conservative | balanced | throughput
    gpu_reserve_gib: 1.0
    host_reserve_gib: 4.0
    temporary_disk_reserve_gib: 8.0
    maximum_stage_retries: 3
    allow_growth_within_run: true
```

Profiles supply reviewed defaults for target utilization, estimator confidence, growth hysteresis, and cache-release
thresholds. Avoid exposing every controller constant as a recipe field.

In adaptive mode, existing physical execution settings are maxima:

- `runtime.block_forward_batch_size` is the no-gradient forward maximum;
- `runtime.activations.batch_size` and `prefetch_batches` are staging maxima;
- `block_tuning.microbatch_size` is the tuning physical maximum, with `None` meaning the logical batch maximum; and
- calibration, distillation, and evaluation batch/chunk fields are maxima where their partition invariance is
  already guaranteed.

The logical tuning `loop.batch_size` is never resized. Fixed mode preserves current behavior exactly. Adaptive mode
should remain opt-in until tiny, 1B, and 4B parity/performance gates pass, then become the resolved default for
`auto` executor configurations.

## 12. Integration with the codebase

### Domain

- Extend `domain/resources.py` with `ResourceEnvelope`, `StageMemoryModel`, `StageExecutionPlan`,
  `ResolvedMemoryPlan`, and `MemoryPlanRevision` pure types.
- Add pure policy functions for envelope calculation, candidate admission, batch selection, and fallback ranking.
- Keep `torch` out of new policy code. The existing `peak_device_memory_bytes` helper should move to infrastructure
  to restore the intended dependency direction.

### Application

- Add a `MemoryController` service that owns stage admission and plan revision decisions.
- Extend stage estimates to declare lifetimes and scalable dimensions.
- Replace disconnected fallback wrappers with one typed recovery protocol used by calibration, resident
  quantization, distillation, and evaluation.
- Make every stage accept a resolved `StageExecutionPlan` rather than the full memory configuration.

### Infrastructure

- Extend host inspection with current process allocation/reservation, configured hard limits, and disk paths.
- Add metadata-based model/block/tensor sizing through `ModelSource` without full materialization.
- Feed `PeakWindow`, periodic samples, WDDM guard observations, and disk counters into stage observations.
- Persist resolved plans and the hardware/version-keyed estimator cache atomically. The cache is advisory and must
  never be required for resume validation.

### Resident and streaming composition

- Call planning from `resident_workflow.py` before constructing `ResidentQuantizationRequest`.
- Remove the validation that rejects non-default `runtime.resources` and replace the CUDA-present `auto` choice with
  resource-plan selection.
- Pass concrete stage plans into `resident_quantization.py`; do not add more independent memory flags to the already
  large request type.
- Replan only at setup, post-load reconciliation, block/layer boundaries, or OOM rollback boundaries.
- Increment `RESIDENT_ALGORITHM_VERSION` if plan integration changes numerical execution or commit compatibility.

### Run state and artifacts

- Add the active memory-plan artifact/revision to `RunState` and the journal.
- Record resolved physical settings in tuning checkpoint identity so incompatible checkpoints cannot be adopted.
- Keep placement-only decisions out of content identities only after explicit equivalence tests prove that safe.

## 13. Observability

Add bounded structured events:

- `memory.plan_created`
- `memory.stage_admitted`
- `memory.stage_resized`
- `memory.observation_recorded`
- `memory.pressure_detected`
- `memory.plan_revised`
- `memory.oom_recovery_started`
- `memory.oom_recovery_exhausted`

Every decision event includes plan revision, stage signature, predicted/observed peak, hard target, reserve,
uncertainty, selected controls, rejected faster candidates, and reason. Existing `resource.sample`, boundary meters,
and OOM snapshot events remain canonical raw observations.

The run report should show planned versus actual peaks by stage and highlight underestimation. This directly closes
the still-open planned-versus-actual comparison in M5.19 and prevents a nominally "safe" plan from being accepted
without measured estimator accuracy.

## 14. Implementation sequence

### Phase 1: make limits and plans real

1. Introduce the new pure contracts and envelope math with table-driven unit tests.
2. Build metadata-derived fixed memory estimates for the current resident Gemma path.
3. Wire `runtime.resources` into resident preflight and persist a plan artifact.
4. Keep current explicit batches/placements fixed; initially use the plan only for refusal and reporting.

This phase is low-risk and immediately prevents starting known-impossible runs.

### Phase 2: adaptive forward and staging sizes

1. Treat block-forward, activation staging, quality, and evaluation batches/chunks as bounded adaptive controls.
2. Use the existing peak windows to compare predictions with observations.
3. Add prefetch/cache admission based on the complete next-stage peak, not tensor size alone.
4. Validate exact/approved parity and benchmark throughput across constrained synthetic envelopes.

These are no-gradient or already streamed paths and provide the safest initial performance gain.

### Phase 3: adaptive tuning microbatch

1. Preserve logical optimizer batch and schedule while resolving the largest safe physical microbatch.
2. Persist the resolved value before the first tuning step and bind checkpoints to it.
3. Prove uninterrupted/resumed equality for each supported microbatch and quantify cross-microbatch numerical spread.
4. Add post-block-refit-specific estimation because it owns the current resident high-water.

### Phase 4: placement transitions and OOM recovery

1. Integrate activation cache eviction, completed-block offload, pageable/mmap tiering, and cache release.
2. Add restartable resident-to-CPU-offload/streaming transitions where stage implementations support them.
3. Replace schema/helper action-name drift with typed fallback actions.
4. Add finite OOM injection tests and prove no completed work is repeated.

### Phase 5: learned estimator and throughput optimizer

1. Retain hardware/version-keyed estimator residuals and stage throughput observations.
2. Rank fitting candidates by measured throughput per byte.
3. Enable conservative in-run growth with hysteresis.
4. Promote adaptive mode only after designated-host 1B and 4B canaries show both fewer/no OOMs and a throughput
   improvement over the best safe fixed configuration.

## 15. Test strategy

### Pure policy tests

- hard ceiling, reserve, reusable allocator pool, external pressure, and fragmentation cases;
- monotone batch selection and deterministic tie-breaking;
- fixed workspace rejection at batch 1;
- host, pinned-host, WDDM shared, and disk commit-peak refusal;
- fallback ranking and finite retry exhaustion; and
- plan serialization, revision lineage, and semantic classification.

### CPU/tiny integration tests

- fake inventories force resident, CPU-offload, streaming, pageable, and mmap candidates;
- predicted stage lifetimes do not double-count disjoint tensors or omit handoff overlap;
- a changed envelope revises only the first incomplete unit;
- interrupted/resumed execution reuses the persisted plan;
- advisory profile-cache corruption is ignored safely; and
- fixed mode reproduces current artifacts exactly.

### CUDA tests

- constrained VRAM budgets select decreasing batches before OOM;
- injected OOM captures forensics, rolls back, revises once, and resumes without repeating committed work;
- tuning keeps logical optimizer steps and scheduler state identical under microbatch reduction;
- block/refit observations stay within the declared estimator allowance;
- no full activation stream becomes pinned; WDDM shared memory remains under its guard;
- completed-block offload removes block-depth growth when inline quality is disabled; and
- cache release decisions reduce reservation without changing allocated-tensor results.

### Real-model gates

Run the pinned Gemma 3 1B workload and the 4B bounded-memory canary on at least two device envelopes (native device
and an artificial lower ceiling). For each, compare adaptive with the best known fixed configuration:

- exact resolved recipe and memory-plan revisions;
- calibration, rank, BPW, factor, block-loss, tuning, and final-quality parity;
- OOM/retry count;
- wall time and stage throughput;
- peak/current allocated and reserved VRAM;
- peak host working set and WDDM dedicated/shared memory;
- temporary/durable disk and bytes transferred; and
- resume behavior after interruption before and after a memory-plan revision.

## 16. Acceptance criteria

- A run whose fixed minimum cannot fit is rejected before model materialization with required, available, reserve,
  and largest-allocation bytes.
- No adaptive decision changes an algorithmic setting without explicit policy and a visible semantic fork.
- On a device where tuning microbatch 8 fits, adaptive mode selects it (or demonstrates a faster admissible choice)
  instead of remaining at a conservative microbatch 1.
- Under an artificial lower ceiling, the same recipe selects a safe smaller physical plan and completes without
  uncontrolled paging or WDDM spill.
- A recoverable injected OOM produces one forensic record and one finite plan revision, then resumes from the latest
  valid boundary with no repeated committed layer/block.
- Planned versus observed stage peaks stay within a measured, published error bound on complete 1B and 4B runs.
- Adaptive mode improves or preserves wall time relative to the best safe fixed configuration on the same envelope;
  it is not accepted merely because it avoids OOM.
- The full pinned parity gates remain satisfied. Tiny fixtures or reduced-iteration probes alone are insufficient.

## 17. Recommended first deliverable

Implement Phase 1 plus read-only sizing for post-block refit first. It connects currently dead resource limits to the
production workflow, exposes the largest known memory peak, and produces planned-versus-actual evidence without yet
changing numerical execution. With that baseline in place, adaptive no-gradient batches and tuning microbatch can be
introduced as separately measurable changes rather than another set of opaque heuristics.
