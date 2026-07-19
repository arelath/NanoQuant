# Stacked Shared-Input Factorization

**Status:** Proposed design

**Primary evidence:** [Stacked Factorization](ImprovementSuggestions/StackedFactorization.md)

**Related design:** [Reconstruction-Informed Rank Planning](30-reconstruction-informed-rank-planning.md)

**Initial target:** The Q/K/V projections in every decoder block of pinned `google/gemma-3-1b-it`

## 1. Decision summary

Add a first-class **shared-input factor group** whose member matrices are concatenated along their output dimension and
factorized once:

```text
W_group = [W_q; W_k; W_v] ~= diag(s_post) U diag(s_mid) V diag(s_pre) + W_outlier
```

`V`, `s_pre`, and `s_mid` are owned once by the group. `U`, `s_post`, group-level outlier values, and any bias are
partitioned into Q, K, and V row slices. The planner treats the group as one allocation unit, the resident workflow
fits and commits it atomically, and deployment evaluates the group once before returning member views.

For the first Gemma-1B experiment:

- enable Q/K/V stacking explicitly in all 26 blocks;
- retain independent factorization for attention output and all MLP projections;
- use the existing diagonal activation-importance objective, with one shared input-importance vector and concatenated
  Q/K/V output-importance vectors;
- select outliers against the stacked residual, rather than combining three independently selected masks;
- fund each group from the exact bits that its three separate baseline projections would have consumed;
- require the logical artifact, packed artifact, PyTorch/CUDA runtime, GGUF converter, and modified llama.cpp reader to
  understand shared ownership before a stacked plan can be selected.

This is a representation and persistence change, not only a factorizer call-site change. Three independent layer
records that happen to contain equal copies of `V` would waste bits, make tuning ownership ambiguous, and fail to
represent the format that produced the measured gain.

## 2. Evidence and design implications

The retained study establishes the following initial policy:

- Q/K/V row stacking reduced combined reconstruction error by 11-24% against uniform separate ranks and by 4-14%
  against separately reallocated ranks at equal factor bits.
- All 26 Gemma-1B attention blocks improved; the aggregate attention error reduction was 14.9%, or about 1.2% over
  all quantized matrices before global rank reallocation.
- Combining stacking with global allocation produced roughly 8.5-9% lower global reconstruction error at equal bits.
- Activation-weighted fitting was much more important than unweighted fitting, reducing activation-space error by
  about 36-44% relative to unweighted separate factors in the tested cases.
- Gate/up stacking lost at both tested budgets. It is excluded from the initial implementation.
- A larger Gemma model showed block-dependent stacking value. The generic implementation must therefore describe
  candidates per block and must not encode “stack every same-input pair” as a universal rule.

The existing factorizer already applies input and output diagonal importance through whitening. The numerical ADMM
solver does not need a second weighted-factorization algorithm. The work is in topology, bit accounting, group state,
resident orchestration, tuning, persistence, and deployment.

## 3. Terminology

“Stacked factorization” in this document means **row stacking matrices that consume the same activation**. It does not
mean serial residual factorization or adding another low-rank stage.

- **Member:** a logical projection such as `q_proj`.
- **Candidate:** a model-adapter assertion that several members consume the same tensor and can be row-stacked.
- **Group:** one selected candidate in one model block.
- **Unit:** the indivisible planner and resident-workflow object: either one ordinary layer or one selected group.
- **Owner:** the module or frozen state that stores a group's tensors exactly once.
- **View:** a member projection that exposes an owner's output-row slice at the model's original path.

## 4. Goals and non-goals

### Goals

- Represent shared factors once from planning through GGUF deployment.
- Preserve the original model's Q, K, and V call sites and output shapes.
- Make group bit cost exact and comparable with the separate baseline.
- Keep the production diagonal Fisher/activation objective and concatenate its output importance correctly.
- Make fitting, tuning, freezing, commit, adoption, and resume atomic at group scope.
- Retain per-member reconstruction and activation-space metrics for diagnosis.
- Keep old independent-layer artifacts and runtimes readable.
- Prove the change with a complete pinned-Gemma compression and matched evaluation.

### Non-goals

- Automatically discovering shared-input topology from Python execution traces.
- Stacking gate/up in the initial policy.
- Sharing factors across decoder blocks.
- Allowing overlapping groups, partial groups, or different member input widths.
- Reusing one member's independently selected outlier mask.
- Treating duplicated right-factor tensors in three layer records as a valid shared format.
- Claiming model-quality improvement from reconstruction measurements alone.

## 5. Mathematical and bit contract

For members `W_j in R^(m_j x n)` with a proven common input, define:

```text
M       = sum_j m_j
W_group = concat_rows(W_0, ..., W_k) in R^(M x n)
```

The group approximation at rank `r` is:

```text
W_hat_group = diag(s_post) U diag(s_mid) V diag(s_pre) + W_outlier
U in {-1,+1}^(M x r)
V in {-1,+1}^(r x n)
```

Member `j` owns no factor tensor. Its result is the row interval `[offset_j, offset_j + m_j)` of the group output.

With the current logical accounting, the factor cost is:

```text
group_factor_bits(M, n, r)
  = r * (M + n) + scale_bits * (M + n + r)
```

Packing padding must be added by the packed-cost function, just as it is for an ordinary matrix. The equal-factor-bit
starting rank derived from separate member ranks `r_j` is the greatest aligned rank that fits:

```text
group_factor_bits(M, n, r_group)
  <= sum_j factor_bits(m_j, n, r_j)
```

Do not rely on the scale-free closed form from the study in production; the planner must include scales, alignment,
packing constraints, and any group outlier cost. Unspent alignment residue returns to the global allocation pool.

The group rank is bounded by `min(M, n)`, then aligned to the configured runtime rank quantum. For Gemma-1B Q/K/V,
`M` exceeds `n`, so the input width is the hard mathematical cap.

## 6. Topology declaration and selection

### 6.1 Adapter-owned candidates

Model-family knowledge remains in the infrastructure adapter. Extend the adapter contract with candidates such as:

```python
SharedInputGroupCandidate(
    name="attn_qkv",
    ordered_members=(q_path, k_path, v_path),
    execution_kind="parallel_projections",
)
```

The adapter is asserting semantic shared input, not merely equal `in_features`. Candidate member order is canonical and
defines row slices, tensor hashes, output-importance concatenation, export naming, and tie-breaking.

Candidate validation must reject:

- duplicate or overlapping members;
- members outside the same block;
- different input widths or incompatible devices/dtypes;
- unsupported bias semantics;
- a member selected by more than one unit;
- a requested candidate not declared by the adapter.

For the initial Gemma implementation, require either no member bias or concatenate all member biases into one group
bias. Mixed bias presence is rejected until a masked-bias contract is designed.

### 6.2 Explicit first-version adoption

Add an opt-in recipe configuration:

```yaml
shared_input_factorization:
  enabled: true
  selected_candidates:
    - pattern: "*.self_attn.attn_qkv"
  outlier_policy: group_residual
  require_deployment_support: true
```

Selection is explicit in version 1. The Gemma-1B evidence justifies choosing Q/K/V for every block; it does not justify
a universal heuristic. A later topology-probe mode can compare separate and grouped baseline fits for models with
variable benefit, but that is a separate experiment and would require extra ADMM work.

### 6.3 Immutable resolved topology

Persist a `QuantizationTopology` before reconstruction probing or production fitting. It contains an ordered partition
of all quantizable logical layers into ordinary-layer and shared-input-group units. Its identity includes:

- adapter and model identity;
- every canonical member path and shape;
- group names, member order, and row offsets;
- selected group/outlier policies;
- format and runtime capability versions.

The topology hash becomes a transitive input to the calibration objective, rank-probe profile, final plan, every
resident commit, logical and packed manifests, and export summary. A run cannot resume under a different topology.

## 7. Planning contracts

Replace the assumption that a `BlockPlan` contains only independent `LayerPlan` entries with a tagged unit contract:

```python
QuantizationUnitPlan = LayerUnitPlan | SharedInputGroupPlan

SharedInputGroupPlan(
    unit_id,
    name,
    members,          # ordered paths, shapes, row slices
    input_features,
    output_features,  # summed output width
    rank,
    outlier_count,
    exact_bit_cost,
    tuning_policy,
    retry_policy,
)
```

The plan must still expose a one-to-one logical-layer index for reports and model editing, but factor ownership and bit
allocation live at unit scope. A group rank is never copied into three member `LayerPlan`s.

Current positional layer schedules are fragile because the resident default order places V, O, Q, and K separately.
For grouped strategies, resolve an explicit ordered unit schedule per block. Q/K/V is one work unit; no member may be
processed before or after it independently. Existing ungrouped strategies retain their old ordering for compatibility.

The reconstruction-informed planner in `30-reconstruction-informed-rank-planning.md` is revised to allocate these
units. In Gemma-1B it probes 26 Q/K/V groups plus O, gate, up, and down for each block: 130 units covering the same 182
logical matrices.

## 8. Calibration and weighted objective

The current calibration hooks collect input activations and output gradients at logical-layer paths. For a selected
group:

1. Validate that every member's input-importance vector has the shared width and agrees within a configured numerical
   tolerance after normalization.
2. Select one canonical shared vector. The initial policy uses Q's vector after equality validation; it must not
   average materially different vectors and hide an invalid topology assumption.
3. Concatenate member output-importance vectors in canonical member/row order.
4. Pass the shared input and concatenated output vectors to the existing `factorize_admm` weighted objective.

Persist both the group objective and the member objective references. Calibration completeness validation requires all
members even though the production factorizer consumes one merged objective.

If shared input statistics differ beyond tolerance, fail preprocessing with member-level diagnostics. Falling back to
separate factors would silently change both topology and bit allocation and is therefore not allowed within a run.

## 9. Group outliers and residual accounting

An independently selected Q, K, or V outlier mask cannot be combined with one shared `s_pre`: zeroing a column in only
one member while retaining the shared factor contribution in the others is not expressible by the current format.

Run outlier selection once against `W_group` and its weighted objective. Persist:

- one group input-index vector;
- one stacked outlier-value matrix of shape `M x k`;
- one optional stacked outlier-scale tensor;
- member row slices for diagnostics and execution.

Group outlier bit cost is computed from `M`, not charged three times. Selection, zeroing, factor fitting, reconstruction,
and retry all use the same group residual convention.

The first tiny/CPU test path may run with zero outliers, but the pinned-Gemma promotion run must use the production
outlier policy. The study did not include the complete outlier pipeline, so its interaction is a required quality gate,
not an assumed benefit.

## 10. Resident factorization and tuning flow

### 10.1 Atomic materialization

When the group unit is reached:

1. Complete its configured non-factorized tuning while all members are still dense.
2. Materialize all current member weights together.
3. Concatenate weights, output importance, bias, and any group residual state in canonical order.
4. Fit and scale one group factorization.
5. Install the group owner and every member view in one editor transaction.
6. Run factorized tuning by selecting the owner parameters exactly once.
7. Freeze and commit the complete group atomically.

There is no valid state in which Q is grouped while K or V remains an independently processable dense or factorized
layer. Failures roll back the entire edit transaction.

### 10.2 Trainable owner and projection views

Add a parameter-owning `TrainableSharedInputFactorGroup` under a deterministic block-local registry, for example
`_nanoquant_factor_groups.attn_qkv`. It owns `left`, `right`, `scale_pre`, `scale_mid`, `scale_post`, group outliers, and
bias. Replace original projection modules with lightweight `SharedInputProjectionView`s carrying only owner identity and
row range.

The views must not register duplicate parameters or tensors in `state_dict`. During research-model execution, a view
may recompute the shared stage-1 operation for each member call. That is slower but gives simple, correct autograd and
avoids unsafe cross-call caches under checkpointing, reentrancy, microbatching, or exceptions. The deployment model
executes the owner once and slices all three outputs.

Replace exact class-name checks in tuning with explicit capabilities or a common factorized-module protocol. Otherwise
non-factorized tuning and global parameter selection will misclassify group views.

### 10.3 Factorized and post-block tuning

`tune_factorized` accepts a unit/owner identity rather than only a logical layer path. Parameter discovery must prove
that each owner tensor is selected once. Member view paths select no parameters.

Post-block refit and global distillation rehydrate one group owner plus member views and include the group scales,
outlier values, and bias according to the existing policy. Reports can attribute losses to member paths, but optimizer
ownership remains at group scope.

### 10.4 Retry

A retry changes the whole group's rank or outlier policy and is charged with group cost. It cannot independently grow
Q, K, or V. For the first fixed-budget stacked-allocation experiment, disable opportunistic retry so the measured result
matches the precomputed plan; retain group-aware retry support for later workflows.

## 11. Frozen state, commits, and resume

Add first-class contracts rather than overloading `FrozenNanoQuantState`:

```python
FrozenSharedInputFactorGroupState(
    group_id,
    ordered_members,
    row_slices,
    left,
    right,
    scale_pre,
    scale_mid,
    scale_post,
    outlier_indices,
    outlier_values,
    outlier_scale,
    bias,
)

SharedInputGroupResult(
    plan,
    frozen_state,
    group_metrics,
    member_metrics,
    tuning_metrics,
)
```

A group commit envelope contains all group tensor descriptors and all member metrics. Journal events use `unit_id` and
emit one started/completed pair. Adoption verifies topology hash, member order/shapes, source hashes for every member,
objective hashes, rank, seeds, algorithm version, and every transitive tensor hash.

Interruption before the atomic group commit leaves no adoptable partial member state. Interruption after it rehydrates
the owner and all views together. Orphan discovery must never infer three layer commits from one group commit.

This changes the resident semantic algorithm and persisted state. Increment `RESIDENT_ALGORITHM_VERSION` when the code
lands; old independent commits must not be adopted into a stacked run.

## 12. Logical and packed artifact formats

The current formats assume one layer owns five factor roles and that tensor keys are unique. Safetensors keys cannot
express storage aliasing, so writing the shared right factor under Q, K, and V would physically duplicate it.

Introduce a backward-readable format revision with:

- an ordinary `layers` collection for independent factors;
- a `shared_input_groups` collection whose entry owns tensors once;
- member records containing logical path, output shape, and row slice only;
- explicit layout/order and topology hashes;
- exact logical and packed bit totals at owner scope.

Packing produces one packed left tensor of shape `M x r`, one packed right tensor, and one set of scales/outliers for
the group. Add a `PackedProjectionGroupSpec` alongside `QuantizedLinearSpec`; do not pretend that a group is three
packed linears.

Readers continue accepting existing format-version-1 artifacts. Writers use the new version only when the plan contains
a group. Validation rejects duplicate membership, overlapping slices, uncovered group rows, inconsistent tensor shapes,
and manifests that charge shared tensors more than once.

## 13. PyTorch and CUDA runtime

The CUDA backend already groups **independent** Q/K/V stage-1 calls by concatenating three right factors. That is a
runtime optimization, not shared factorization, and remains the compatibility path for old artifacts.

For a true shared group, ordinary NanoQuant stage 1 and stage 2 can evaluate the stacked matrix directly; no new
numerical kernel is required for correctness. Extend the prepared attention module to detect a Q/K/V group, execute one
packed group projection in both prefill and decode, then slice Q, K, and V using the persisted row ranges.

Tests must cover:

- group output versus concatenated member-view outputs;
- CPU reference versus CUDA for zero and production outlier counts;
- prefill and decode shapes, GQA Q/K/V widths, batching, and non-contiguous input tensors;
- backward compatibility with independently factorized Q/K/V;
- proof that the packed group owns one right factor and launches one logical group operation.

## 14. GGUF and modified llama.cpp

The current converter emits NanoQuant sidecars for separate canonical Q, K, and V prefixes, while the modified
llama.cpp graph can fuse their independent stage-1 computations. A shared group needs distinct ownership.

Add a canonical fused-QKV NanoQuant sidecar family and metadata that records Q/K/V row extents. The converter must:

1. detect a shared group in the logical/packed manifest;
2. apply the architecture-required Q and K row permutations within the corresponding stacked row slices;
3. write one left/right/scale/outlier sidecar set without duplication;
4. preserve member dimensions and ordering in metadata;
5. include group bytes in export-summary and BPW accounting.

Extend the llama.cpp loader to associate that sidecar with a block-level QKV group. The graph evaluates one NanoQuant
linear and creates Q/K/V views, analogous to the existing dense `wqkv` branch. The existing separate NanoQuant fused
stage-1 path remains unchanged for older GGUFs.

An export target that does not advertise the required group format/graph capability must fail plan validation or
export. Expanding one group into three duplicate sidecars is not an acceptable fallback because it changes BPW and the
runtime representation.

Add converter inspection tests, llama.cpp tensor-loading tests, CPU graph parity, CUDA parity, and end-to-end GGUF
generation/load/evaluation before promotion.

## 15. Metrics and reports

For every group, retain:

- separate-baseline funded bits and rank;
- selected group rank, logical bits, packed bits, and alignment residue;
- weighted and unweighted group reconstruction error;
- member-slice reconstruction errors and norms;
- group and per-member activation-space error;
- factorization, scale-fit, tuning, and runtime timings;
- outlier count and residual contribution;
- measured improvement against protocol-matched separate factors.

Model summaries must count 182 logical projections but only 130 factor owners for the initial Gemma-1B topology. BPW
uses physical owner bytes. Reports should make this distinction explicit so group sharing does not look like missing
layers or free duplicated state.

## 16. Failure policy

Fail closed on topology or representation inconsistencies. In particular, do not silently fall back to separate
factorization after the rank plan is fixed, because the separate units have different cost and response curves.

A group can be disabled only by creating a new topology and rank plan. Valid old artifacts remain immutable evidence.
If a full run shows an end-quality regression, retain its profile and outputs, compare member/block losses and outlier
behavior, then change the next run's explicit topology or policy.

## 17. Testing and validation gates

### Unit and property tests

- Group bit cost equals physical tensor and scale accounting and is monotonic in aligned rank.
- Equal-budget rank never exceeds the sum of member baseline costs.
- Member slices exactly partition `[0, M)` in canonical order.
- Weighted stacked factorization equals direct use of concatenated output importance and shared input importance.
- Group outlier reconstruction matches the dense stacked reference.
- Owner parameters occur once in module traversal, state dicts, tuning selection, and optimizer groups.
- Frozen/trainable round trips reproduce group and member outputs.

### Contract and workflow tests

- Adapter topology is the only source of shared-input candidates.
- Architecture dependency direction remains valid.
- Unit scheduling never processes a member independently.
- Injected failure at every group phase either leaves dense members intact or one complete adoptable group commit.
- Resume rehydrates exactly one owner and all member views.
- Changing topology, member order, objective, source, rank, format capability, or algorithm version invalidates reuse.
- Logical and packed format readers remain backward compatible.

### Pinned-Gemma gate

1. Produce a protocol-matched separate baseline and stacked Q/K/V rank-probe profile at equal total target bits.
2. Confirm all 26 attention groups improve the combined production-objective reconstruction error, or document and
   explicitly review any exception before continuing.
3. Run the complete resident workflow including outliers, non-factorized tuning, factorized tuning, post-block refit,
   global policy, validation, logical and packed export, checkpoint, and GGUF.
4. Run matched frozen-PyTorch, packed CUDA, and modified llama.cpp parity checks.
5. Run the retained WikiText-2 limited protocol and required task evaluations against BF16, legacy, and the nearest
   separate NanoQuant baseline.
6. Compare rank, BPW, quality, block/layer losses, wall time, peak GPU/host memory, artifact bytes, and resume behavior.

Reconstruction improvement is necessary evidence for this feature but is not the promotion gate. End-model quality,
physical BPW, runtime parity, and complete export are mandatory.

## 18. Code changes by boundary

### Domain and configuration

- `src/nanoquant/domain/models.py`: add topology/unit identifiers, group plan/result DTOs, member row slices, and frozen
  group state. Preserve ordinary `LayerPlan` and `FrozenNanoQuantState` for ungrouped artifacts.
- `src/nanoquant/domain/planning.py`: add group factor/outlier cost and allocate `QuantizationUnitPlan` variants.
- `src/nanoquant/config/schema.py`, `validation.py`, and codecs: add explicit selected-candidate configuration,
  group-unit schedules, capability requirements, canonical round trips, and semantic hashing.

### Model and application boundaries

- Extend the model-adapter port and Hugging Face adapter definitions to declare shared-input candidates; generic
  application code must not infer Gemma Q/K/V paths.
- `src/nanoquant/application/calibration.py`: validate shared input statistics and create concatenated group output
  objectives while retaining member provenance.
- `src/nanoquant/application/planning.py`: resolve topology before allocation and plan ordinary/group units.
- `src/nanoquant/application/layers.py`: add trainable/frozen group owners, member views, atomic editor operations, and
  freeze/rehydration support.
- `src/nanoquant/application/tuning.py` and `src/nanoquant/global_distillation.py`: select owner parameters once and recognize
  factorized capabilities instead of relying on the current exact trainable-linear class name.

### Resident workflow and evidence

- `src/nanoquant/resident_quantization.py`: iterate the resolved unit schedule, materialize group members together,
  fit/tune/freeze/commit atomically, restore group owners, emit group/member metrics, and increment the resident
  algorithm version.
- Extend journal, commit-envelope, descriptor, validator, cleanup, and report traversal to treat a group as one owner
  with multiple logical members.

### Logical, packed, and deployment surfaces

- Revise the logical artifact and packed artifact schemas/loaders/validators to own group tensors once.
- `src/nanoquant/runtime/backend.py`: add a projection-group spec alongside `QuantizedLinearSpec`.
- `src/nanoquant/runtime/torch_model.py`: bind the group to prepared attention and slice one fused output.
- `src/nanoquant/runtime/cuda_backend.py`: execute the stacked owner with the ordinary packed-linear primitive; keep
  `CudaProjectionGroup` for legacy independent-factor fusion.
- Update `src/nanoquant/runtime/llamacpp.py`, GGUF export infrastructure, and
  `tools/llamacpp/convert_nanoquant_to_gguf.py` for one fused sidecar owner and Q/K row permutation within slices.
- Update the modified `D:\dev\research\llama.cpp` loader/model/graph plumbing. Its current Q/K/V NanoQuant branch
  fuses stage 1 across three independent factor states; the new branch consumes one stacked state and returns views.

All changes must preserve the dependency direction enforced by `tests/contract/test_architecture.py`.

## 19. Implementation sequence

1. Add adapter candidates, resolved topology contracts, group bit accounting, and pure validation tests.
2. Generalize planning from layers to units and update reconstruction-informed rank planning.
3. Add calibration-objective merging and stacked factorization/outlier tests using direct tensor references.
4. Implement trainable/frozen owners, member views, editor transactions, and tuning parameter discovery.
5. Add atomic group results, commits, journal/resume/adoption, algorithm-version bump, and validator traversal.
6. Revise logical and packed schemas, packing, loading, BPW, reports, and backward-compatibility tests.
7. Add prepared PyTorch/CUDA execution and parity/performance tests.
8. Add GGUF conversion, llama.cpp loading/graph execution, and inspection/runtime tests.
9. Run a tiny complete workflow, then the full pinned-Gemma stacked/reconstruction-aware experiment and quality gate.

Do not start the costly full fit until topology resolution, the all-unit probe pass, exact allocation, deployment
capability checks, and final plan persistence have all succeeded.

## 20. Deferred extensions

- Topology probing and evidence-driven per-block group selection for larger model families.
- Other same-input projection groups supported by model adapters and measured at equal physical bits.
- A fused autograd research module that computes Q/K/V once without unsafe transient caching.
- Group-specific outlier allocation and response curves learned from complete production runs.
- Specialized CUDA kernels if profiling shows the ordinary stacked linear leaves material performance on the table.
