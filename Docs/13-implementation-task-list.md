# Rewrite Implementation Task List

This checklist converts the [delivery roadmap](11-delivery-roadmap.md) into executable work. Task IDs are stable references for issues, pull requests, run reports, and design reviews.

## How to use this checklist

- Check a task only when its implementation, tests, and required documentation/evidence are complete.
- A milestone is complete only when every required task and its exit gate are checked.
- Tasks may be split into smaller issues, but the parent task ID remains the traceability anchor.
- Algorithm changes should not be mixed into parity tasks unless the task explicitly calls for them.
- Every performance task must name its workload and baseline; every quality task must name its evaluator and baseline.
- The current codebase is `D:\dev\research\NanoQuant-OfficalCode\`.
- The modified NanoQuant llama.cpp reference is `D:\dev\research\llama.cpp\`.
- Its CUDA implementation is `D:\dev\research\llama.cpp\ggml\src\ggml-cuda\nanoquant.cu`.

## Milestone 0 — Freeze current evidence and scope

Dependencies: none  
Outcome: the rewrite has trustworthy parity, quality, and performance baselines.

- [x] **M0.1** Record the NanoQuant repository revision, dirty-worktree patch, environment, package versions, and supported hardware baseline.
- [x] **M0.2** Record the modified llama.cpp revision, dirty patch, build configuration, conversion script revision, and `nanoquant.cu` content hash.
- [ ] **M0.3** Select and document the reference development GPU, large-model host, stable performance host, and expected power/clock configuration.
- [x] **M0.4** Freeze the exact current configuration for Experiment 019, including every effective default that is currently spread across config representations.
- [x] **M0.5** Preserve `outputs/019-phase1-weight-errors.md` and its CSV as golden reporting inputs.
- [x] **M0.6** Capture at least three layer fixtures: attention, MLP, and one historically difficult reconstruction layer.
  The immutable references and old/new objective results for block-0 `self_attn.q_proj`, block-1 `mlp.gate_proj`,
  and the difficult block-0 `mlp.down_proj` are retained in
  `evidence/m2/gemma-diagonal-objective-parity.json`; the capture reuses content-addressed tensors instead of
  duplicating the large matrices.
- [ ] **M0.7** Capture at least two block fixtures from different model depths, including inputs, teacher targets, statistics, weights, and accepted results.
- [ ] **M0.8** Capture one deterministic tiny-model end-to-end legacy run.
- [ ] **M0.9** Capture one representative 1B legacy quantization run with stage timings, peak GPU/host memory, temporary storage, final artifact size, and evaluation results.
- [ ] **M0.10** Reproduce and profile the current research/test inference workload near the observed 20 tokens/second.
- [ ] **M0.11** Reproduce and profile the intended modified llama.cpp reference workload associated with the approximately 400 tokens/second observation.
- [ ] **M0.12** Reconcile the two inference protocols: model artifact, BPW, prompt length, decode length, batch, sampling, KV cache, hardware, warm-up, and timed boundaries.
- [ ] **M0.13** Capture llama.cpp kernel, layer/block, prefill, decode, and end-to-end benchmark JSON for the agreed comparison workloads.
- [ ] **M0.14** Capture representative Nsight or equivalent traces for both current runtimes and account for at least 90% of wall time.
- [x] **M0.15** Inventory every current configuration field and confirm that each has a destination or explicit removal in the migration map.
- [x] **M0.16** Inventory supported model families, checkpoint variants, datasets, evaluation tasks, packing formats, and CUDA architectures.
- [x] **M0.17** Decide whether DBF remains supported, becomes research-only, or is deprecated; record the decision in an ADR.
- [x] **M0.18** Decide which calibration modes and Hessian representations are productized versus experimental; record the decision.
- [x] **M0.19** Decide the compatibility policy for existing `.pt`, packed extension, and GGUF artifacts.
- [x] **M0.20** Create a requirements-to-milestone traceability table covering every requirement ID in `01-requirements.md`.
- [ ] **M0.GATE** Verify that legacy fixtures, evaluation baselines, Experiment 019 reports, and both inference profiles can be reproduced by another maintainer from recorded instructions.

## Milestone 1 — Establish configuration, run, event, and artifact foundations

Dependencies: Milestone 0 baseline formats identified  
Outcome: even legacy-backed work can run through the new auditable shell.

- [x] **M1.1** Create the proposed package boundaries: `domain`, `application`, `ports`, `infrastructure`, `runtime`, `config`, and `cli`.
- [x] **M1.2** Add automated forbidden-import/dependency-direction checks from `02-architecture.md`.
- [x] **M1.3** Implement every frozen nested dataclass in `03-configuration-reference.md` as the sole canonical configuration schema.
- [x] **M1.4** Implement enum, tuple, optional, and nested-dataclass decoding from YAML and JSON.
- [x] **M1.5** Reject unknown configuration paths with full-path errors and nearest-name suggestions.
- [x] **M1.6** Implement canonical configuration serialization with deterministic field/value encoding.
- [x] **M1.7** Implement sparse schema-aware CLI overrides without declaring duplicate defaults.
- [x] **M1.8** Implement pre-resolution, resolved, and planned validation phases with stable diagnostic codes.
- [x] **M1.9** Implement immutable resolution of source revisions, tokenizer revisions, and `auto` executor/storage choices using the same `RunConfig` type.
- [x] **M1.10** Implement legacy flat-config migration and require every legacy field to be mapped, removed explicitly, or rejected.
- [x] **M1.11** Generate CLI help and configuration reference data from the canonical schema.
- [x] **M1.12** Implement `IntentConfig.experiment_number` and validation against a numbered launcher filename.
- [x] **M1.13** Implement the thin numbered zero-argument runfile adapter calling the shared application service.
- [x] **M1.14** Add a `000_experiment_template.py` showing the approved no-argument runfile pattern without consuming a research experiment number.
- [x] **M1.15** Add static tests prohibiting `argparse`, copied orchestration, and direct infrastructure imports in numbered runfiles.
- [x] **M1.16** Implement run IDs, run lifecycle states, parent/fork relationships, and active-process leases.
- [x] **M1.17** Implement `LauncherProvenance`, including kind, experiment number, repository-relative path, content hash, revision, and arguments.
- [x] **M1.18** Implement the versioned `RunManifest` and run-directory layout.
- [x] **M1.19** Implement allowlisted environment capture with credential/secret redaction.
- [x] **M1.20** Implement the structured event envelope, monotonic run sequence numbers, spans, and local JSONL sink.
- [x] **M1.21** Implement concise console rendering from structured events without parsing or duplicating messages.
- [x] **M1.22** Implement stable warning/diagnostic code registration with documentation links.
- [x] **M1.23** Implement the local content-addressed artifact store and common artifact descriptor.
- [x] **M1.24** Implement temporary leases, hashing, validation, atomic commit descriptors, and cleanup of abandoned writes.
- [x] **M1.25** Implement stage-specific semantic cache keys and visible reuse/invalidation explanations.
- [x] **M1.26** Add a compatibility adapter that invokes the legacy pipeline while producing new manifests, events, and artifact references.
- [x] **M1.27** Render a minimal completed/failed/interrupted run report from structured data only.
- [x] **M1.28** Add config parity, schema migration, event ordering, secret redaction, artifact corruption, and atomic-write tests.
- [ ] **M1.GATE** Run a legacy-backed numbered experiment with no arguments and verify canonical resolved config, launcher provenance, structured events, artifact commits, and a self-contained report.

## Milestone 2 — Extract and verify the mathematical domain

Dependencies: Milestone 0 fixtures; Milestone 1 artifact/test foundations  
Outcome: core NanoQuant mathematics is pure, typed, replayable, and parity-tested.

- [x] **M2.1** Implement canonical `BlockId`, `LayerId`, `TensorId`, `ComponentRef`, `ArtifactRef`, `TensorSpec`, and `TensorRef` types.
- [x] **M2.2** Implement canonical model/dataset identities and model inventory domain objects.
- [x] **M2.3** Implement `BitCost`, exact logical/storage accounting helpers, and reconciliation tests.
- [x] **M2.4** Extract raw, per-element, objective-weighted, normalized, and staged export reconstruction metrics into pure functions.
- [x] **M2.5** Extract diagonal reconstruction objective and verify legacy parity on captured layers.
  `tools/compare_diagonal_objective.py` extracts the exact legacy `_weighted_weight_error` function from the
  source AST and compares it with the pure typed objective. On the three pinned Gemma fixtures, all weighted error,
  target-norm, and normalized-error values pass `atol=1e-6, rtol=2e-6`; the maximum observed relative difference
  is `9.87e-8`. Unit coverage also locks the legacy `1e-12` importance floor for zero-valued statistics.
- [x] **M2.6** Extract dense-Hessian objective, whitening/unwhitening, regularization, and triangular-solve behavior.
- [x] **M2.7** Define the block-diagonal and low-rank-plus-diagonal objective contracts, even if optimized implementations arrive in Milestone 5.
- [x] **M2.8** Extract uniform rank allocation and exact BPW budgeting.
- [x] **M2.9** Extract sensitivity/utility allocation, rounding, floor/ceiling, and edge-boost policies.
- [x] **M2.10** Extract rank retry scoring, caps, attempt limits, and global extra-bit budget as a pure policy.
- [x] **M2.11** Extract Fisher outlier selection and its bit accounting.
- [x] **M2.12** Extract residual-probe outlier selection and isolate its factorizer dependency.
- [x] **M2.13** Extract outlier removal/reconstruction and BF16/FP16/INT8 storage behavior.
- [x] **M2.14** Extract NanoQuant ADMM solve steps, schedules, convergence metrics, and deterministic generator use.
- [x] **M2.15** Extract DBF only if retained by M0.17, with an explicit component/version and parity expectations.
  ADR 0006 made DBF research-only and explicitly permits omission from the first release. The rewrite therefore does
  not identify DBF as NanoQuant ADMM and returns the stable `CAL004` unsupported-mode diagnostic, covered by tests.
- [x] **M2.16** Extract scale-pre/mid/post fitting, objective comparison, protected outlier columns, and rollback.
- [x] **M2.17** Implement `MaterializedFactorizationInput/Output` for pure in-stage tensor computation.
- [x] **M2.18** Implement persisted `FactorizationRequest`, `FactorizationResult`, convergence, scale, outlier, retry, tuning, and layer-result DTOs.
- [x] **M2.19** Implement separate `TrainableNanoQuantState` and `FrozenNanoQuantState` with validated conversion.
- [x] **M2.20** Ensure domain functions do not print, access files, traverse models, consult global configuration, or mutate caller tensors.
- [x] **M2.21** Add small deterministic CPU unit tests for every mathematical component and boundary case.
- [x] **M2.22** Add property tests for pack-independent reconstruction, retry monotonicity, budgets, scaling invariants, and deterministic logical seeds.
- [x] **M2.23** Run old/new layer-fixture comparisons and document all numerical tolerances or intentional differences.
  `evidence/m2/gemma-admm-factorization-parity-native-v25.json` first isolated the intentional native-orientation
  difference: tall MLP factors were exact, while wide attention/down-projection factors differed despite objective
  deltas of only `1.68e-5` and `4.78e-4`. The explicit transposed replay policy restores the legacy wide-matrix
  solve; `gemma-admm-factorization-parity.json` reports exact old/new equality for every latent factor, binary factor,
  scale, reconstruction, and final RNG state on all three fixtures, with zero objective delta. The native policy
  remains the production default because it is the one validated by the full system trajectory.
- [x] **M2.GATE** Verify captured layer fixtures reproduce accepted legacy factors/metrics within approved tolerances using only typed requests, tensor artifacts, and domain components.
  The source-hashed legacy oracle, typed committed layer results, immutable tensor references, pinned Gemma revision,
  factorization protocol, environment, per-tensor comparisons, and pre-fix intentional-difference record are retained
  under `evidence/m2/`. All explicit transposed-replay factor and metric comparisons are exact.

## Milestone 3 — Implement model sources, adapters, datasets, and calibration

Dependencies: Milestones 1–2  
Outcome: architecture-specific behavior is isolated and calibration emits portable artifacts.

- [x] **M3.1** Implement the `ModelSource` port for config/tokenizer metadata and tensor inventory without weight materialization.
- [x] **M3.2** Implement safe sharded-safetensors lookup, tensor shape/dtype inspection, direct reads, memory mapping, and hash verification.
- [x] **M3.3** Implement the `ModelAdapter` contract for block inventory, source-key mapping, block construction/loading, quantizable layers, prefix, block, suffix, and LM head.
- [x] **M3.4** Implement the Llama-compatible adapter.
- [x] **M3.5** Implement the Gemma/Gemma 3 adapter, including text-stack and position/attention metadata behavior.
- [x] **M3.6** Implement the Qwen adapter.
- [x] **M3.7** Implement the OPT adapter if retained in the supported-family decision.
- [x] **M3.8** Add explicit unsupported-variant diagnostics instead of best-effort architecture guessing.
- [x] **M3.9** Create an offline deterministic tiny causal-transformer adapter/fixture for integration tests.
- [x] **M3.10** Implement versioned dataset-source, mixture, formatting, selection, tokenization, and fingerprint services.
- [x] **M3.11** Pin tokenizer revision, chat template, BOS/EOS/padding behavior, and selected sample identities.
- [x] **M3.12** Implement CUDA, pinned-RAM, and pageable-RAM activation stores behind one contract.
- [x] **M3.13** Implement model-prefix input capture without permanent module replacement or exception-based hidden control flow.
- [x] **M3.14** Implement typed calibration accumulators and portable `CalibrationStats` artifacts.
- [x] **M3.15** Implement online Fisher calibration with current behavior documented and tested.
- [x] **M3.16** Implement two-phase Fisher calibration and deterministic partition/order behavior.
- [x] **M3.17** Implement forward-only calibration for low-resource execution.
- [x] **M3.18** Implement the retained DBF/other calibration modes or emit explicit unsupported diagnostics.
- [x] **M3.19** Implement objective builders producing per-layer `ObjectiveSpec` artifacts from calibration statistics.
- [x] **M3.20** Implement calibration OOM policies as explicit finite fallbacks with events and plan revisions.
- [x] **M3.21** Add common adapter contract tests for complete tensor mapping, block ordering, prefix/block/suffix parity, tied weights, and streamed loading.
- [x] **M3.22** Add calibration parity/stability tests, batch-partition invariance tests, and cached-versus-uncached equivalence tests.
- [x] **M3.GATE** Calibrate the tiny model and one supported 1B model through the new adapters and produce validated, replayable statistics/objective artifacts without using legacy traversal helpers.

## Milestone 4 — Build the resident pipeline, replay workflow, and resume semantics

Dependencies: Milestones 1–3  
Outcome: a complete 1B-class run uses the new pipeline and survives interruption.

- [x] **M4.1** Implement the generic typed `Stage[Request, Result]` interface, stage registry, semantic key, resource estimate, execute, and validate lifecycle.
- [x] **M4.2** Implement `StageContext` with executor, artifact/tensor stores, event sink, cancellation, and no global full config.
- [x] **M4.3** Implement the resident executor, device scopes, tensor leases, buffer reuse, and deterministic release without hot-loop allocator clearing.
- [x] **M4.4** Implement planning from model inventory, calibration artifacts, allocation config, outliers, objectives, and exact bit cost.
- [x] **M4.5** Emit and validate complete `QuantizationPlan`, `BlockPlan`, and `LayerPlan` artifacts before model mutation.
- [x] **M4.6** Implement source/working block construction through the adapter.
- [x] **M4.7** Implement non-factorized tuning as an independent typed service.
- [x] **M4.8** Implement outlier-selection stage and artifact commit.
- [x] **M4.9** Implement factorization-attempt stage, scale-fit stage, reconstruction evaluation, and attempt events.
- [x] **M4.10** Implement the pure retry-decision loop and update retry budget only after accepted layer commit.
- [x] **M4.11** Implement factorized tuning and best-state restoration as an independent service.
- [x] **M4.12** Implement post-block refit as an optional independent service.
- [x] **M4.13** Implement `LayerFreezer`, `BlockEditor`, and explicit installation of frozen layer state into the stage-owned working block.
- [x] **M4.14** Record source-reference, block-entry, after-each-layer, post-refit, and final-frozen-pre-KD loss snapshots.
- [x] **M4.15** Commit immutable `LayerResult` after each accepted layer.
- [x] **M4.16** Commit immutable `BlockResult`, frozen block state, and teacher/compressed next-block activation generations atomically.
- [x] **M4.17** Implement `RunState`, `ProgressCursor`, `BudgetState`, and journaled committed-artifact references.
- [x] **M4.18** Implement logical seed derivation from run seed, stage, block, layer, and attempt.
- [x] **M4.19** Implement resume validation, latest-valid-commit discovery, and restart of the first incomplete unit.
- [x] **M4.20** Implement fork semantics and a visible upstream-reuse/downstream-invalidation plan.
- [x] **M4.21** Inject failure before/after every layer/block commit operation and verify equivalence with uninterrupted controls. Factorized tuning now also commits a bounded, safe safetensors resume generation after every epoch; a fresh-process tiny integration test interrupts at that boundary and reproduces the uninterrupted frozen states exactly.
- [x] **M4.22** Implement `capture-layer`, `replay-layer`, `capture-block`, and `replay-block` using canonical fixture artifacts.
- [x] **M4.23** Implement `FrozenModelResult` assembly from block artifacts without requiring a mutable full model object.
- [x] **M4.24** Render the Experiment 019-style per-layer reconstruction and final-block-pre-KD tables from structured results.
- [x] **M4.25** Run the deterministic tiny end-to-end pipeline entirely on new components.
- [x] **M4.26** Run a representative 1B resident quantization and compare factors, BPW, block losses, quality, memory, and time with the legacy baseline.
- [x] **M4.27** Meet the captured-layer under-60-second and tiny-model under-10-minute feedback targets where the reference hardware/factorizer settings permit them.
- [x] **M4.28** Implement memory-bounded top-k model-level distillation, durable per-epoch teacher-cache and optimizer resume, immutable tuned-state persistence, atomic activation, and frozen-loader integration with a complete tiny-model test.
- [x] **M4.29** Run the pinned 1B eight-epoch model-level KD protocol, evaluate the tuned artifact, and compare its quality/memory/time with a protocol-matched legacy result. The versioned legacy Python/device-RNG cache sampler completed 2,048 steps at 2.70 GB peak allocated CUDA memory and reached exact serial PPL 454.431. A fresh legacy run over the same pinned token tensor reached PPL 444.333; its eight KD losses end at 2.1430 versus rewrite 2.1404, establishing optimization-boundary parity despite the accepted 2.27% end-to-end numerical-realization spread.
- [x] **M4.GATE** Complete, interrupt, resume, replay, and compare a 1B resident run with no legacy quantization/orchestration dependency and approved parity differences only. Contemporary legacy and rewrite match all 182 ranks (rank sum 105,856 and therefore identical binary BPW/outlier-count cost), all 26 post-refit block boundaries within -2.20% to +2.01%, KD objectives within 0.12% at the final epoch, and exact serial PPL within 2.27%. Historical Experiment 018's lower trajectory is not reproducible under the current pinned CUDA/CCE environment.

## Milestone 5 — Add bounded-memory streaming and 70B scaling

Dependencies: Milestone 4 resident correctness  
Outcome: the same pipeline operates from GPU-resident 1B models through disk-streamed 70B models.

- [x] **M5.1** Implement host inventory and resource planning for source, output, block weights, factors, Hessians, activations, tuning state, and temporary disk.
- [x] **M5.2** Add safety margins and refuse execution before expensive work when declared memory/disk minima cannot be met.
- [x] **M5.3** Implement block-aligned source streaming directly from sharded safetensors without constructing a full `state_dict`.
- [x] **M5.4** Implement a memory-mapped activation store with preallocated files, batched reads/writes, hashes, and atomic generation commit.
- [x] **M5.5** Implement automatic activation-tier selection across CUDA, pinned RAM, pageable RAM, and mmap from the resource plan.
- [x] **M5.6** Implement double-buffered activation propagation and bounded batch staging.
- [x] **M5.7** Generate teacher block outputs by loading one original block, then release it before/while processing the working block as allowed by the plan.
- [x] **M5.8** Write frozen/packed block shards incrementally so no complete quantized model state must reside in RAM.
- [ ] **M5.9** Implement source-block/layer prefetch with explicit tensor/buffer leases and measured benefit.
- [x] **M5.10** Implement forward-only streamed calibration.
- [ ] **M5.11** Implement streamed forward/backward calibration with boundary activation commits and block recomputation when Fisher statistics are required.
- [x] **M5.12** Implement block-diagonal covariance objective storage/execution.
- [x] **M5.13** Implement low-rank-plus-diagonal covariance objective storage/execution.
- [x] **M5.14** Enforce dense-Hessian per-layer workspace reservations and reject/explicitly fall back when dimensions exceed policy.
- [x] **M5.15** Implement finite OOM fallback actions for batch size and activation tier; distinguish algorithm-preserving from algorithm-changing fallbacks.
- [x] **M5.16** Implement disk-full, corrupt-shard, changed-source, and interrupted-activation-generation recovery behavior.
- [x] **M5.17** Add a constrained-resource integration test that forces mmap activation storage on a tiny model.
- [ ] **M5.18** Verify resident and streaming executors produce equivalent tiny/1B results within approved tolerances.
  The tiny half is covered by an exact two-block comparison using full-batch resident execution versus
  batch-bounded streaming through committed mmap activation generations; the required 1B comparison remains open.
- [ ] **M5.19** Compare planned versus actual peak GPU, host, temporary disk, and I/O use and set estimator error thresholds.
  Actual block commits now retain host high-water and CUDA reserved high-water instead of zero/allocated-only
  values, and opt-in profiles separate host working/private memory, CUDA allocated/reserved memory, device-wide
  pressure, allocator churn, and I/O bytes. `tools/validate_resident_run.py` now aggregates validated committed
  ranks, bit-cost categories/effective BPW, source parameter count, block losses/wall time, memory high-water, and
  artifact bytes for the final planned/actual report. Protocol-matched comparisons and acceptance thresholds remain
  required before closing this item.
- [ ] **M5.20** Implement the distributed-executor port and decide which distributed calibration/tuning operations are required for the first release.
- [ ] **M5.21** Run a 70B metadata-only plan and validate source/output/storage estimates before weight execution.
- [ ] **M5.22** Run a real large-model selected-block canary using disk-backed state and resume it after intentional interruption.
- [ ] **M5.23** Run a complete 70B streaming quantization when hardware/time budget permits and record cost, failures, throughput, and quality.
- [ ] **M5.GATE** Demonstrate bounded peak model-weight memory proportional to the active block/workspace, valid incremental artifacts, and resume on a real large-model canary.

## Milestone 6 — Build the deployment runtime and correctness reference

Dependencies: Milestone 2 frozen logical state; Milestone 4 artifacts  
Outcome: packed artifacts load and generate correctly without research dependencies.

- [ ] **M6.1** Split the deployment runtime into a separately installable/importable surface without datasets, calibration, ADMM, optimizers, or experiment orchestration.
- [x] **M6.2** Finalize and version the backend-independent frozen logical NanoQuant format. `nanoquant-v1`
  now has a deployment-owned operation specification and validator covering factor shapes/rank/signs/dtypes,
  pre/mid/post scales, bias, paired salient indices/values, optional quantized-outlier scales, finite values,
  canonical names, and exact tensor inventories. Schema-v1 logical artifacts use a bounded JSON descriptor,
  one immutable safetensors shard per block, file hashes/sizes, header-only inspection, atomic creation, and
  per-layer lazy loading. Full shared/model-shell/tokenizer export remains open rather than being implied here.
- [x] **M6.3** Finalize and version the first CUDA packed layout, including padding, alignment, scale, outlier, bias,
  and tensor-name metadata. `llama.cpp-i32-lsb-v1` is specified in
  `Docs/19-nanoquant-packed-layout-v1.md`, enforced by the deployment runtime, and binds its exact modified
  llama.cpp reference provenance in descriptor schema 1.
- [x] **M6.4** Implement offline frozen-to-packed conversion and block-aligned artifact sharding. Conversion streams
  the validated logical artifact one block at a time into atomic safetensors shards bound to the source descriptor
  hash. The accepted 26-block Gemma artifact reconstructed all 1,274 tensors across 182 layers exactly; its 26 packed
  shards contain 87,072,592 bytes, 3.2764% of the logical shard bytes. Complete model-shell conversion remains open.
- [x] **M6.5** Implement packed artifact inspection and validation without loading all tensors. Descriptor, path,
  size, hash, inventory, shape, and dtype checks use shard headers; exact payload/reference validators are separate
  explicit passes. Packed reference execution matched the logical backend exactly over 459,264 real-shape output
  elements.
- [x] **M6.6** Implement the logical dense-reconstruction reference backend.
- [x] **M6.7** Implement the factorized PyTorch reference backend.
- [x] **M6.8** Implement the `RuntimeBackend` capability/support/prepare/linear contract. The deployment-only
  contract declares logical/device/input/factor/scale/outlier/workload support, deterministic capability,
  shape alignments, workload bounds, stable rejection codes, prepared-layer ownership, and self-contained
  dense/factorized PyTorch reference implementations.
- [x] **M6.9** Implement backend planning for every layer before inference and strict-mode failure on unexpected fallback.
  Planning resolves the complete unique layer inventory once, preserves every rejected backend and reason in a
  structured dispatch report, counts named fallbacks, refuses the first unexpected fallback in strict mode, and
  prepares only the exact planned state/backend inventory without repeating capability discovery in `linear()`.
- [x] **M6.10** Map the modified llama.cpp GGUF/NanoQuant logical and packed representation to the rewrite's format documentation. `Docs/19-nanoquant-packed-layout-v1.md` records source/dirty-patch hashes, shapes, names, sign bits, padding, alignment, scales, salient semantics, bias handling, and Q/K row-permutation responsibility.
- [x] **M6.11** Implement conversion/parity tests between rewrite frozen state, packed runtime state, and modified
  llama.cpp/GGUF state where semantically compatible. The pinned Gemma bridge exported 26 source-bound checkpoint
  shards; the exact pinned converter accepted 182 groups and mapped 1,274 tensors. Direct GGUF inspection matched all
  22,719,854 normalized NanoQuant elements exactly, retained 158 model-shell tensors, and the pinned CPU build loaded
  the resulting 699,863,936-byte GGUF and generated one token. Native rewrite execution is covered by M6.12.
- [x] **M6.12** Integrate or port the initial NanoQuant CUDA backend with explicit version/capabilities. Version 1
  (`cuda-packed-triton`) ports the pinned two-stage packed-sign operation to Triton, accumulates and returns F32,
  and declares CUDA architecture, input/factor/scale/outlier dtype, bias, deterministic, prefill, and decode support.
  Preparation transfers immutable packed tensors once; execution consumes sign words directly and fuses pre/mid/post
  scales, floating or scaled-I8 salient columns, and optional bias. Leased CUDA tests cover all declared dtypes,
  tail words, bias/outlier branches, deterministic replay, and multi-tile salient input. Full pinned Gemma validation
  passed all 182 layers and 18 real shape/rank combinations for one-token decode (459,264 outputs, maximum absolute
  error `1.9073486328125e-06`, 1,177,088 peak incremental allocated bytes) and four-token prefill (1,837,056
  outputs, maximum absolute error `3.814697265625e-06`, 1,370,112 peak incremental allocated bytes). Separate
  Model-shell generation, static workspace ownership, and performance tuning remain later gates.
- [x] **M6.13** Implement separate prefill and decode execution plans. `ExecutionPlans` resolves one ordered layer
  inventory against independent prefill/decode backend priorities, preserves each workload's fallback report, requires
  a shared device type and one token per decode batch item, and supports strict specialized backends. Paired
  preparation caches unique layer/backend payloads so identical selections share device weights while divergent
  selections prepare separately. `linear_at()` enforces planned device, dtype, feature width, and batch geometry;
  decode token geometry remains exact while prefill accepts a positive chunk no larger than its planned prompt bound,
  without repeating capability discovery or backend lookup. On the complete Gemma artifact, both plans selected CUDA
  with zero fallback, all 182 dispatches shared the same prepared layers (87,087,616 incremental bytes), and execution
  produced 1,837,056 prefill plus 459,264 decode outputs with 342,528 peak incremental bytes. Evidence is
  `evidence/m6/gemma-pageable-v28-execution-plans-validation.json`.
- [x] **M6.14** Implement generation-engine prompt batching, positions, attention metadata, stopping, and deterministic
  mode. The deployment engine builds explicit left-padded ragged batches, derives physical cache-aligned positions,
  advances
  attention/cache metadata while masking finished rows, supports EOS/token-sequence/maximum-token stopping, and
  performs exact greedy replay. A real two-prompt Gemma pass exercised one prefill plus three decode forwards through
  all 182 packed linears with zero fallback and exact replay.
- [x] **M6.15** Implement bounded/static or paged KV-cache management and verify positions/padding across batches.
  The Transformers shell owns a total-length-bounded `HybridCache`; the generation request fixes its maximum at
  padded prompt width plus the output limit. Fixture assertions cover unequal prompt lengths, inactive-row masking,
  positions, cache positions, and growing attention masks, while the pinned Gemma batch used prompt lengths 2 and 8
  with a 12-token cache bound. A 32-token unequal-length Gemma fixture exactly matches Transformers HybridCache
  generation and verifies fixed local/global cache extents after sliding-window rollover.
- [x] **M6.16** Keep packing, capability discovery, device transfers, and allocator cleanup outside the token hot loop.
  Packed states are planned/prepared and model linears are bound before `generate()`; the loop calls only the already
  selected prepared dispatches. Static cache-position and attention storage are preallocated, and the real validator
  performs source-weight release and allocator cleanup before timing. Both real plans had zero fallback; first and
  second deterministic passes retained 699,885,568 and 699,886,080 allocated CUDA bytes, respectively.
- [x] **M6.17** Implement device-side greedy and configured sampling paths with explicit synchronization boundaries.
  Greedy argmax and seeded categorical temperature/top-k/top-p processing remain on the logits device. Static
  sampling configuration and the device generator are created before the loop; stopping checks use a declared
  batching interval and the result reports stopping and terminal host synchronization counts. A pinned two-prompt,
  eight-token Gemma pass used temperature 0.8, top-k 64, top-p 0.95, and seed 20260715; all 182 packed linears had
  zero fallback, both passes returned identical tokens, and only one stopping plus one terminal sync was recorded.
- [x] **M6.18** Add logical-reference versus factorized-reference tests over real model shapes.
- [x] **M6.19** Add packed-backend parity tests over the full declared shape/rank/dtype/outlier matrix. The finite
  capability Cartesian product executes all 540 combinations of three input dtypes, three source-factor dtypes,
  three scale dtypes, five salient encodings/presence states, bias on/off, and prefill/decode on deliberately
  unaligned 35x17 rank-33 word-tail geometry; every case matches the independent F32 operation and replays exactly.
  The complementary complete-artifact passes cover all 182 Gemma layers, 18 real shape/rank combinations, and real
  salient counts 2/7 for both one-token decode and four-token prefill.
- [x] **M6.20** Add long-enough generation tests for output parity, cache correctness, and memory growth. A 32-token
  unequal-prompt Gemma fixture matches the Transformers HybridCache reference exactly and verifies fixed local/global
  cache shapes. The pinned packed model then generated 128 forced tokens twice in the reconciled F32 shell plus
  Gemma chat-template protocol: its first 16 tokens exactly equal the retained modified llama.cpp CUDA/CPU output,
  all 182 linears used CUDA with zero fallback, the cache bound was 144 tokens, peak allocation was 1,313,887,232
  bytes, and retained allocation differed by only 1,024 bytes between deterministic passes.
- [x] **M6.21** Implement kernel, layer, block, prefill, decode, and end-to-end benchmark commands with JSON output.
  `tools/benchmark_runtime.py` selects any combination of the six scopes, keeps preparation outside timed regions,
  retains every raw sample, and reports p10/p50/p90/p99 latency and throughput, warm-ups, repetitions, peak CUDA/host
  memory, environment/artifact identity, fallback counts, and deterministic output hashes. The pinned F32/chat Gemma
  run executed ten cases with three warm-ups and ten samples each, all 182 linears on CUDA, and zero fallback. Median
  single-token model decode was 96.44 ms (10.37 tokens/s) and 32-token end-to-end generation was 2.922 s
  (10.95 tokens/s), establishing the unoptimized rewrite baseline for Milestone 7 rather than accepting its speed.
- [x] **M6.22** Verify a clean runtime-only installation can load a packed artifact and generate text. The atomic
  runtime bundle contains the complete packed artifact, config/tokenizer assets, 158 ordinary checkpoint tensors,
  and three explicitly derived non-persistent buffers while excluding all 182 dense source linears. The current
  56,582-byte `nanoquant-runtime` wheel contains exactly 23 deployment members and no research packages. Installed
  into an isolated target, it loaded the 731,007,650-byte bundle without a source-model argument, replaced all 182
  linears, selected CUDA with zero fallback, bound all 157 fused RMSNorms, 26 decode-only RoPE sites, and 22 guarded
  short-context sliding layers, executed 330 prepared sliding-prefix updates, and generated the exact retained
  16-token llama.cpp text.
- [x] **M6.GATE** Load and run a packed NanoQuant artifact through reference and CUDA backends with complete numerical
  parity coverage and no research-package dependency. Full-artifact logical/packed reference validation covers all
  182 Gemma layers; the native CUDA path covers every real shape plus the 540-case declared capability matrix; long
  generation matches the retained llama.cpp prefix and remains memory-bounded; and the isolated deployment wheel
  repeats the full composed CUDA generation without importing any research module. Stable performance parity remains
  Milestone 7 rather than being implied by this correctness gate.

## Milestone 7 — Close the inference performance gap

Dependencies: Milestone 6 correctness and benchmark suite; Milestone 0 profiles  
Outcome: runtime performance is measured, explained, and competitive with the modified llama.cpp reference.

- [x] **M7.1** Freeze the apples-to-apples NanoQuant versus modified llama.cpp benchmark protocol and artifact pair.
  The v28 packed descriptor and its exactly derived GGUF are bound by hashes in
  `Docs/20-inference-performance-protocol.md`. The matched workload uses the Gemma chat template, 16 prompt plus 32
  generated tokens, batch one, F16 KV storage, F32 operation/shell boundary, non-flash attention, greedy sampling,
  three rewrite warm-ups, and ten samples. Exact-prompt prefill is 96.61% of llama CLI, while decode is only 5.53%,
  so the protocol is frozen but the throughput gate decisively fails.
- [x] **M7.2** Record warm-up, repetitions, median/p10/p90, TTFT, prefill throughput, inter-token latency, decode throughput, memory, and fallback count.
  The protocol-matched F16-KV CUDA Graph record retains all ten raw samples after three warm-ups. It reports
  16-token prefill p50 41.945 ms / 381.67 tokens/s, isolated inter-token model-decode p50 4.567 ms / 218.97
  tokens/s, TTFT p50 42.226 ms, and complete 32-token p10/p50/p90 183.185/201.963/228.256 ms. Peak allocation is
  806,364,672 bytes; all 182 linears use CUDA and graph, prefill, and decode fallback counts are zero. The bracketing
  F16 candidate/control/candidate records preserve the exact output hash and environment/configuration identity.
- [x] **M7.3** Capture a new end-to-end profile and account for at least 90% of wall time before choosing optimizations.
  The protocol-matched, three-pass CUDA-event profile uses separate sparse top-level, block-component, and
  prepared-linear passes to avoid cross-level event inflation. Its top-level pass accounts for 97.77% of synchronized
  wall time at p50 (97.56–97.82% p10–p90), with model CUDA time itself accounting for 99.85%. The component pass
  attributes about 51% of profiled model time to eager attention and 13% to MLP; the linear pass attributes about
  29% to all 182 prepared linears. Instrumented absolute latency is diagnostic only and is compared separately with
  the uninstrumented M7.1 baseline. A separate warmed Kineto trace counted 2,558 CUDA kernels per token: 364
  NanoQuant stage kernels and 2,194 shell/cache/attention kernels. NanoQuant consumed about 40% of non-nested device
  kernel time, proving that framework launch structure is at least as important as packed-kernel arithmetic.
- [ ] **M7.4** Remove per-token Python/device synchronizations, scalar reads, repeated capability checks, and layout conversions.
  A binding-time prevalidated-dispatch experiment retained the checked public path as a control and removed repeated
  plan/backend validation only from internally bound linears. Two candidate runs bracketed one same-code control.
  All exact output hashes matched, but 32-token medians were 2.969 s candidate, 2.982 s control, and 3.023 s
  candidate. Because the second candidate regressed and the spread is within observed WDDM variance, the fast path
  was reverted and this task remains open.
  The first promoted launch reduction binds all 157 Gemma3 RMSNorms to PyTorch's native fused F32 operation after
  shell loading. It is bit-exact on the accepted output, preserves the legacy expression for non-F32 inputs, reduces
  the one-token kernel count from 2,558 to 1,616, and lowered matched 32-token latency from 3.268 s to stable 2.467 s
  and 2.440 s candidate medians. This is a 24.9% latency reduction, but the broader task remains open.
  A later guarded cache kernel removes 130 more launches/token by combining F32-to-F16 K/V conversion, both prefix
  writes, and both F32 attention-view materializations into one launch per layer. It is bit-exact for backing caches
  and views, falls back at rollover or for unsupported inputs, and improves candidate/control/candidate isolated
  decode from 30.88 ms control to 28.77 and 29.38 ms. Python/framework work outside these promoted paths remains.
- [ ] **M7.5** Ensure every intended NanoQuant layer dispatches to the optimized backend or records an actionable unsupported reason.
- [ ] **M7.6** Port/evaluate packed sign-word loads, aligned vector loads, lane-zero broadcasts, and branchless sign-bit application from `nanoquant.cu`.
  A decode-only Triton port evaluated one packed sign word per rank/output tile, eight output rows per program,
  branchless sign application, and four independent accumulators. All 27 CUDA backend/matrix tests passed, but the
  representative Gemma gate-projection kernel regressed from 155.14 to 207.90 microseconds p50 (+34.0%), and the
  complete prepared-linear call regressed from 175.62 to 228.35 microseconds. The candidate was reverted before an
  end-to-end promotion run; the broader task remains open for a mapping that fits Triton's execution model.
- [ ] **M7.7** Port/evaluate sign-aware FMA and multiple-accumulator decode loops from `nanoquant.cu`.
- [ ] **M7.8** Implement and tune dtype-specialized contiguous fast paths for the actual runtime input/scale dtypes.
- [x] **M7.9** Evaluate fused first-stage Q/K/V or other shared-input projections using real block profiles.
  The accepted CUDA/F32 decode specialization prepackages each block's compatible Q/K/V factors and executes all
  three first stages in one launch and all three reconstruction/outlier stages in a second launch. Prefill and
  unsupported backends, dtypes, shapes, bias, or outlier-scale layouts retain the individual prepared-linears. A
  direct CUDA comparison bounds reduction-order differences below 5e-5 (3.05e-5 observed); the complete real-model
  hash remains exact. All 26 groups bind, reducing kernels from 729 to 625, launch APIs from 726 to 622, ATen calls
  from 2,411 to 2,255, and device self time from 5.269 to 4.887 ms. Candidate/control/candidate decode medians are
  45.28, 49.98, and 43.75 ms; 32-token medians are 1.220, 1.440, and 1.360 s.
  The same guarded primitive groups each block's compatible MLP gate/up pair. All 26 pairs bind and preserve the
  exact full hash, reducing kernels from 625 to 573, launch APIs from 622 to 570, ATen calls from 2,255 to 2,203,
  and device self time from 4.887 to 4.267 ms. Candidate/control/candidate decode medians are 26.48, 42.95, and
  22.84 ms; 32-token medians are 0.995, 0.993, and 0.934 s. The combined immutable prepack costs 55 MB of steady
  allocation (7.8%), while production peak remains unchanged at 1.296 GB.
- [ ] **M7.10** Tune decode kernels across real rank/shape/alignment cases and report achieved memory bandwidth/occupancy.
- [ ] **M7.11** Tune prefill kernels independently across representative token and batch sizes.
- [ ] **M7.12** Fuse scale, bias, and salient-outlier operations only where end-to-end profiles show a net benefit.
- [ ] **M7.13** Optimize outlier paths for zero/small/common counts and ensure specialization does not regress general cases.
- [ ] **M7.14** Measure and optimize KV cache, attention, final vocabulary projection, and sampling after quantized linears no longer dominate.
  The first accepted attention-side specialization fuses the pinned batch-one/F32/one-token Gemma3 Q/K rotary
  operation into one Triton launch per block while retaining eager prefill and unsupported-shape/device fallbacks.
  Explicit `mul.rn.f32` and `add.rn.f32` instructions make the kernel bit-identical to the pinned eager expression,
  not merely output-token equivalent. The real trace bound all 26 attentions, preserved token 236764 in every pass,
  and removed exactly 234 launches/token (1,616 to 1,382) while reducing non-nested device kernel time from 9.40 to
  7.58 ms. WDDM timing varied materially, but both candidates bracketed around the first control were faster and the
  final adjacent control/candidate pair improved exact 32-token latency from 1.391 to 1.208 s (13.1%). Cache
  promotion, eager attention proper, the vocabulary projection, and sampling remain.
  The next accepted attention-side change recognizes that the pinned 48-token protocol is wholly inside Gemma's
  512-token sliding window. A guarded decoder-layer binding returns the existing causal mask unchanged only while
  both mask dimensions and the cache position remain inside that window; longer contexts execute the original
  `ones_like`/`tril`/`where` path. All 22 sliding layers bound, tokens remained exact, and the kernel census fell by
  another 88 launches/token (1,382 to 1,294). Candidate/control/candidate model-decode medians were 73.23, 82.49,
  and 77.20 ms; the candidates average 8.8% below control. Exact 32-token medians averaged 2.384 s versus 2.429 s
  control (1.8% lower) with unchanged allocation and zero fallback.
  The prepared HybridCache now removes the larger identity update before rollover: instead of constructing rotation
  indices, gathering the full K/V cache, zeroing its backing tensors, and adding the gathered copy back, it uses the
  generation adapter's host-known position/length to perform the identical indexed prefix write without a CUDA
  scalar read. At the exact rollover boundary it delegates to Transformers. A tiny 32-token test crosses rollover and
  matches exactly. Combined with mask elision, the pinned trace falls from 1,382 to 964 launches/token and 5,261 to
  4,271 ATen calls/token with every token exact. The stable adjacent control/candidate pair improves exact 32-token
  latency from 1.141 to 1.000 s (12.4%) and isolated decode from 37.80 to 32.95 ms (12.8%).
  The next accepted cache specialization fuses both F32-to-F16 conversions, both backing-cache prefix writes, and
  both full F32 attention-view materializations. Direct CUDA tests are bit-exact against the PyTorch storage and
  promotion sequence. The pinned trace executes one fused kernel in each of 26 layers and falls from 964 to 834
  kernels/token, 961 to 831 launch APIs, and 4,271 to 3,595 ATen calls. Candidate/control/candidate exact 32-token
  medians were 0.851, 0.914, and 0.893 s; isolated decode medians were 28.77, 30.88, and 29.38 ms. Rollover and every
  unsupported dtype/device/layout use the unchanged fallback. Eager attention proper, the vocabulary projection,
  and sampling remain.
  The vocabulary projection is now specialized around the bundle/reference's actual BF16 tied table instead of
  expanding it to F32. A fused embedding lookup preserves F32 embedding values bit-for-bit, while the mixed
  BF16-weight/F32-input output kernel has 3.70e-6 maximum real-logit error and exact argmax/hash. The head kernel falls
  from 2.945 to 1.481 ms, total device self time from 6.82 to 5.35 ms, and matched peak allocation from 1.218 to
  0.655 GB. Candidate/control/candidate isolated decode medians are 28.60, 36.92, and 29.18 ms. Eager attention and
  sampling remain, so the broader task stays open.
  A further fixed-geometry kernel fuses grouped-query score matmul, scale/mask/softmax, and value matmul for the
  pinned batch-one decode while cache length is at most 64. It falls back for longer/unsupported cases. The real
  trace preserves every token while reducing kernels from 833 to 729 and ATen calls from 3,581 to 2,411. Both
  candidate isolated-decode medians beat control (31.90, 38.78, and 33.11 ms); end-to-end candidates straddle control
  but average 4.1% lower. Sampling and broader/long-context attention work remain.
- [x] **M7.15** Compare eager and compiled/static decode-step execution with stable shape/correctness coverage.
  Direct whole-model and decode-only `torch.compile` feasibility probes preserved the exact token but are rejected
  in their current form. Whole-model compilation produced 58 graphs and 47 workload-`ContextVar` breaks. Restricting
  compilation to decode after eager prefill still produced 49 graphs and 40 such breaks; its stabilized last-five
  median was 134.22 ms versus 41.95 ms eager (3.20x slower). Retrying requires a traceable fixed-workload packed op
  and stable cache/layer specialization, so the broader task remains open.
  A bounded fixed-shape `torch.compile(mode="reduce-overhead")` probe preserved the exact token but generated 49
  graphs, hit 40 `ContextVar.get` graph breaks, exceeded per-layer recompilation limits, and skipped CUDA graphs for
  mutating HybridCache updates. After a 32.98 s first call, compiled p50 was 890.47 ms versus 80.85 ms eager (11.0x
  slower). Broad model compilation is rejected.
  The accepted alternative captures one batch-one, one-token CUDA graph per pre-rollover decode position after an
  eager shape warm-up, restores the cache snapshot after capture, and replays later requests with the same static
  geometry. Unsupported devices, batches, token counts, cache objects, or rollover positions use eager execution.
  A tiny CUDA regression is exact across eager, capture, and replay. On the pinned Gemma workload, the production
  validator exactly reproduces all 32 tokens and the retained llama.cpp prefix with 31 captures, 31 replays, and
  zero graph or packed fallback. Candidate/control/candidate complete-generation medians are 186.80, 848.19, and
  192.03 ms; the graph-aware profile records one `cudaGraphLaunch`, four dynamic-input copies, and about 4.40 ms
  model-device time rather than the eager path's 570 ordinary launch APIs.
- [ ] **M7.16** Establish designated-host performance CI with raw sample retention and environment health checks.
- [ ] **M7.17** Add no-more-than-10% regression gates against the accepted NanoQuant runtime baseline.
- [x] **M7.18** Reach at least 70% of the fastest compatible modified llama.cpp reference throughput on the agreed workload, or produce an accepted profile-backed gap analysis and follow-up plan.
  Protocol-matched F16-KV static CUDA Graph decode averages 200.52 ms for the exact 32-token workload across the two
  bracketing candidate records, or 159.59 tokens/s. That is 86.50% of the compatible 184.50 tokens/s modified
  llama.cpp decode result and
  exceeds the 70% acceptance threshold while retaining the exact output hash, zero fallback, and unchanged
  1,295,585,792-byte production peak allocation.
- [x] **M7.19** Re-run numerical, artifact-size, VRAM, and quality gates after every accepted layout/kernel change.
  The final accepted runtime preserves packed descriptor
  `b4f0c6270c4b59f8293c909ddeb21042ad1a2d7ee18601c77e4c57563c900487`, 87,072,592 weight bytes,
  87,507,442 physical bytes, and the source run's 0.996318 effective BPW. A fresh all-layer CUDA comparison checks
  7,348,224 outputs across all 182 linears with 4.77e-6 maximum absolute error and deterministic replay. A fresh
  serial 64x128 WikiText-2 run reproduces PPL 453.5709857733353 exactly over 8,128 tokens. Production graph
  validation preserves the exact 32-token hash and llama.cpp prefix with zero fallback and unchanged
  1,295,585,792-byte peak allocation.
- [x] **M7.GATE** Publish a reproducible NanoQuant-versus-llama.cpp report with source/kernel/artifact hashes, workload, profiles, parity, quality/BPW, and explained end-to-end performance.
  `Docs/20-inference-performance-protocol.md` binds the pinned model, rewrite/llama sources, CUDA kernel, packed
  artifact, converted GGUF, prompt/cache/sampling protocol, raw profiles, A/B/A samples, and post-optimization gates.
  The final F16-KV rewrite averages 159.59 tokens/s versus 184.50 tokens/s for compatible llama.cpp (86.50%), with
  exact retained output behavior, zero fallback, 1.296 GB production peak, 0.996318 effective BPW, and exact serial
  PPL 453.571 versus contemporary legacy 444.333 (+2.08%). Remaining M7 CI and broader/general-path tuning tasks
  stay explicitly open; they do not invalidate the published designated-host parity result.

## Milestone 8 — Complete evaluation, diagnostics, and self-documenting reports

Dependencies: Milestones 1, 4, and 6; can progress in parallel with Milestone 7  
Outcome: runs produce cheap-to-expensive decision evidence and actionable reports.

- [x] **M8.1** Implement the versioned evaluator registry and immutable evaluator specifications.
- [x] **M8.2** Implement calibration, quick-decision, and final-evaluation partitions with content hashes and overlap detection.
- [x] **M8.3** Implement artifact-structure and packed-reference parity smoke evaluators.
  Versioned smoke-tier adapters now reuse the strict packed artifact opener and packed reference validator through
  `EvaluatorRegistry`. The structure result records descriptor identity, blocks/layers/tensors, logical weight and
  physical bytes, and successful hash/header verification. The parity result executes every requested logical/packed
  layer pair with an explicit tolerance and records output count, maximum error/layer, and pass state. A fixture runs
  both evaluators cumulatively through the smoke tier, while corruption/reference behavior remains covered by the
  underlying validator suite and the architecture contract keeps deployment runtime independent of application code.
- [x] **M8.4** Implement token-accurate negative-log-likelihood/perplexity evaluation with BOS/EOS, causal shift, padding, stride, and partial-window tests.
- [x] **M8.5** Implement the selected zero/few-shot task evaluators with pinned dataset/task/prompt revisions.
  The versioned suite reproduces the legacy lm-eval 0.4.12 PIQA, ARC Easy/Challenge, HellaSwag, WinoGrande, and
  BoolQ renderers, metrics, zero-shot protocol, 200-row ordering, exact dataset commits/splits, and harness prompt
  commit. ARC uses the harness-selected test split; retained row-zero prompts caught and prevent a validation-split
  substitution. Hugging Face causal pair encoding pins Gemma BOS behavior, trailing-space movement, concatenated
  tokenization, and the harness's `max_length + 1` target window. Complete task-input identities bind raw selected
  document content, tokenized partitions, tokenizer behavior files/parameters, prompts, and any exact few-shot
  demonstrations. Known-logit/batching/truncation tests and six real cached dataset plus pinned Gemma tokenizer
  comparisons reproduce retained legacy row-zero text and token hashes; all six 200-row input identities are
  recorded in `Docs/08-evaluation.md`.
- [x] **M8.6** Implement deterministic generation sanity/regression cases.
  A versioned smoke evaluator binds case name/version, prompt tokens, expected generated tokens, expected stop reason,
  and sanity thresholds into one immutable case identity. It compares multiple observed runs, preserves expected and
  per-run token hashes, reports the first mismatch and longest repeated-token run, and requires exact tokens, exact
  stop reasons, repeat determinism, minimum output length, and bounded repetition. Tests cover registry dispatch,
  exact repeatable output, stable case identity across observations, mismatch/nondeterminism/stop/repetition/empty
  diagnostics, and malformed cases.
- [x] **M8.7** Implement long-context evaluation where supported by the model/runtime plan.
  The versioned full-tier Gemma3 HybridCache protocol binds the declared 32,768-token limit, 512-token local window,
  six-layer global interval, prefill chunk size, and zero-fallback policy. The evaluator requires cases to cross both
  the window and a chunk and checks exact tokens/stopping, cache length, prefill/decode calls, fallbacks, and peak
  device bytes. Generation streams prompt chunks through one cache; prepared prefill plans accept bounded variable
  chunks; local cache updates and masks preserve chronological sliding-window semantics after rollover; and requests
  beyond the model limit fail before execution. The production packed Gemma bundle agrees exactly with a monolithic
  oracle at 1,025 and 4,097 prompt tokens. A near-ceiling 32,761+4-token case agrees between independent 256- and
  512-token chunkings with zero fallbacks and 1,592,178,176 peak allocated bytes on the 12 GB GPU. A deterministic
  tiny Gemma fixture also reaches its exact configured ceiling with Transformers token parity. Protocol and retained
  evidence identities are recorded in `Docs/08-evaluation.md`.
- [x] **M8.8** Implement smoke, quick, standard, and full evaluation tiers from the registry.
- [x] **M8.9** Implement predefined promotion, rejection, and inconclusive decisions with immutable gate policy.
- [x] **M8.10** Implement paired comparisons, bootstrap/appropriate confidence intervals, repeated-run variability, and minimum meaningful deltas.
  `compare_paired` validates finite equal-length observations, normalizes deltas so positive always means improvement,
  and uses a seeded paired bootstrap to produce a requested confidence interval. Results preserve raw and
  direction-normalized means, candidate/baseline/paired sample standard deviations, sample count, bootstrap identity,
  and the predefined minimum meaningful delta. Interval placement yields explicit meaningful-improvement,
  meaningful-regression, no-meaningful-difference, or inconclusive outcomes. Tests cover both metric directions,
  deterministic replay, variability, every outcome, and invalid/non-finite requests.
- [x] **M8.11** Implement exact effective core/artifact BPW, bytes, memory, quantization cost, and runtime metrics as separate dimensions.
  Immutable evaluation dimensions now preserve exact source-parameter/core-bit numerators, logical and complete
  deployable byte counts, and separately derived effective-core and artifact BPW. Quantization device/host/temp-disk
  memory is distinct from runtime device/host memory; six named compression/evaluation stage durations retain their
  accounted total; TTFT, prefill throughput, inter-token latency, decode throughput, and fallback count remain a
  separate runtime dimension. Constructors reject negative, non-integral, and non-finite values. Tests demonstrate
  that core BPW, artifact BPW, bytes, the two memory phases, cost, and runtime values cannot be substituted silently.
- [x] **M8.12** Preserve per-layer objective-weighted reconstruction tables for every run.
- [x] **M8.13** Preserve the Experiment 019-style final-block-versus-block-entry table with positive, negative, and `n/a` semantics.
- [x] **M8.14** Preserve source/base-model, block-entry, final-pre-KD, and final-post-KD snapshots and named comparisons.
  Immutable block results continue to retain the objective-weighted source reference, block entry, each accepted
  layer boundary, optional post-block refit, and final frozen pre-KD value with named denominator-safe comparisons.
  Global-tuning schema version 2 adds an ordered per-block pre/post-KD table measured against the same deterministic
  base-model hidden-state reference. Its protocol identity binds token content and limits, padding, BF16 reference
  storage, FP32 accumulation, unweighted MSE, denominator floor, and version; legacy schema-version-1 artifacts load
  with an explicitly absent table. The probe streams one sequence and keeps only bounded pageable host references.
  Reports preserve the local pre-KD table and label local versus probe metrics so unlike objectives are not silently
  compared. Unit/integration tests cover exact and changed models, padding identity, near-zero `n/a`, alignment,
  legacy loading, and end-to-end KD persistence. The complete pinned Gemma run was backfilled without changing any
  tuned tensors or epoch results: active artifact
  `sha256-edef5622c5b03e24b75d77ee05f389e064e24d73a3ff7087282d6c3761629669` contains all 26 blocks under protocol
  `sha256:cf208a4f3632f640e2ec4e1ac12e8cafbcf4bbbff03d17839ec29ad8ae79098c`.
- [x] **M8.15** Implement diagnostic rules for calibration instability, Hessian conditioning, ADMM plateau, export gaps, ineffective retry/outliers, poor tuning recovery, and runtime fallback.
  A versioned immutable `DiagnosticPolicy` now owns every threshold and has a semantic identity. `diagnose` derives
  stable, location-aware findings from structured observations for non-finite or cross-partition calibration,
  excessive Hessian condition/jitter, ADMM tail plateau/divergence, latent-to-export error amplification, retry
  improvement per added bit, outlier block-loss utility, tuning recovery fraction, and unexpected runtime fallback.
  Findings retain code, severity, evidence, artifact validity, and a recommended next diagnostic. All new codes are
  registered with documentation/remediation metadata. Tests exercise every required family, a healthy observation,
  invalid-artifact handling, configurable/versioned thresholds, and invalid policy/observation inputs.
- [x] **M8.16** Implement complete summary reports for completed, failed, interrupted, resumed, and forked runs.
  Reports now derive a typed `RunSummary` from the manifest and canonical event stream rather than console text. They
  retain event/stage counts, warnings/errors with codes and fields, attempts/resume count, complete lifecycle and
  terminal context, failure metadata, artifacts, parent/fork stage, and manifest/event consistency warnings. Markdown
  renders execution, lineage, issues, failure/interruption context, and artifacts for every outcome. Tests cover
  completed, failed, interrupted, resumed-to-completion, and forked histories plus sequence, foreign-run, and
  terminal-status inconsistencies; the existing foundation run still generates its self-contained summary.
- [x] **M8.17** Implement candidate-versus-baseline reports with semantic config diff, artifact reuse, per-layer/block alignment, uncertainty, warnings, and Pareto dimensions.
  `tools/compare_block_trajectories.py` now provides the block-alignment slice: it selects the latest journal
  identity, rejects stale/noncontiguous prefixes, resolves committed block artifacts, and compares any number of
  named legacy post-refit trajectories with JSON/Markdown deltas. Optional legacy rank-utility CSVs add exact
  prefix rank-sum/mismatch checks and scoped BPW. Rewrite BPW includes every committed bit-cost category; the
  legacy CSV's rank-dependent BPW includes binary factors and middle scales but excludes pre/post scales and
  outliers, so it is explicitly not presented as like-for-like total BPW. The application comparison report now
  adds canonical semantic config diffs with explicit missing values; source/dataset/evaluator/environment
  comparability; stage-aware artifact reuse; exact aligned layer/block absolute and denominator-safe relative deltas;
  seeded paired-bootstrap uncertainty; new/resolved/shared warning codes; and a typed promotion-gate conclusion.
  Its Pareto view keeps quality, core/artifact storage, every quantization-cost stage, quantization/runtime memory,
  prefill, decode, and fallback coverage separate, retaining integer byte deltas exactly. Required identity mismatch
  produces a `not-comparable` conclusion and suppresses metric-delta rendering instead of implying parity. Tests
  cover semantic exclusions/additions/removals, cross-stage reuse, missing and near-zero alignments, exact large-byte
  deltas, uncertainty, Pareto tradeoffs, warning transitions, invalid ambiguity, and non-comparable reports.
- [x] **M8.18** Include experiment number, zero-argument runfile path/hash, purpose, hypothesis, baseline, environment, cost, conclusion, and recommended next action.
  Typed run summaries now retain intent fields; launcher kind, experiment, repository-relative path, content hash,
  code revision, and arguments; the full redacted environment; manifest elapsed time; structured event timing and
  memory observations; exact observed device/host/temporary-disk peaks; conclusion; and recommended next action.
  The report explicitly identifies a zero-argument numbered runfile and flags experiment mismatch, unexpected
  runfile arguments, missing repository-relative provenance, invalid timestamps, and malformed cost observations.
  Explicit structured conclusion/action fields win; deterministic status-aware defaults keep completed, failed,
  interrupted, running, and created legacy manifests self-documenting. Markdown renders dedicated outcome, launcher,
  environment, and cost sections. Unit tests cover exact provenance/environment/cost, explicit and default outcomes,
  failures, interruption, resume, forks, mismatch warnings, and structured cost observations; the numbered foundation
  integration run verifies that its generated report contains the complete envelope.
- [x] **M8.19** Implement evaluator and task-result caching using complete semantic identities.
  Separate immutable `evaluation-task-inputs` and `evaluation-result` artifacts allow tokenized/formatted task data
  to be reused across models without ever reusing a model result across packed artifacts. Task identities bind the
  evaluator, task/dataset revisions and content, split, ordered sample selection, partition revision, tokenizer
  revision/content/behavior, prompt revision/content, exact few-shot demonstrations, selection seed, and
  preprocessing version. Result identities add the exact model artifact, runtime backend/version/mode and numerical
  parameters, optional environment identity, and seed. The run-local sorted index is published atomically under a
  cross-process lock; hits validate the artifact and embedded identity, misses state that no complete identity
  matches, and conflicting payloads under one identity fail instead of overwriting evidence. Typed cached execution
  returns the same result contract on hit and miss. Tests cover every invalidation boundary, order-normalized named
  parameters, model-independent task reuse, changed-model misses, durable reopen, conflict/corruption rejection,
  typed no-reexecution behavior, and concurrent publication without lost entries.
- [x] **M8.20** Add golden report tests using Experiment 019 data and synthetic near-zero denominators.
  Golden Markdown now covers an actual seven-layer block from the frozen Experiment 019 weight-error and
  rank-utility CSVs. The test first verifies both complete source files against their M0 manifest hashes, then maps
  the retained ranks, bit counts, raw/weighted reconstruction errors, and tuned block boundary into current typed
  contracts and compares the whole rendered report byte-for-byte. A second golden uses non-zero source and block
  entry losses below a configured denominator floor, preserving absolute deltas while requiring both relative
  values to render as `n/a`. The fixtures therefore catch numerical-column, sign/baseline, precision, ordering, and
  Markdown drift without normalizing away meaningful report changes.
- [x] **M8.21** Validate evaluator batching, caching, sample limiting, distributed reduction, and known-logit results.
  The causal evaluator supports a positive deterministic sample limit applied before windowing and batching and now
  records selected samples in addition to windows and valid targets. Serial and multi-window batched runs agree;
  constructed next-token logits verify causal shift and overlap exactly; BOS/EOS, padding, partial final windows,
  and invalid/empty cases retain exact denominators. The distributed reducer sums per-shard total NLL with `fsum`,
  divides once by the global token count, and sums sample/window counts. Its unequal shards deliberately have
  different loss distributions, ruling out mean-of-means or mean-perplexity reductions. M8.19 tests additionally
  prove cached typed results equal uncached results, skip re-execution, survive reopen, and invalidate on every
  semantic boundary.
- [x] **M8.GATE** Demonstrate that a candidate can progress from layer replay through quick/standard/full evaluation and that its run directory alone explains intent, execution, issues, cost, results, and comparison.
  The reusable campaign service requires ordered exact tier-local plans, executes registered evaluators, binds
  specification/result/policy identities, and stops before later tiers on rejection or inconclusive evidence. The
  retained Gemma v28 campaign promotes a 0.3540% four-block replay through complete-artifact/rank/BPW/trajectory
  quick checks, exact 8,128-target WikiText-2 standard evaluation, and full long-context/runtime/memory/fallback
  checks. Its directory copies all seven compact source results and contains a resolved intent/environment manifest,
  structured lifecycle and promotion events, canonical evaluator outputs, immutable gate decisions, resource/cost
  observations, a comparison, conclusion, and next action. The generated summary has no consistency warnings or
  warning/error events. Evidence and identities are documented in `Docs/08-evaluation.md`; this workflow gate does
  not substitute for the broader M10.14 release-candidate suite.

## Milestone 9 — Migrate supported workflows and cut over

Dependencies: Milestones 4–8 required for their respective workflows  
Outcome: supported users no longer need legacy orchestration.

- [x] **M9.1** Create a migration inventory for every numbered legacy experiment and its replacement status.
  `Docs/22-legacy-experiment-migration-inventory.md` freezes all 19 top-level legacy sources by byte count and SHA-256,
  records purpose/config distinctions, maps each to concrete rewrite services/evidence, and assigns validated,
  implemented-but-unvalidated, or partial status with an explicit remaining gate. It also captures legacy provenance
  defects (016's ` copy`, 017's 016 output names, 018's stale printed title, and 019's misleading 1B filename despite
  its 4B model), cross-cutting mechanism dispositions, and a migration order. No native numbered rewrite runfile is
  counted prematurely; M9.2 remains open.
- [ ] **M9.2** Convert supported historical experiments into thin numbered zero-argument runfiles while preserving numbers, names, purpose, and lineage.
  Experiments 008, 013, and 018 are completed migrations: each zero-argument runfile imports one canonical typed
  recipe and calls the shared resident workflow, which resolves the pinned model/calibration and composes compression
  followed by the legacy-default model KD stage. Exact recipe-delta tests distinguish the pre-Phase-1 008/013 lineage
  from 018's tapered tuning/refit recipe. M9.2 remains open for the other supported inventory rows.
- [ ] **M9.3** Move copied dotenv, tee logging, output-directory, model-loading, save, and evaluation mechanics into shared infrastructure/application services.
- [ ] **M9.4** Provide generated YAML/resolved-recipe views for numbered runfiles where useful without making YAML mandatory.
- [ ] **M9.5** Implement or document migration/import for supported legacy `.pt` checkpoints.
- [ ] **M9.6** Implement or document conversion/compatibility with the modified llama.cpp GGUF NanoQuant representation.
- [ ] **M9.7** Implement supported artifact schema/layout migrations as new immutable artifacts with lineage.
- [ ] **M9.8** Implement the final CLI commands: inspect, calibrate, plan, quantize, resume, fork, replay, pack, validate, evaluate, benchmark, compare, and report.
- [ ] **M9.9** Implement the stable Python application API using the same canonical types and services.
- [ ] **M9.10** Rebuild Hugging Face load/save/publish integration as infrastructure over validated artifacts.
- [ ] **M9.11** Ensure publishing cannot alter numerical content without creating/revalidating a new artifact.
- [ ] **M9.12** Publish contributor guidance for adding a factorizer, policy, adapter, evaluator, packed layout, and backend.
- [ ] **M9.13** Publish operator guidance for resident, offload, streaming, resume, evaluation, benchmark, and failure recovery.
- [ ] **M9.14** Remove duplicate CLI/config dataclasses and flat dictionary config factories after migration tests pass.
- [ ] **M9.15** Remove duplicate auto-model/Hub orchestration paths after application API cutover.
- [ ] **M9.16** Remove report/CSV writes and global logging from mathematical/compression code.
- [ ] **M9.17** Remove mutable train/runtime mode switching from the deployment linear implementation.
- [ ] **M9.18** Archive legacy orchestration without deleting historical experiment chronology or evidence.
- [ ] **M9.19** Add deprecation errors/messages for unsupported legacy entry points and artifact versions.
- [ ] **M9.GATE** Verify every supported workflow has a tested new path and no supported numbered runfile imports legacy orchestration.

## Milestone 10 — Release qualification

Dependencies: Milestones 0–9  
Outcome: the rewrite is safe to use for research and deployment.

- [ ] **M10.1** Enforce formatting, lint, type, schema, import-boundary, unit, property, and tiny-pipeline tests in the default CPU CI lane.
- [ ] **M10.2** Enforce selected kernel parity, resident/streaming equivalence, and short-generation tests in the PR CUDA lane.
- [ ] **M10.3** Run the complete adapter, CUDA shape, resume-failure, evaluator, and compatibility suites nightly.
- [ ] **M10.4** Run designated-host kernel/layer/generation/streaming performance regression suites on schedule.
- [ ] **M10.5** Run periodic real large-model streaming/resume canaries.
- [ ] **M10.6** Pass all failure-injection points for atomic artifact and layer/block commits.
- [ ] **M10.7** Pass corruption, disk-full, source-change, OOM fallback, cancellation, and expired-lease tests.
- [ ] **M10.8** Pass security review for non-executable artifacts, path traversal, JSON limits, remote-code policy, secret redaction, and source integrity.
- [ ] **M10.9** Pass clean-install tests for research and runtime-only packages.
- [ ] **M10.10** Complete the legacy-versus-rewrite tiny and 1B parity reports with approved tolerances.
- [ ] **M10.11** Complete the resident-versus-streaming equivalence report.
- [ ] **M10.12** Complete a large-model canary report showing bounded memory, disk use, interruption, and resume.
- [ ] **M10.13** Complete the modified llama.cpp performance/correctness comparison on the agreed workload.
- [ ] **M10.14** Complete quick, standard, and full evaluation of the release-candidate artifact against the accepted baseline.
- [ ] **M10.15** Verify all supported artifact/config migrations with golden and numerical tests.
- [ ] **M10.16** Verify all warning codes, schemas, public interfaces, and compatibility tables are documented.
- [ ] **M10.17** Rebuild every release report from manifests/events/results without console-log parsing.
- [ ] **M10.18** Close or explicitly defer every requirement in the requirements-to-milestone traceability table.
- [ ] **M10.19** Publish release notes with correctness, quality, BPW, quantization cost, memory, prefill, decode, compatibility, and known limitations.
- [ ] **M10.GATE** Obtain final design, research-quality, runtime-performance, and operations sign-off and make the rewrite the default supported implementation.

## Post-release backlog

These tasks should not block the first rewrite release unless promoted by an ADR or requirement change.

- [ ] **P1** Implement a production distributed executor for multi-node/multi-GPU calibration and global tuning.
- [ ] **P2** Add remote/object-storage artifact-store implementations.
- [ ] **P3** Add OpenTelemetry or experiment-dashboard event export.
- [ ] **P4** Add additional packed layouts for new GPU architectures or CPU backends.
- [ ] **P5** Add automated representative-case selection from accumulated layer/block diagnostics.
- [ ] **P6** Add cost-aware experiment scheduling across available machines.
- [ ] **P7** Add safe artifact garbage-collection tooling with dry-run/reachability reports.
