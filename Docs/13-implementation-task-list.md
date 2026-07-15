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
- [ ] **M6.11** Implement conversion/parity tests between rewrite frozen state, packed runtime state, and modified llama.cpp/GGUF state where semantically compatible.
- [ ] **M6.12** Integrate or port the initial NanoQuant CUDA backend with explicit version/capabilities.
- [ ] **M6.13** Implement separate prefill and decode execution plans.
- [ ] **M6.14** Implement generation-engine prompt batching, positions, attention metadata, stopping, and deterministic mode.
- [ ] **M6.15** Implement bounded/static or paged KV-cache management and verify positions/padding across batches.
- [ ] **M6.16** Keep packing, capability discovery, device transfers, and allocator cleanup outside the token hot loop.
- [ ] **M6.17** Implement device-side greedy and configured sampling paths with explicit synchronization boundaries.
- [x] **M6.18** Add logical-reference versus factorized-reference tests over real model shapes.
- [ ] **M6.19** Add packed-backend parity tests over the full declared shape/rank/dtype/outlier matrix.
- [ ] **M6.20** Add long-enough generation tests for output parity, cache correctness, and memory growth.
- [ ] **M6.21** Implement kernel, layer, block, prefill, decode, and end-to-end benchmark commands with JSON output.
- [ ] **M6.22** Verify a clean runtime-only installation can load a packed artifact and generate text.
- [ ] **M6.GATE** Load and run a packed NanoQuant artifact through reference and CUDA backends with complete numerical parity coverage and no research-package dependency.

## Milestone 7 — Close the inference performance gap

Dependencies: Milestone 6 correctness and benchmark suite; Milestone 0 profiles  
Outcome: runtime performance is measured, explained, and competitive with the modified llama.cpp reference.

- [ ] **M7.1** Freeze the apples-to-apples NanoQuant versus modified llama.cpp benchmark protocol and artifact pair.
- [ ] **M7.2** Record warm-up, repetitions, median/p10/p90, TTFT, prefill throughput, inter-token latency, decode throughput, memory, and fallback count.
- [ ] **M7.3** Capture a new end-to-end profile and account for at least 90% of wall time before choosing optimizations.
- [ ] **M7.4** Remove per-token Python/device synchronizations, scalar reads, repeated capability checks, and layout conversions.
- [ ] **M7.5** Ensure every intended NanoQuant layer dispatches to the optimized backend or records an actionable unsupported reason.
- [ ] **M7.6** Port/evaluate packed sign-word loads, aligned vector loads, lane-zero broadcasts, and branchless sign-bit application from `nanoquant.cu`.
- [ ] **M7.7** Port/evaluate sign-aware FMA and multiple-accumulator decode loops from `nanoquant.cu`.
- [ ] **M7.8** Implement and tune dtype-specialized contiguous fast paths for the actual runtime input/scale dtypes.
- [ ] **M7.9** Evaluate fused first-stage Q/K/V or other shared-input projections using real block profiles.
- [ ] **M7.10** Tune decode kernels across real rank/shape/alignment cases and report achieved memory bandwidth/occupancy.
- [ ] **M7.11** Tune prefill kernels independently across representative token and batch sizes.
- [ ] **M7.12** Fuse scale, bias, and salient-outlier operations only where end-to-end profiles show a net benefit.
- [ ] **M7.13** Optimize outlier paths for zero/small/common counts and ensure specialization does not regress general cases.
- [ ] **M7.14** Measure and optimize KV cache, attention, final vocabulary projection, and sampling after quantized linears no longer dominate.
- [ ] **M7.15** Compare eager and compiled/static decode-step execution with stable shape/correctness coverage.
- [ ] **M7.16** Establish designated-host performance CI with raw sample retention and environment health checks.
- [ ] **M7.17** Add no-more-than-10% regression gates against the accepted NanoQuant runtime baseline.
- [ ] **M7.18** Reach at least 70% of the fastest compatible modified llama.cpp reference throughput on the agreed workload, or produce an accepted profile-backed gap analysis and follow-up plan.
- [ ] **M7.19** Re-run numerical, artifact-size, VRAM, and quality gates after every accepted layout/kernel change.
- [ ] **M7.GATE** Publish a reproducible NanoQuant-versus-llama.cpp report with source/kernel/artifact hashes, workload, profiles, parity, quality/BPW, and explained end-to-end performance.

## Milestone 8 — Complete evaluation, diagnostics, and self-documenting reports

Dependencies: Milestones 1, 4, and 6; can progress in parallel with Milestone 7  
Outcome: runs produce cheap-to-expensive decision evidence and actionable reports.

- [x] **M8.1** Implement the versioned evaluator registry and immutable evaluator specifications.
- [x] **M8.2** Implement calibration, quick-decision, and final-evaluation partitions with content hashes and overlap detection.
- [ ] **M8.3** Implement artifact-structure and packed-reference parity smoke evaluators.
- [x] **M8.4** Implement token-accurate negative-log-likelihood/perplexity evaluation with BOS/EOS, causal shift, padding, stride, and partial-window tests.
- [ ] **M8.5** Implement the selected zero/few-shot task evaluators with pinned dataset/task/prompt revisions.
- [ ] **M8.6** Implement deterministic generation sanity/regression cases.
- [ ] **M8.7** Implement long-context evaluation where supported by the model/runtime plan.
- [x] **M8.8** Implement smoke, quick, standard, and full evaluation tiers from the registry.
- [x] **M8.9** Implement predefined promotion, rejection, and inconclusive decisions with immutable gate policy.
- [ ] **M8.10** Implement paired comparisons, bootstrap/appropriate confidence intervals, repeated-run variability, and minimum meaningful deltas.
- [ ] **M8.11** Implement exact effective core/artifact BPW, bytes, memory, quantization cost, and runtime metrics as separate dimensions.
- [x] **M8.12** Preserve per-layer objective-weighted reconstruction tables for every run.
- [x] **M8.13** Preserve the Experiment 019-style final-block-versus-block-entry table with positive, negative, and `n/a` semantics.
- [ ] **M8.14** Preserve source/base-model, block-entry, final-pre-KD, and final-post-KD snapshots and named comparisons.
- [ ] **M8.15** Implement diagnostic rules for calibration instability, Hessian conditioning, ADMM plateau, export gaps, ineffective retry/outliers, poor tuning recovery, and runtime fallback.
- [ ] **M8.16** Implement complete summary reports for completed, failed, interrupted, resumed, and forked runs.
- [ ] **M8.17** Implement candidate-versus-baseline reports with semantic config diff, artifact reuse, per-layer/block alignment, uncertainty, warnings, and Pareto dimensions.
  `tools/compare_block_trajectories.py` now provides the block-alignment slice: it selects the latest journal
  identity, rejects stale/noncontiguous prefixes, resolves committed block artifacts, and compares any number of
  named legacy post-refit trajectories with JSON/Markdown deltas. Optional legacy rank-utility CSVs add exact
  prefix rank-sum/mismatch checks and scoped BPW. Rewrite BPW includes every committed bit-cost category; the
  legacy CSV's rank-dependent BPW includes binary factors and middle scales but excludes pre/post scales and
  outliers, so it is explicitly not presented as like-for-like total BPW. Config/artifact reuse, uncertainty,
  warnings, and Pareto reporting remain open.
- [ ] **M8.18** Include experiment number, zero-argument runfile path/hash, purpose, hypothesis, baseline, environment, cost, conclusion, and recommended next action.
- [ ] **M8.19** Implement evaluator and task-result caching using complete semantic identities.
- [ ] **M8.20** Add golden report tests using Experiment 019 data and synthetic near-zero denominators.
- [ ] **M8.21** Validate evaluator batching, caching, sample limiting, distributed reduction, and known-logit results.
- [ ] **M8.GATE** Demonstrate that a candidate can progress from layer replay through quick/standard/full evaluation and that its run directory alone explains intent, execution, issues, cost, results, and comparison.

## Milestone 9 — Migrate supported workflows and cut over

Dependencies: Milestones 4–8 required for their respective workflows  
Outcome: supported users no longer need legacy orchestration.

- [ ] **M9.1** Create a migration inventory for every numbered legacy experiment and its replacement status.
- [ ] **M9.2** Convert supported historical experiments into thin numbered zero-argument runfiles while preserving numbers, names, purpose, and lineage.
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
