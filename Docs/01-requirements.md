# Requirements and Acceptance Criteria

This document turns the rewrite goals into testable requirements. Requirement identifiers are stable and should appear in implementation issues, design reviews, tests, and release reports.

## 1. Supported workflows

The rewrite must support four primary workflows:

1. **Algorithm development:** change a factorizer, objective, allocator, outlier policy, or tuning method and obtain a useful result without a full model run.
2. **Quantization production:** quantize a supported model reproducibly, resume after interruption, and emit a portable packed artifact.
3. **Quality investigation:** identify where a run regressed, reproduce the affected layer or block, and compare it with a baseline.
4. **Inference deployment:** load a packed artifact without training dependencies and run a validated high-performance backend.

## 2. Functional requirements

### Configuration and planning

- **CFG-001:** There is exactly one versioned run-configuration schema.
- **CFG-002:** CLI, YAML/JSON recipes, and the Python API resolve to the same immutable configuration object.
- **CFG-003:** Invalid combinations fail before downloading or allocating model weights.
- **CFG-004:** Defaults are visible in the resolved recipe and never depend silently on the entry point.
- **CFG-005:** The planner estimates source size, output size, peak GPU/CPU memory, activation storage, temporary disk, and major stage costs.
- **CFG-006:** A run has a stable identity derived from normalized semantic inputs; presentation-only fields do not invalidate computational caches.

### Experiment provenance

- **EXP-001:** Promoted research experiments may be launched by monotonically numbered, descriptive Python files.
- **EXP-002:** A numbered runfile executes without experiment-defining command-line parameters.
- **EXP-003:** A runfile constructs the canonical `RunConfig` or loads one canonical colocated recipe and contains no copied quantization, logging, checkpoint, or evaluation orchestration.
- **EXP-004:** The run manifest records experiment number, launcher path, repository-relative path when available, launcher content hash, and code revision.
- **EXP-005:** A completed experiment number is not reused for a semantically different configuration; a change receives the next number or an explicit fork/run relationship.
- **EXP-006:** Generic YAML/CLI launch remains supported for automation and sweeps; it does not eliminate numbered zero-argument runfiles.

### Calibration and quantization

- **QNT-001:** Calibration emits a versioned `CalibrationStats` artifact independent of the execution strategy that produced it.
- **QNT-002:** Rank allocation emits an inspectable `QuantizationPlan` before any source weight is replaced.
- **QNT-003:** Each factorizer implements one typed interface and returns reconstruction metrics with its factors.
- **QNT-004:** Objective selection, rank allocation, retry policy, outlier selection, scale fitting, and tuning are independent strategies.
- **QNT-005:** Each accepted layer result records its complete lineage: source tensor identity, statistics identity, plan, random seed, attempt history, and metrics.
- **QNT-006:** Quantization is deterministic within the documented tolerance for the same recipe, source revision, environment class, and deterministic mode.
- **QNT-007:** The system can replay a captured layer or block without loading the entire source model.

### Checkpoint and resume

- **RES-001:** Stages and loop units have explicit commit boundaries.
- **RES-002:** A commit is atomic: after a crash, an output is either valid and discoverable or ignored as incomplete.
- **RES-003:** Resume verifies artifact hashes and semantic compatibility before reuse.
- **RES-004:** Completed layers or blocks are not repeated unless explicitly invalidated.
- **RES-005:** A run can resume after process termination, CUDA OOM, host reboot, or evaluation interruption.
- **RES-006:** Retry attempts are deterministic and do not accidentally consume a different random stream after resume.
- **RES-007:** Operators can restart from a named stage or fork a prior run with a changed downstream recipe.

### Model scale and resources

- **SCL-001:** A resident executor supports models that fit entirely in GPU memory.
- **SCL-002:** A streaming executor supports models whose source weights fit only in CPU memory or on disk.
- **SCL-003:** Algorithm components are unchanged between resident and streaming executors.
- **SCL-004:** Streaming peak model-weight memory is bounded by the active block/layer plus declared workspaces.
- **SCL-005:** Activation stores support CUDA, pinned RAM, pageable RAM, and memory-mapped disk.
- **SCL-006:** Dense Hessians are subject to explicit memory limits and have diagonal, block-diagonal, or low-rank alternatives.
- **SCL-007:** Resource exhaustion produces a diagnostic and a documented fallback; the system never silently begins uncontrolled paging.
- **SCL-008:** The preflight plan refuses to start when known minimum disk or memory requirements cannot be met.

### Inference

- **INF-001:** Training and packed inference modules are different types.
- **INF-002:** A clear PyTorch reference backend defines output correctness.
- **INF-003:** Every optimized backend declares supported device, dtype, shapes, ranks, batch regimes, and alignment requirements.
- **INF-004:** Backend dispatch is observable and never silently falls back without recording why.
- **INF-005:** Prefill, single-token decode, sampling, cache updates, model loading, and end-to-end generation are benchmarked separately.
- **INF-006:** Kernel and end-to-end outputs match the reference within backend-specific tolerances.
- **INF-007:** Packed artifacts load without calibration, optimizer, dataset, or experiment-framework dependencies.
- **INF-008:** Performance and packing comparisons record the modified NanoQuant llama.cpp implementation revision and `ggml/src/ggml-cuda/nanoquant.cu` source hash when it is the reference.

### Logging and reporting

- **OBS-001:** All events have a timestamp, run ID, stage, severity, event name, and structured fields.
- **OBS-002:** Layer events include model location, tensor shape, rank, objective, attempt, timing, memory, and decision metrics where applicable.
- **OBS-003:** Human console output is derived from structured events, not parsed later from free-form text.
- **OBS-004:** Expected warnings have stable codes and remediation text.
- **OBS-005:** A run report contains purpose, hypothesis, baseline, recipe, environment, progress, resource use, results, comparisons, warnings, and conclusion.
- **OBS-006:** Secrets, access tokens, private dataset examples, and prompts marked sensitive are redacted.
- **OBS-007:** Every block records source-reference, block-entry, after-layer, final-frozen-pre-KD, and—when applicable—final-post-KD losses with named absolute and relative comparisons.
- **OBS-008:** Reports render the per-layer weight-reconstruction table and final block error versus its explicitly named pre-quantization/base-model baseline; pre-KD results remain available after KD.

### Evaluation

- **EVAL-001:** Evaluation supports smoke, quick, standard, and full tiers.
- **EVAL-002:** Every metric records dataset/task revision, sample selection, prompt format, few-shot setup, seed, batch settings, and evaluator version.
- **EVAL-003:** Candidate reports compare against a named baseline and show absolute and relative deltas.
- **EVAL-004:** Stochastic or sampled metrics include uncertainty or repeated-run variability where meaningful.
- **EVAL-005:** Quality, storage, quantization cost, memory, prefill, and decode are reported as separate dimensions.
- **EVAL-006:** Promotion gates can stop an obviously bad run before expensive evaluation.

### Testing

- **TST-001:** Mathematical functions have CPU unit tests with small deterministic tensors.
- **TST-002:** Numerical invariants and serialization round trips have property tests.
- **TST-003:** Every model adapter passes the same contract suite.
- **TST-004:** Every optimized backend passes reference parity across a shape matrix.
- **TST-005:** End-to-end tests cover quantize, save, load, resume, evaluate, and infer.
- **TST-006:** Failure-injection tests interrupt each commit boundary and verify resumed equivalence.
- **TST-007:** Performance regression tests use stable benchmark hosts and statistical thresholds, not ordinary shared CI runners.

## 3. Non-functional requirements

### Maintainability

- **NFR-001:** Dependencies point inward: domain math has no dependency on application orchestration or infrastructure.
- **NFR-002:** No untyped configuration dictionaries cross domain boundaries.
- **NFR-003:** Public interfaces and artifact schemas carry explicit versions.
- **NFR-004:** Module ownership and extension instructions are documented.
- **NFR-005:** Model-family exceptions live in adapters, not generic orchestration.

### Performance and cost

- **NFR-010:** Every major stage emits wall time, GPU active time when available, peak allocated/reserved VRAM, peak host memory, and bytes read/written.
- **NFR-011:** Cached stage reuse is visible and validated rather than inferred from file existence.
- **NFR-012:** A quick experiment has a declared maximum time or sample budget.
- **NFR-013:** Inference comparisons use a shared benchmark protocol and report median plus tail behavior.

### Reliability

- **NFR-020:** Artifacts are checksummed and written atomically.
- **NFR-021:** Schema readers reject unsupported future versions and clearly report migrations for old versions.
- **NFR-022:** Runs remain inspectable even when they fail.
- **NFR-023:** Partial output and temporary data are distinguishable from committed artifacts.

### Security and supply chain

- **NFR-030:** Loading a packed artifact does not require arbitrary pickle execution.
- **NFR-031:** Remote model code is disabled by default or explicitly recorded and isolated when required.
- **NFR-032:** Source model revisions and downloaded file hashes are recorded.
- **NFR-033:** Environment capture records package versions without copying credentials or unrelated environment variables.

## 4. Performance acceptance protocol

The observed 20 versus 400 tokens/second comparison must be reproduced under one protocol before it becomes a rewrite gate. At minimum, record:

- exact model and quantized artifact;
- hardware, power mode, driver, CUDA, and CPU thread configuration;
- batch size;
- prompt token counts and generated token counts;
- dtype and cache dtype;
- eager/compiled mode;
- sampling algorithm and its location;
- whether tokenizer time, first-token time, synchronization, and model load are included;
- warm-up count and measured repetitions;
- median, p10, and p90 tokens/second;
- peak VRAM and host memory.

Initial inference gates should be expressed relative to an eligible reference on the same host:

1. no numerical parity failure;
2. no unexplained backend fallback;
3. at least 70% of the fastest compatible reference's steady-state decode throughput;
4. no more than 10% regression from the last accepted NanoQuant runtime baseline;
5. a profile accounting for at least 90% of measured wall time when the relative target is missed.

The 70% value is a starting engineering gate, not a claim that NanoQuant and llama.cpp have identical representations or kernel opportunities.

## 5. Prototype-speed acceptance protocol

A change should move through progressively more expensive evidence:

| Feedback level | Target | Intended decision |
| --- | ---: | --- |
| Pure unit test | seconds | Is the local math or policy implementation coherent? |
| Captured layer replay | under 60 seconds when factorizer settings permit | Does the change improve its intended layer metric? |
| Captured block replay | under 5 minutes | Does local improvement survive block behavior? |
| Tiny-model smoke | under 10 minutes | Does the full pipeline still work? |
| Representative subset | explicitly budgeted, normally under 30 minutes | Is there enough evidence for a full run? |
| Full quantization/evaluation | hours or days | Is this a candidate result? |

Hardware and recipe differences must be attached to the recorded target rather than hidden behind a universal wall-clock promise.

## 6. Release acceptance

A rewrite release cannot be called production-ready until:

- all `must` requirements above are implemented or explicitly deferred in a signed design decision;
- the legacy parity recipe passes;
- at least one resident 1B run and one streaming large-model dry run pass;
- crash/restart equivalence passes at every commit boundary;
- packed inference parity passes for every supported backend;
- quick and full evaluation reports are reproducible from their manifests;
- the performance baseline and comparison methodology are published with the release artifacts.
