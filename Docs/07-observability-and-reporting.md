# Observability, Diagnostics, and Run Reporting

## 1. Purpose

Logging must answer three questions:

1. What is the system doing and how far has it progressed?
2. Why did it make a decision or fail?
3. Did the model become better or worse, and where?

Free-form print output cannot reliably answer these questions after an hours-long run. Structured events are the source of truth; console text and Markdown reports are views.

## 2. Event envelope

Every event uses a common envelope:

```json
{
  "schema_version": 1,
  "timestamp": "2026-07-11T18:42:31.123456Z",
  "run_id": "run_01J...",
  "sequence": 1842,
  "severity": "info",
  "event": "layer_attempt_completed",
  "stage": "quantize_blocks",
  "span_id": "span_...",
  "parent_span_id": "span_...",
  "model_location": {
    "block": 12,
    "layer": "self_attn.v_proj"
  },
  "fields": {}
}
```

Event order is established by a per-run sequence number in addition to timestamps. Distributed workers include worker/rank identity and synchronize through the event collector.

## 3. Required spans

Timed spans include:

- source resolution and download;
- dataset preparation and tokenization;
- calibration total and per block;
- rank planning;
- block processing;
- non-factorized tuning;
- Hessian/objective preparation;
- each factorization attempt;
- scale fitting;
- factorized tuning;
- block output propagation;
- global tuning;
- packing and validation;
- each evaluator/task;
- each inference benchmark configuration.

Each span reports wall duration. GPU spans optionally report device-event duration, but GPU timing must not introduce hot-loop synchronization in ordinary runs.

The aggregate implementation and its versioned `profile.json`/`profile.md` artifacts are defined in
[Performance Profiling and Micro-Profiling](15-performance-profiling.md). Durable phase-event mirroring is
opt-in because measured JSONL flush overhead exceeds the ordinary macro-profiling budget on short runs;
the existing decision, stage, warning, and commit events remain first-class audit evidence.

## 4. Metric namespaces

Metrics use stable names and explicit units:

```text
resource.gpu.allocated_bytes
resource.gpu.reserved_bytes
resource.host.rss_bytes
resource.io.read_bytes
resource.io.write_bytes

calibration.tokens
calibration.clipped_fraction
calibration.stat_min
calibration.stat_max
calibration.zero_fraction

factorization.weighted_normalized_error
factorization.raw_normalized_error
factorization.primal_residual
factorization.dual_residual
factorization.wall_seconds

block.loss.source_reference
block.loss.entry_pre_quantization
block.loss.after_layer
block.loss.final_frozen_pre_kd
block.loss.final_post_kd
block.loss.final_vs_entry_absolute
block.loss.final_vs_entry_relative
block.loss.final_vs_source_absolute

model.effective_bpw
model.packed_bytes
quality.perplexity
runtime.decode_tokens_per_second
runtime.inter_token_latency_ms
```

Metrics distinguish counter, gauge, distribution, and scalar result semantics. Units do not live only in display labels.

## 5. Decision events

Every policy decision emits inputs, thresholds, result, and rationale. Examples:

- rank assigned or redistributed;
- outlier columns selected;
- factorization attempt accepted or rejected;
- retry allowed or blocked by rank/bit budget;
- scale-fit result accepted or rolled back;
- factorized tuning skipped because loss jump was small;
- cache reused or invalidated;
- activation store moved to another tier;
- runtime backend selected or rejected;
- evaluation promoted, stopped, or escalated.

This makes a report explain behavior without reconstructing it from source code and configuration alone.

## 6. Warning and diagnostic codes

Warnings have stable identifiers:

```text
NQ-CAL-001  Calibration statistic contains non-finite values
NQ-CAL-002  Calibration clipping rate exceeds configured diagnostic threshold
NQ-CAL-003  Calibration statistics are unstable across sample partitions
NQ-HES-001  Hessian regularization required repeated jitter escalation
NQ-FAC-001  Factorizer failed to converge within configured iterations
NQ-FAC-002  Export error materially exceeds latent/objective error
NQ-RNK-001  Retry requested but global bit budget rejected it
NQ-RNK-002  Retry added bits without sufficient reconstruction improvement
NQ-RNK-003  Outlier allocation added bits without sufficient block-loss benefit
NQ-TUN-001  Tuning restored an earlier best state after regression
NQ-TUN-002  Tuning recovered too little of the quantization loss jump
NQ-RUN-001  Resource fallback changed the execution plan
NQ-INF-001  Optimized backend unavailable; reference fallback selected
NQ-INF-002  Runtime output exceeded reference tolerance
NQ-EVL-001  Candidate failed quick-evaluation promotion gate
```

Each code has documentation containing:

- meaning;
- likely causes;
- evidence fields to inspect;
- whether the artifact remains valid;
- recommended next diagnostic;
- related tests.

## 7. Actionable model diagnostics

The reporting layer derives diagnoses from structured metrics. Initial rules include:

### Calibration

- near-zero importance for a large channel fraction;
- excessive clipping or unstable statistics between sample partitions;
- large divergence between forward-only and Fisher statistics on validation fixtures;
- low effective token diversity;
- order sensitivity outside the expected bound.

### Factorization and export

- ADMM residual plateau or divergence;
- large gap between objective-space, unwhitened, exported, and post-scale-fit error;
- repeated retry with negligible error reduction per added bit;
- raw error acceptable but objective-weighted error poor, or the reverse;
- outlier path consumes budget without block-loss benefit;
- rank at physical or configured cap.

### Block/model behavior

- layer reconstruction appears good but block-output loss jumps;
- tuning recovers little of the quantization jump;
- a small set of blocks dominates final degradation;
- candidate improves calibration loss but regresses held-out loss;
- quality changes without a corresponding representation-size change.

### Runtime

- fallback layers or layout conversions;
- excessive launch count;
- device idle gaps;
- decode throughput dominated by logits/sampling rather than quantized linears;
- memory growth across tokens;
- unexpected host-device transfers.

Diagnostic thresholds are recipe/versioned policy, not hard-coded report logic.

## 8. Required block-final error view

The report at `D:\dev\research\NanoQuant-OfficalCode\outputs\019-phase1-weight-errors.md` established a useful diagnostic that is mandatory in the rewrite.

For every block, structured results retain:

- original source/base-model reference loss;
- block-entry pre-quantization loss;
- loss after each accepted/tuned/frozen layer;
- post-block-refit loss when present;
- final frozen loss before model-level KD;
- final loss after KD when present;
- named absolute and relative deltas.

The default table renders per-layer residual error and a `Block final vs block-entry pre-quantization baseline` column. If the baseline denominator is near zero, the relative value is `n/a` while absolute values remain visible. The pre-KD table is never overwritten by post-KD results.

The report also retains the per-layer objective-weighted weight-reconstruction table from the same reference file. Together, these distinguish a matrix approximation problem from a block-behavior/tuning problem.

Every numbered compression also creates `weight-errors.md` in the run root before resident execution starts and
immediately publishes the same file under `Results/NNN/weight-errors.md`. The live report uses a stable hard link (or
symbolic-link fallback), so updates remain visible from `Results` without copying artifacts. It is rewritten after
each durable layer journal commit and each block commit. Partial layer rows are explicitly labeled `layer commit`;
when post-block refit finishes, the block's final durable rows replace them and the completed-block loss table grows.
Resume reconstructs the table from committed layer/block artifacts before new work begins.

Runs that started with an older worker can be backfilled without acquiring the GPU or mutating compression state:

```powershell
.\.venv\Scripts\python.exe tools\update_live_weight_errors.py evidence\m13\006-compress-and-benchmark-gemma-3-1b-it
```

## 9. Console output

The console is concise and optimized for active monitoring:

```text
[12/80] self_attn.v_proj attempt 1 rank=1536 err=0.284 threshold=0.350 18.4s ACCEPT
[12/80] block entry=1.91e-4 final-pre-KD=2.02e-4 delta=+1.10e-5 (+5.76%)
[resources] gpu=31.2/44.0 GiB host=42.8/64.0 GiB temp=38.1/500 GiB
```

Detailed tensors and histories remain in structured artifacts. Progress rendering must not synchronize CUDA or become a material part of stage time.

## 10. Run summary report

Every completed, failed, or interrupted run renders `reports/summary.md` with:

1. **Intent:** experiment number/runfile, purpose, hypothesis, owner, baseline, evidence tier.
2. **Outcome:** completed/failed/stopped, promotion decision, concise conclusion.
3. **Recipe delta:** changes from baseline and fully resolved recipe link.
4. **Inputs:** model, revision, datasets, tokenizer, calibration sample identity.
5. **Environment:** code revision, packages, CUDA, driver, hardware, executor.
6. **Resource plan versus actual:** peak memory, disk, I/O, stage timings.
7. **Quantization:** effective BPW, ranks, retries, outliers, per-layer reconstruction, and final block error versus named source/block-entry baselines before and after KD.
8. **Quality:** evaluation tier, metrics, deltas, uncertainty, failed gates.
9. **Performance:** packing, load, prefill, decode, memory, fallback coverage.
10. **Warnings and diagnostics:** grouped by severity and model location.
11. **Artifacts:** stable references and hashes.
12. **Recommended next action:** promote, investigate, rerun, or reject, with evidence.

The report generator reads only the manifest, events, and result artifacts. It never scrapes console logs.

Implementation note (2026-07-15): the typed run summary and Markdown renderer preserve experiment intent and the
complete `LauncherProvenance` envelope, explicitly identifying zero-argument numbered runfiles. They render the
allowlisted/redacted environment, elapsed manifest time, and every structured event timing or peak-memory cost
observation without summing nested timings. Explicit structured conclusion/recommended-action fields take priority;
status-aware defaults ensure older completed, failed, interrupted, running, and created runs still explain their
outcome and next step. Provenance mismatches, runfile arguments, timestamp problems, and malformed cost observations
remain visible as summary consistency warnings.

## 11. Comparison report

Candidate-versus-baseline reports include:

- semantic configuration diff;
- source/dataset/environment comparability assessment;
- stage reuse and compute cost;
- aligned per-layer and per-block deltas;
- Pareto view of quality, BPW, speed, and memory;
- statistically meaningful evaluation deltas;
- new/resolved warning codes;
- conclusion against predefined promotion gates.

When two runs are not directly comparable, the report says why rather than producing a misleading delta.

Implementation note (2026-07-15): `nanoquant.application.comparison_report` provides the immutable comparison
request/result contracts and deterministic Markdown renderer. Comparability is explicit rather than inferred;
required identity mismatches suppress metric-delta sections. Semantic config comparison ignores intent,
observability, and output-location roots by default while retaining numerical/runtime behavior changes. Artifact
reuse is grouped by producing stage, sampled metrics use the seeded paired-bootstrap evaluator, warning codes are
classified as new/resolved/shared, and the Pareto view keeps quality, representation, cost, quantization/runtime
memory, prefill, decode, and fallback dimensions separate. Promotion conclusions consume the versioned
`GateDecision` contract rather than accepting an unstructured label.

## 12. Event sinks and retention

Initial sinks:

- append-only local JSONL;
- human console renderer;
- in-memory sink for tests;
- optional OpenTelemetry-compatible exporter later.

Large traces, profiles, and per-step convergence arrays are separate compressed artifacts referenced by events. Routine event logs remain small enough to inspect and diff.

Retention classes:

- manifests, recipes, conclusions, and final metrics: permanent;
- committed model/calibration artifacts: policy controlled and reference counted;
- replay fixtures: explicit retention;
- temporary activations and traces: short-lived unless pinned by a run;
- incomplete temporary writes: removable after lease expiry.

## 13. Logging tests

Tests verify:

- required event fields and monotonic sequence numbers;
- secret redaction;
- every warning code links to documentation;
- run reports render for completed, failed, interrupted, and resumed runs;
- comparison aligns layers despite output ordering;
- console rendering never changes computation;
- performance mode does not add per-token synchronization;
- event schema migrations preserve old run readability.
- the Experiment 019-style weight-error and final-block tables render from structured fixtures with correct positive/negative/n-a semantics.
