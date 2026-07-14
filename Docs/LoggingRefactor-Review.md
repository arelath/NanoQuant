# Review: Ergonomic Logging and Run Discoverability

**Document reviewed:** [LoggingRefactor.md](LoggingRefactor.md)

**Review status:** Changes requested

**Reviewed against:** the current codebase on 2026-07-14

## 1. Verdict

The proposal identifies real problems and chooses the right product-level outcomes: a developer should be able to find a recent run immediately, read useful progress without decoding JSON, and inspect historical runs through the CLI. Keeping `events.jsonl` as structured evidence is also the correct foundation.

The implementation design is not ready as written. It assumes that all important executions use `RunDirectory`, treats a renderer as an observer of `JsonlEventSink` even though the proposed destinations have different severity thresholds, and does not define failure, resume, concurrency, or security semantics. Those gaps are material for NanoQuant because runs last many hours, are resumed in multiple processes, and currently write both audit events and a separate durability journal.

The proposal should be replaced by a design that separates event creation and routing from individual destinations, treats manifests as the source of truth for run discovery, and provides a shared run-observability composition path for both managed application runs and the direct resident research workflows.

## 2. What the proposal gets right

### 2.1 The developer problems are correctly identified

The current `ConsoleRenderer` is intentionally terse, and raw JSONL is awkward for casual monitoring. A stable CLI abstraction for locating and following a run is substantially better than asking researchers to know directory layouts. This is especially valuable while comparing long Gemma runs whose output directories have meaningful experiment names rather than generated run IDs.

### 2.2 Structured events should remain authoritative

The proposal preserves `events.jsonl` rather than replacing it with free-form text. That agrees with [07-observability-and-reporting.md](07-observability-and-reporting.md), which defines structured events as the source of truth and text/report output as views.

### 2.3 Destination-specific verbosity is useful

Keeping the console concise while making deeper diagnostics available elsewhere is desirable. The existing `ObservabilityConfig` already exposes `console_level`, `event_level`, `record_resource_interval_seconds`, and `record_admm_steps`, so the proposal is directionally consistent with the intended configuration model.

### 2.4 Human-readable output should be derived

Generating `run.log` from typed event fields avoids a second, unrelated logging vocabulary. That gives developers readable output without forcing report code to scrape text.

## 3. Findings that must be addressed

### 3.1 `RunDirectory` does not cover the runs that currently matter most

The proposal says that changing `RunDirectory` will create `run.log`, `LATEST`, and history entries for all new runs. That is not true today.

`bootstrap.run_experiment` uses `RunDirectory`, but the active resident parity path constructs `JsonlEventSink(request.output / "events.jsonl", ...)` directly in `resident_quantization.py`. The resident factorization slice and `tiny_pipeline.py` do the same. These paths accept caller-selected output directories such as `evidence/m4/gemma-metadata-isolated-v24-canary`; they do not create generated children under `OutputConfig.run_root` and do not necessarily have a `manifest.json`.

Consequently, wiring the feature only into `RunDirectory` would improve the foundation smoke path while missing the long-running research jobs that motivated the request. V2 needs a shared observability/session composition API and an explicit adoption plan for direct resident tools.

### 3.2 `JsonlEventSink` is the wrong composite boundary

The proposal attaches multiple filtered renderers to `JsonlEventSink`. This cannot implement the stated defaults consistently:

- `event_level=info` would prevent debug events from reaching a downstream `TextFileRenderer(file_level=debug)` if filtering occurs before persistence.
- If `JsonlEventSink` writes everything and only observers filter, then `event_level` has no effect.
- If each observer independently creates events, sequence numbers and timestamps can diverge.

The system needs one event router/dispatcher that validates severity, assigns the envelope and sequence exactly once, and sends the resulting `Event` to independently filtered destinations. JSONL, console, and text are destinations of that router; JSONL is not the parent of the other two.

The configuration must also define the relationship between thresholds. If `events.jsonl` remains the canonical audit trail, a derived `run.log` must not contain events absent from it. Either the canonical event threshold must be at least as verbose as all derived destinations, or invalid threshold combinations must be rejected. V2 should make this invariant explicit.

### 3.3 Observer failure currently propagates into the algorithm

`JsonlEventSink.emit` writes and flushes JSON, releases its lock, and then invokes its one observer. An exception from `ConsoleRenderer` or a proposed `TextFileRenderer` propagates to the caller. A full disk, closed console pipe, encoding error, or renderer bug could therefore fail quantization after the canonical event was already written.

For an S0 observability change, derived-view failures must not change mathematical execution or commit state. The design needs destination failure policies, a way to surface degraded logging without recursion, and tests proving that optional console/text failures do not fail the run. Failure of the configured canonical JSONL destination is different and should normally remain fatal because the audit contract can no longer be met.

### 3.4 The durability claim is inaccurate

`JsonlEventSink` flushes its Python file handle for every event but does not call `os.fsync`. Its docstring says each event is flushed durably, while `ProgressJournal.append` and manifest replacement do perform an `fsync`. The logging design should not promise crash durability that the implementation does not provide.

The design must deliberately choose and document an event durability window. Per-event `fsync` would be expensive and conflicts with profiling guidance; buffered or flushed events can be acceptable because `state/journal.jsonl`, not `events.jsonl`, is the authoritative resume boundary.

### 3.5 The progress journal must not be treated as a log

The proposal uses “logs” and “journals” interchangeably and suggests `inspect-run ... --path-only` prints “the latest journal.” NanoQuant has two distinct JSONL streams:

- `events.jsonl` is diagnostic/audit evidence.
- `state/journal.jsonl` records validated artifact commit boundaries and is authoritative for resume.

Human log rendering, severity filtering, tailing, or recovery must never rewrite or reinterpret the progress journal. CLI commands need unambiguous names and output—for example, `logs` resolves an event view, while `runs path --kind journal` explicitly resolves the progress journal.

### 3.6 The global ledger design is underspecified and can become stale

A creation-only row containing `status` immediately becomes stale when the manifest transitions from created to running, completed, failed, or interrupted. Multiple launchers can append concurrently, and Windows append/replace/locking behavior needs an explicit contract. A crash can occur between directory creation, manifest write, catalog update, and latest-pointer update.

The per-run `manifest.json` is already atomically replaced and is the natural source of truth. A global catalog should be a rebuildable discovery accelerator, not a second authoritative lifecycle database. Listing must fall back to scanning manifests, and a `runs rebuild-index` operation should recover from missing or corrupt index data.

### 3.7 `LATEST` semantics are ambiguous

“Latest” could mean most recently created, most recently updated, most recently completed, or the currently active run. Parallel runs make those different answers. The proposal also alternates between a symlink and JSON pointer without specifying a stable contract.

For Windows portability, V2 should use an atomically replaced JSON pointer with a schema version, run ID, root-relative path, and timestamp. `latest` should mean most recently registered/created unless the user requests a status filter. Commands should also support `--status running` and deterministic tie-breaking. A stale pointer must fall back to catalog/manifest discovery rather than fail mysteriously.

### 3.8 Raw command lines can leak secrets

The proposed history record includes `command_line`. Arguments commonly contain tokens, private paths, URLs with credentials, or environment-specific details. The current environment capture is allowlisted and tested for secret redaction, and `LauncherProvenance` already provides a safer typed location for launcher metadata and arguments.

The catalog must not store raw `sys.argv` by default. It should copy only approved manifest fields. If arguments are later recorded, they need an explicit allowlist/redaction policy and tests covering token-like values.

### 3.9 The configuration change is a real schema change

Adding `file_level` to `ObservabilityConfig` changes strict decoding, canonical serialization, emitted config reference data, and `config_hash`. The current decoder rejects unknown fields, validation only accepts `schema_version == 1`, and observability levels are unvalidated strings. Saying that no migration is strictly required obscures those effects.

V2 should specify whether this is a backward-compatible schema-1 additive field with a default or a schema-version bump. In either case it must update validation, generated configuration documentation, config-hash expectations, and migration policy. Severity should be a typed enum or receive strict validation.

### 3.10 Proposed high-cardinality logging conflicts with existing controls and profiling

Per-iteration ADMM logging can produce hundreds of records per attempt across many layers, with a flush and string/JSON formatting cost for each. [15-performance-profiling.md](15-performance-profiling.md) intentionally keeps aggregate profiling in memory and makes span-event mirroring opt-in because durable event emission exceeds the tiny-run overhead budget. The domain factorizer already returns trace points and records profiling counters.

`record_admm_steps` must remain the explicit opt-in gate. Iteration data should be sampled or summarized by default and emitted only after the factorizer returns, rather than injecting I/O-capable event dependencies into the pure domain loop. Retry math belongs in the application retry loop, where the policy inputs and decision are already available. Memory snapshots should reuse the profiling/resource sampler and report allocated, reserved, and peak bytes; backend-specific “fragmentation state” must be optional and bounded.

### 3.11 Domain instrumentation would violate the current dependency boundary

The proposal asks to inject `context.events.emit` calls into `domain/factorization.py` and `scale_fit.py`. Those modules are deliberately side-effect-free domain math. `StageContext` and `EventSink` belong at the application/stage boundary. Adding event I/O to domain functions would weaken determinism, complicate tests, and contradict the documented architecture.

Domain functions should return typed diagnostics or populate the existing recorder/trace results. Application stages decide which diagnostics become events.

### 3.12 Resume behavior for `run.log` is missing

Append mode alone does not define what happens when a process resumes an existing output directory:

- historical events may be absent if `run.log` was introduced after the original run;
- re-rendering all events can duplicate lines;
- a crash can write JSONL but not text, leaving the derivative behind;
- two processes writing the same run can interleave output despite the in-process lock.

Because `run.log` is a derivative, it should be rebuildable from `events.jsonl`. The simplest safe contract is to rebuild it atomically at session open and then append new events, or to store a last-rendered sequence sidecar. The design must prevent multiple active writers for one event stream or explicitly provide a cross-process collector.

### 3.13 Multiline formatting harms tailing and parsing

Pretty-printing arbitrary nested fields and tracebacks across multiple lines makes `tail`, `grep`, and sequence correlation harder. The default text view should render exactly one physical line per event with deterministic key ordering and escaped/compact values. A CLI can provide a separate expanded view for one event or a traceback when needed.

### 3.14 Existing event-schema weaknesses should not be amplified

Severity is an arbitrary string, event fields are unbounded `object` values, and JSON serialization uses `default=str`, which can silently turn unexpected values into unstable representations. Adding high-volume debug events without field-size, type, and redaction rules would make logs unreliable and potentially enormous.

V2 should define allowed severities, bounded JSON-compatible field values, stable names/units, maximum rendered line size, and redaction. Large tensors, arrays, optimizer state, and full configurations belong in artifacts, with references in events.

### 3.15 The test plan is incomplete

Current tests cover monotonic sequences across a reopen and atomic manifest replacement, but not renderer filtering, destination failures, concurrent catalog updates, stale pointers, crash recovery, resume catch-up, secret redaction, old runs, or high-volume overhead. The proposal needs acceptance tests for each of these behaviors before it can claim S0.

## 4. Recommended scope changes

The first implementation should deliver discoverability and safe human rendering before adding new algorithm events:

1. Define the event router, typed levels, canonical/derived invariants, and failure policy.
2. Add a manifest-backed `RunCatalog` and portable latest pointer with scan/rebuild fallback.
3. Add CLI discovery and follow commands that work for old JSONL-only runs.
4. Introduce a shared run-observability session and adopt it in managed and resident entry points.
5. Add a deterministic single-line `run.log` derivative with resume/rebuild semantics.
6. Add bounded diagnostics at application boundaries, using existing `record_admm_steps`, profiling, and resource-sampling controls.

This ordering provides immediate DX value without first increasing event volume or changing algorithm hot loops.

## 5. Required acceptance criteria for a replacement design

A replacement design should not be approved until it specifies and tests all of the following:

- managed foundation runs and direct resident parity runs are both discoverable;
- one component creates event envelopes and routes them to filtered destinations;
- the canonical event stream is at least as verbose as every derived event view;
- console/text failures cannot change quantization or commit results;
- canonical event-sink failure behavior is explicit;
- event and progress-journal roles are unambiguous;
- catalog and latest data are concurrency-safe, atomically updated, and rebuildable from manifests;
- `LATEST` resolution is deterministic under parallel runs and stale pointers;
- no raw secrets are copied into global metadata or text logs;
- resume does not duplicate or silently omit text-log history;
- legacy JSONL-only runs remain inspectable;
- per-iteration diagnostics remain opt-in and meet a measured overhead budget;
- strict config decoding, validation, hashing, and generated references are updated;
- CLI paths, exit codes, follow behavior, and ambiguity errors are specified for Windows and POSIX.

The accompanying [LoggingRefactorV2.md](LoggingRefactorV2.md) supplies a design that satisfies these requirements.
