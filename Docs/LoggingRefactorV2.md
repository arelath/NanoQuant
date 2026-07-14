# Logging and Run Discoverability V2

**Status:** Proposed

**Audience:** Maintainers, tooling engineers, and algorithm researchers

**Supersedes:** [LoggingRefactor.md](LoggingRefactor.md)

**Related:** [LoggingRefactor-Review.md](LoggingRefactor-Review.md), [07-observability-and-reporting.md](07-observability-and-reporting.md), [15-performance-profiling.md](15-performance-profiling.md)

## 1. Summary

NanoQuant will provide one safe event-routing pipeline, a rebuildable manifest-backed run catalog, portable `latest` resolution, deterministic human-readable logs, and CLI commands for finding and following runs.

`events.jsonl` remains the canonical diagnostic event stream. `run.log` is a disposable, reproducible view of that stream. `state/journal.jsonl` remains the authoritative artifact-commit and resume journal and is never filtered, rendered, or repaired by the logging system. Per-run manifests remain the source of truth for identity and lifecycle; the root catalog and latest pointer accelerate discovery but can always be rebuilt.

The design covers both schema-driven runs created through `bootstrap.run_experiment` and direct resident research runs that write into caller-selected evidence directories.

## 2. Goals

- Resolve the latest or a selected run without manual directory traversal.
- List recent runs using stable manifest metadata and status.
- Follow a concise human event view on Windows and POSIX.
- Preserve structured evidence and stable sequence ordering.
- Allow different console, canonical-event, and text-view verbosity without creating unaudited derived records.
- Prevent optional logging destinations from changing algorithm or commit behavior.
- Make catalog, pointer, and text views recoverable after crashes or partial writes.
- Add useful algorithm diagnostics at application boundaries without putting I/O in pure domain math or materially slowing normal runs.
- Support old runs that contain only some of `manifest.json`, `events.jsonl`, and `state/journal.jsonl`.

## 3. Non-goals

- Replacing `profile.json`, `profile.md`, CUDA traces, or the in-memory profiling aggregator.
- Turning the progress journal into a diagnostic log.
- Providing a distributed log collector in this iteration.
- Adding log rotation to individual run directories. Artifact-retention tooling may remove derivative `run.log` files and regenerate them later.
- Recording arbitrary Python objects, tensors, full model state, or unredacted command lines in events.
- Guaranteeing that every event survives machine power loss. Resume correctness is provided by the progress journal and committed artifacts, not by diagnostic logs.
- Defining a remote observability backend. The router interface may support one later.

## 4. Terminology and sources of truth

| Data | Purpose | Authority | Recovery behavior |
|---|---|---|---|
| `manifest.json` | Run identity, provenance, lifecycle, resolved configuration, artifacts | Authoritative | Atomically replaced by run lifecycle code |
| `events.jsonl` | Ordered diagnostic and audit events | Canonical diagnostic stream | Valid prefix is readable; a malformed trailing record is ignored/reported |
| `state/journal.jsonl` | Validated layer/block commit boundaries used for resume | Authoritative resume stream | Existing `ProgressJournal` rules apply; logging code never writes it |
| `run.log` | Single-line human rendering of events | Derived | May be rebuilt atomically from `events.jsonl` |
| `.nanoquant-runs/index.jsonl` | Run discovery acceleration | Rebuildable index | Rebuilt by scanning registered roots/manifests |
| `.nanoquant-runs/latest.json` | Most recently registered run pointer | Rebuildable hint | Validated and replaced from index/manifest discovery |
| `profile.json` / `profile.md` | Aggregated performance evidence | Profiling artifact | Governed by the profiling design, not this event pipeline |

The word “journal” refers only to state required for resume. CLI help and code names must not call `events.jsonl` or `run.log` a journal.

## 5. Current constraints

The implementation must account for these existing facts:

- `JsonlEventSink` currently creates event envelopes, assigns sequences, writes JSONL, and optionally calls one observer.
- `bootstrap.run_experiment` is the only main path that creates a generated `RunDirectory` and full manifest.
- Resident quantization, resident factorization slices, and the tiny pipeline construct event sinks directly in caller-selected output directories.
- Long resident runs resume by reopening the same `events.jsonl` and `state/journal.jsonl` in later processes.
- `ObservabilityConfig.console_level` and `event_level` exist but are not currently enforced.
- `record_admm_steps` and resource sampling already express opt-in high-cardinality behavior.
- Profiling deliberately avoids durable per-phase event writes by default.
- Windows is a primary platform; symlink creation and sharing semantics cannot be assumed.

## 6. Event architecture

### 6.1 Separate envelope creation from destinations

Replace the composite-observer role of `JsonlEventSink` with an `EventRouter` implementing the existing `EventSink` port:

```python
class EventRouter:
    def __init__(
        self,
        run_id: str,
        sequence_store: EventSequenceStore,
        destinations: tuple[EventDestination, ...],
    ) -> None: ...

    def emit(
        self,
        stage: str,
        severity: Severity,
        name: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        **fields: JsonField,
    ) -> Event: ...
```

The router performs these steps exactly once per emitted event:

1. validate stage, event name, severity, and fields;
2. acquire the per-process sequence lock;
3. assign the next sequence and UTC timestamp;
4. construct one immutable `Event` envelope;
5. deliver the envelope to destinations in configured order;
6. return the envelope.

Initial destinations are:

- `JsonlEventDestination` for the canonical stream;
- `ConsoleEventDestination` for active monitoring;
- `TextEventDestination` for `run.log`.

The existing `JsonlEventSink` name may remain as a compatibility wrapper for tests and narrow callers, but new composition code uses `EventRouter`.

### 6.2 Severity type and threshold ordering

Define a string enum with this order:

```text
trace < debug < info < warning < error < critical
```

Every destination has an inclusive minimum threshold. Unknown severities fail at configuration validation or event construction rather than silently sorting incorrectly.

The canonical invariant is:

```text
event_level <= min(console_level, file_level)
```

where a lower level is more verbose. This ensures every console or text record also exists in `events.jsonl`. Configuration that violates the invariant fails with a precise validation code. Defaults are:

```yaml
observability:
  event_level: info
  console_level: info
  file_level: info
```

Setting `file_level: debug` therefore also requires `event_level: debug`. This makes the storage and performance cost explicit. `run.log` never contains a debug record that is missing from canonical evidence.

### 6.3 Destination failure policy

Destinations have declared roles:

- **required:** canonical JSONL by default;
- **optional:** console and human text by default.

A required destination failure raises `EventWriteError`. The current operation must stop because the configured audit contract is broken. An optional destination failure is quarantined for the rest of the process and does not propagate into quantization. The router writes a compact diagnostic directly to the remaining healthy required destination, bypassing the failed destination and avoiding recursion:

```text
observability.destination_disabled
destination=text
error_type=OSError
```

If no required destination remains, the router raises. Tests must prove that optional destination failures do not alter returned stage results, committed artifacts, or progress-journal contents.

### 6.4 Event field contract

Fields are bounded JSON-compatible values: `null`, booleans, finite numbers, strings, and bounded lists/maps of the same. Production code must not rely on `json.dumps(default=str)`.

Rules:

- field keys and event/stage names use stable lowercase dotted or snake-case identifiers;
- values with units encode the unit in the field name, following the namespaces in the observability design;
- a string field is capped at a configured safe length in the text view and canonical size validation rejects pathological payloads;
- secrets are redacted before routing using an allowlist for known provenance/config fields and key-pattern rejection for token/password/credential fields;
- tensors, large arrays, optimizer states, and full configurations are persisted as artifacts and referenced by ID;
- dictionaries are rendered in deterministic key order;
- NaN and infinity are rejected or represented as explicit diagnostic strings by the producer, never emitted as invalid JSON.

The first implementation should cap a canonical event at 64 KiB and a rendered text line at 8 KiB. Oversized optional detail is replaced with an artifact reference or a truncation marker carrying original byte length.

### 6.5 Sequence and writer ownership

One process owns the event writer for a run session. The existing per-process thread lock remains, but it is not presented as cross-process serialization. Managed runs already hold a per-run lease. Direct resident entry points must acquire the same output-session lease before opening the router.

Resume in a later process reads the last valid JSONL record and continues at the next sequence. The initial implementation may scan the file, but should read backward from the end in a follow-up performance change so reopen cost does not grow with the full event history. A malformed trailing line is reported and preserved; the next writer must either truncate only that invalid tail under the run lease or refuse to append with an actionable recovery command. It must never skip to an ambiguous sequence.

## 7. Human-readable `run.log`

### 7.1 Format

The default format is exactly one physical UTF-8 line per event:

```text
2026-07-14T16:12:33.482Z 0001842 INFO    quantize-blocks block.completed block=5 loss=3.5971968174 wall_seconds=421.8
```

Properties:

- UTC RFC 3339 timestamp;
- zero-padded event sequence;
- fixed-width uppercase severity;
- stage and event name;
- fields sorted by key and rendered with compact JSON escaping;
- newline, tab, and control characters escaped;
- no locale-dependent number or date formatting.

This format supports `tail`, `Select-String`, `grep`, and exact correlation with `events.jsonl`. An expanded CLI view may pretty-print one selected event, but the stored file remains one-event-per-line.

### 7.2 Derivation and resume

`run.log` is disposable. At run-session open:

1. if the file is absent, render the valid canonical event prefix to a temporary file and atomically replace `run.log`;
2. if its final sequence matches canonical JSONL, append only new routed events;
3. if it is behind, append the missing canonical events before accepting new ones;
4. if it is ahead, malformed, or uses an unsupported renderer version, rebuild atomically.

Store renderer metadata in `.run-log-state.json` with schema version, renderer version, and last rendered sequence. The sidecar is a cache hint, not authority; mismatches are resolved from `events.jsonl`.

The CLI command `nanoquant logs RUN --rebuild` performs the same atomic rebuild. Removing `run.log` is always safe.

### 7.3 Flush and durability

Console output is immediate. JSONL and text destinations flush at event boundaries by default so follow commands see progress promptly, but they do not claim power-loss durability and do not call `fsync` per event. The router closes and flushes all destinations at process/session exit.

The progress journal retains its existing validated-artifact-before-append and `fsync` semantics. This distinction is documented in CLI help and operator guidance.

## 8. Run sessions and adoption

### 8.1 Shared composition API

Introduce a composition-root helper rather than constructing sinks throughout the codebase:

```python
@contextmanager
def open_run_observability(
    output: Path,
    run_id: str,
    observability: ObservabilityConfig,
    *,
    manifest: RunManifest | None,
    catalog: RunCatalog | None,
) -> Iterator[RunObservability]: ...
```

`RunObservability` contains the `EventSink`, output paths, and close/finalization behavior. It does not expose artifact-commit operations or mutate numerical requests.

Managed `run_experiment` supplies its manifest and catalog. Direct resident entry points supply their caller-selected output and a lightweight manifest described below. Tests may continue to use a compatibility `JsonlEventSink` when they do not need catalog/text behavior.

### 8.2 Lightweight manifests for direct research runs

Direct resident runs need discoverable metadata even when they are not children of `OutputConfig.run_root`. At session creation they write an atomic `manifest.json` in the selected output directory using the existing domain manifest where possible. Required fields are:

- stable run ID distinct from the pipeline component name;
- status and lifecycle timestamps;
- output path;
- experiment number/name when provided;
- config/request identity hash;
- launcher provenance and repository revision when available;
- parent/resume lineage;
- environment summary using the existing allowlisted capture.

The constant strings `resident-quantization` and `resident-factorization-slice` remain stage/component names, not globally unique run IDs. Resumed processes preserve the run ID from the manifest.

If adopting the full `RunManifest` is too large for the first resident patch, a versioned `ResearchRunManifest` adapter may be used temporarily, but catalog readers must normalize both schemas and the migration must be scheduled.

## 9. Run catalog and latest pointer

### 9.1 Catalog location and authority

Each configured catalog root contains:

```text
<run_root>/.nanoquant-runs/index.jsonl
<run_root>/.nanoquant-runs/latest.json
<run_root>/.nanoquant-runs/catalog.lock
```

The catalog record contains only discovery metadata copied from a manifest:

```json
{
  "schema_version": 1,
  "recorded_at": "2026-07-14T16:12:33.482Z",
  "run_id": "run_20260714T161233482000_ab12cd34",
  "path": "../evidence/m4/gemma-v24-canary",
  "created_at": "2026-07-14T16:10:00Z",
  "updated_at": "2026-07-14T16:12:33Z",
  "status": "running",
  "experiment_number": 24,
  "name": "gemma-v24-canary",
  "config_hash": "sha256:...",
  "revision": "..."
}
```

No raw command line or environment dump is stored. Paths are root-relative when possible and normalized before resolution. A catalog record is a snapshot; lifecycle listing verifies it against the manifest. The manifest wins on conflict.

### 9.2 Concurrency and crash recovery

Catalog append/compaction and latest replacement occur under a root-scoped catalog lock. Do not reuse the per-run lease: different runs legitimately update one root concurrently. The implementation must use a tested cross-process lock that works on Windows and POSIX, or a create-exclusive lease with owner metadata, stale-owner validation, timeout, and explicit recovery.

Within the lock:

1. confirm the manifest exists and its run ID matches;
2. append a complete JSONL snapshot and flush it;
3. write `latest.json` to a unique temporary file, flush it, and atomically replace the old pointer;
4. release the catalog lock.

If a process crashes, discovery scans the valid index prefix and verifies manifest paths. `nanoquant runs rebuild-index` scans configured roots and explicitly registered external paths, writes a compact index atomically, and regenerates `latest.json`.

The index may contain multiple lifecycle snapshots for a run. Readers fold by `run_id`, selecting the record with the greatest manifest `updated_at`, then verify the current manifest. Compaction removes superseded snapshots under the catalog lock.

### 9.3 `latest` semantics

Unqualified `latest` means the run with the greatest `(created_at, run_id)` among discoverable manifests, regardless of status. This is stable and does not change merely because an old run resumes.

Status-specific selection is explicit:

```text
latest --status running
latest --status completed
```

If `latest.json` is missing, corrupt, points outside permitted registered roots, or points to a missing/mismatched manifest, the resolver falls back to the folded index and then to a manifest scan. It emits a warning and optionally repairs the pointer under the catalog lock.

Use an atomic JSON pointer on all platforms. Do not create a symlink variant; one representation makes behavior and tests consistent.

### 9.4 External evidence directories

Caller-selected evidence paths may live outside `run_root`. A launch explicitly registers its output path with the selected catalog root. The catalog stores a relative path when possible and an absolute normalized path otherwise. CLI resolution rejects path traversal from hand-edited relative records and verifies that the target manifest's run ID matches.

Catalog rebuild cannot discover arbitrary drives. Users may configure additional scan roots or pass `--include-root PATH`. Registered absolute paths already present in a valid index are revalidated during rebuild.

## 10. CLI design

The CLI uses one global catalog-root option and consistent run selectors:

```bash
nanoquant --run-root runs runs list --limit 10
nanoquant --run-root runs runs list --status running --experiment 19
nanoquant --run-root runs runs show latest
nanoquant --run-root runs runs path latest --kind events
nanoquant --run-root runs runs path latest --kind journal
nanoquant --run-root runs logs latest
nanoquant --run-root runs logs latest --follow
nanoquant --run-root runs logs run_20260714T161233482000_ab12cd34 --level warning
nanoquant --run-root runs logs exp:19 --follow
nanoquant --run-root runs runs rebuild-index --include-root evidence/m4
```

Selectors:

- exact run ID;
- `latest`;
- `exp:N`, resolved to the newest created run for experiment N;
- a direct path only when `--allow-path` is supplied.

`runs list` prints a stable table by default and supports `--json`. `runs show` prints normalized manifest/catalog metadata and reports stale or missing files. `runs path` prints one absolute path and no decoration, suitable for shell composition.

`logs` behavior:

- without `--follow`, render existing `run.log` or stream-render `events.jsonl` if the derivative is absent;
- with `--follow`, read existing events, wait for appended complete JSONL lines, and render them without requiring the writer to maintain `run.log`;
- `--level` applies a display filter only and cannot add events absent from JSONL;
- `--json` outputs canonical event objects;
- `--rebuild` atomically regenerates `run.log` before display;
- Ctrl+C exits cleanly without changing run status.

Follow reads only complete newline-terminated records. It tolerates temporary EOF and detects file replacement/truncation. Poll intervals are configurable and default to one second for logs; this is filesystem monitoring, not CUDA job status polling.

Exit codes:

- `0`: success;
- `2`: invalid command, selector, or ambiguity;
- `3`: no matching run;
- `4`: corrupt manifest/index/event prefix;
- `5`: permission or I/O failure.

If `exp:N` has multiple runs with identical creation keys or inconsistent manifests, the command reports candidates and returns ambiguity code 2 rather than guessing.

## 11. Configuration and compatibility

### 11.1 Schema

Add these fields to `ObservabilityConfig`:

```python
console_level: Severity = Severity.INFO
event_level: Severity = Severity.INFO
file_level: Severity = Severity.INFO
write_text_log: bool = True
event_flush_interval: int = 1
text_flush_interval: int = 1
```

Keep existing resource, ADMM, reconstruction, block-loss, and trace controls.

This is an additive schema-1 change because strict decoding fills absent fields from defaults and old recipes remain valid. It is still a schema-affecting change: update `schema_reference`, configuration docs, canonical config snapshots, validation, and tests. Resolved config hashes will change when the dataclass's canonical serialization gains defaulted fields; cache/commit identities that intentionally exclude observability must continue to exclude them. Any identity that currently hashes full `RunConfig` must be reviewed and documented before merge.

If preserving existing full-config hashes is required for compatibility, omit default-valued presentation fields from that specific identity projection rather than pretending the schema did not change.

### 11.2 Backward compatibility

- Old runs with a manifest but no index are found by scanning and may be indexed lazily.
- Old runs with `events.jsonl` but no `run.log` are rendered on demand.
- Old resident evidence with no manifest can be opened by direct path and shown as `unmanaged`; `runs import PATH` creates catalog metadata only after validating the directory. It does not rewrite numerical evidence.
- Existing event schema version 1 remains readable.
- The compatibility `JsonlEventSink` preserves its public `emit` and `span` surface during migration.
- `state/journal.jsonl` format and resume behavior do not change.

## 12. Diagnostic instrumentation policy

### 12.1 Boundary rule

Pure domain math remains side-effect-free. `domain/factorization.py`, `domain/scale_fit.py`, and planning policies return typed results, trace samples, or profiler counters. Application stages and orchestration code emit events from those results.

### 12.2 Default information events

The normal info stream should be enough to monitor an hours-long run:

- run created/resumed/started/completed/failed/interrupted;
- calibration and plan identity selected or reused;
- block start/completion with block index, committed sequence, loss, and elapsed time;
- layer attempt completion and retry decision with rank, error metrics, thresholds, accepted attempt, budget delta, and reason;
- scale-fit acceptance/rollback with before/after metrics;
- tuning epoch summary at existing commit/cooldown boundaries, not every microbatch;
- artifact commit references and resume discoveries;
- sampled resource summaries at the configured interval.

Policy decisions include inputs, thresholds, result, and rationale as required by the observability design.

### 12.3 Debug and trace events

- `record_admm_steps=false` emits one aggregate factorization diagnostic per attempt.
- `record_admm_steps=true` permits sampled ADMM trace points at debug. The sample cadence is explicit and bounded; first, last, convergence checks, and anomalous residual changes are preferred over every iteration.
- Trace-level records are reserved for narrowly selected blocks/layers and must not be enabled globally by default.
- Retry budget arithmetic is emitted by `application.retry_loop`, where attempts, thresholds, and budget are already available.
- Scale-fit internals are emitted by `ScaleFitStage` from typed fit diagnostics.
- Memory events reuse the profiling/resource sampler. Report host RSS/private bytes and CUDA allocated/reserved/peak/device-used bytes. Expensive allocator snapshots are trace-only and versioned.

The event stream never duplicates raw phase samples already retained by the profiler. It may reference a profile phase or summarize an anomaly.

### 12.4 Performance budgets

On the CPU tiny integration pipeline, default observability must add no more than both:

- 2% wall-time at the median of repeated runs after warmup; and
- 10 ms absolute wall time.

For a representative resident factorization attempt, default info logging must add less than 1% wall time and must not add a CUDA synchronization. Debug ADMM sampling gets a separate measured budget and remains opt-in.

Benchmarks compare:

- router with canonical JSONL only;
- router with console disabled and text enabled;
- current baseline sink;
- debug ADMM sampling enabled/disabled.

Results are recorded in [16-behavior-preserving-optimizations.md](16-behavior-preserving-optimizations.md) if a candidate is rejected or materially changed for performance.

## 13. Lifecycle and crash behavior

### 13.1 New run

1. Resolve and validate configuration.
2. Create the output directory and initial manifest atomically.
3. Register the manifest snapshot and update `latest.json` under the catalog lock.
4. Acquire the per-run lease.
5. Open observability destinations and reconcile `run.log` from existing events.
6. Transition the manifest to running and emit `run.started`.
7. Execute the application.

### 13.2 Resume

1. Resolve the selected output and validate manifest/run ID.
2. Acquire the per-run lease before opening append destinations.
3. Validate the canonical event prefix and continue its sequence.
4. Reconcile the text derivative.
5. Discover the progress journal using existing artifact validation.
6. Transition interrupted to running where allowed and emit `run.resumed` with the committed journal sequence.

### 13.3 Completion or failure

1. Emit the terminal event while the router is healthy.
2. Flush and close all event destinations.
3. Atomically transition the manifest to its terminal status.
4. Append a catalog lifecycle snapshot.
5. Render reports from manifest, canonical events, and result artifacts.
6. Release the per-run lease in `finally`.

If report or optional text rendering fails, the manifest still records the numerical run outcome and a catalog warning is shown later. If canonical event writing fails before the terminal event, the manifest records a structured observability failure when possible.

## 14. Implementation phases

### Phase 1: Foundations and compatibility

- Add `Severity`, threshold validation, JSON field validation, and redaction utilities.
- Implement `EventRouter` and destination failure isolation.
- Keep `JsonlEventSink` as a compatibility wrapper.
- Make existing `console_level` and `event_level` effective.
- Add unit tests for routing, ordering, thresholds, close behavior, and failures.

### Phase 2: Human event view

- Implement deterministic `TextEventDestination`.
- Add `file_level`, `write_text_log`, and config validation.
- Implement sidecar reconciliation and atomic rebuild.
- Wire through `open_run_observability` in the foundation run.

### Phase 3: Catalog and CLI

- Implement `RunCatalog`, root lock, atomic latest pointer, scan fallback, and rebuild.
- Add `runs list/show/path/rebuild-index` and `logs` commands.
- Test stale pointers, corrupt index tails, external paths, concurrency, and follow behavior on Windows.

### Phase 4: Resident adoption

- Add stable manifests and leases to resident quantization and factorization-slice outputs.
- Replace direct sink construction with `open_run_observability`.
- Register caller-selected evidence directories with the configured catalog.
- Adopt the same helper in the tiny pipeline where useful for integration coverage.

### Phase 5: Diagnostic enrichment

- Add retry, scale-fit, tuning-boundary, resource, and commit events at application/orchestration boundaries.
- Add bounded ADMM trace emission behind `record_admm_steps`.
- Compare event diagnostics with existing profile and report outputs to avoid duplication.
- Measure overhead and document retained and rejected candidates.

Each phase is a major feature and is committed independently on the current branch after its tests pass.

## 15. Test and acceptance matrix

### Event pipeline

- sequences and timestamps are created once and shared by all destinations;
- all six severities filter correctly;
- invalid thresholds and severities fail configuration validation;
- derived destinations cannot be more verbose than canonical JSONL;
- field ordering and escaping are deterministic;
- oversized/invalid/non-finite fields are rejected or safely summarized;
- optional destination failures are quarantined and canonical failure is fatal;
- close is idempotent and all handles are closed in success, interruption, and failure paths;
- reopening continues from the last valid sequence.

### Text derivative

- one event always produces one physical line;
- absent, behind, ahead, corrupt, and old-version text views reconcile correctly;
- rebuild uses a unique temporary and atomic replacement;
- a crash between JSONL and text writes is recovered without duplication;
- Unicode and multiline values remain tail-safe.

### Catalog and latest

- two processes can register different runs without losing either record;
- lifecycle snapshots fold to the current manifest status;
- missing/corrupt index and pointer fall back to manifest scans;
- stale pointer repair is deterministic;
- latest selection follows `(created_at, run_id)` and status filters;
- external paths are normalized, verified, and traversal-safe;
- rebuild is idempotent;
- secrets and raw argv do not appear in catalog records.

### CLI

- exact ID, latest, experiment, status, and missing-run selectors have documented results and exit codes;
- table and JSON output remain stable;
- `runs path` prints exactly one absolute path;
- follow handles partial lines, replacement, truncation, Ctrl+C, and a completed run;
- old JSONL-only and manifest-only runs remain useful;
- event logs and progress journals are never confused.

### Integration and parity safety

- foundation runs produce manifest, events, text log, catalog entry, latest pointer, and report;
- resident runs produce the same discovery envelope in caller-selected evidence paths;
- resume preserves run identity and event sequence and discovers the same progress boundary;
- optional logging failures produce byte-identical committed numerical artifacts for a deterministic fixture;
- logging does not change config/request identities that intentionally exclude presentation settings;
- default logging adds no CUDA synchronization and satisfies the performance budgets.

## 16. Rejected alternatives

### Platform-dependent `LATEST` symlink

Rejected because Windows symlink privileges and behavior differ across environments. One atomically replaced JSON pointer is simpler to test and support.

### Global ledger as lifecycle authority

Rejected because append-only status rows become stale and create a second database requiring transaction semantics. Manifests remain authoritative; the index is rebuildable.

### Renderers attached to `JsonlEventSink`

Rejected because destination thresholds cannot be composed cleanly and optional observer exceptions currently propagate. Routing belongs above all destinations.

### Debug `run.log` with info-only `events.jsonl`

Rejected because the derived human view would contain evidence absent from the canonical stream. Users who request debug text must accept debug canonical storage.

### Multiline pretty-printed stored logs

Rejected because they break one-event-per-line tailing, filtering, and sequence correlation. Expanded rendering belongs in an interactive CLI view.

### Direct event emission inside pure domain math

Rejected because it introduces side effects and infrastructure dependencies into deterministic mathematical functions. Domain results expose diagnostics; application boundaries emit them.

### Per-iteration ADMM events by default

Rejected because event construction, serialization, and flushing are high-cardinality overhead and duplicate profiler/trace capabilities. Bounded sampling remains opt-in through `record_admm_steps`.

## 17. Definition of done

The logging refactor is complete when a researcher can start or resume either a managed or resident run, resolve it through `latest` or a stable selector, follow a concise log without opening JSON manually, rebuild every derivative/index from authoritative files, and obtain actionable decision diagnostics without changing numerical artifacts, resume boundaries, or the default performance envelope.
