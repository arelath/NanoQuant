# VRAM Diagnostics

**Status:** Proposed

**Audience:** Maintainers, tooling engineers, and algorithm researchers

**Related:** [LoggingRefactorV3.md](LoggingRefactorV3.md), [15-performance-profiling.md](15-performance-profiling.md), [16-behavior-preserving-optimizations.md](16-behavior-preserving-optimizations.md), [07-observability-and-reporting.md](07-observability-and-reporting.md)

## 1. Summary

NanoQuant already measures CUDA memory in at least eight places — the profiler's opt-in phase counters, the resident batch-size probes, block peak capture, tuning's reserved-bytes cooldown, resource planning, OOM fallbacks, several standalone tools, and hand-rolled assertions in GPU tests — and each one invents its own field names, its own peak-reset policy, and its own output location. There is no way to watch VRAM on a live run, no standard way to gate a VRAM regression in a test, and nothing captured at the moment an OOM actually happens.

This design defines **one meter vocabulary, one sampling function, and five instruments** built on it, each answering a different question and each feeding an existing evidence channel:

| Instrument | Question | Consumer | Gate | Default |
|---|---|---|---|---|
| Periodic sampler | What is VRAM doing *right now* on an hours-long run? | `resource.sample` events → `logs --follow` | `record_resource_interval_seconds` | On (5 s) |
| Boundary checkpoints | Which block/epoch/probe moved the peak, and did it match the plan? | Enriched existing info events | Always with default logging | On |
| Phase memory counters | Which phase grows or holds memory, aggregated over a run? | `profile.json` / `profile.md` (existing) | `profiling.memory_counters` | Off |
| Test budget fixture | Did this change regress peak VRAM for a known workload? | pytest assertion + JSONL report | Per-test opt-in | Off |
| Allocator history + OOM forensics | *Which allocations* are alive, and what did the allocator look like when we died? | Snapshot artifacts + `resource.oom_snapshot` event | `capture_cuda_trace` / env override | Off |

Everything routes through the [LoggingRefactorV3](LoggingRefactorV3.md) event pipeline where events are involved, and through the [Docs/15](15-performance-profiling.md) profiler where aggregates are involved. This design adds **zero** new configuration fields: it makes two existing dead knobs effective — `ObservabilityConfig.record_resource_interval_seconds` (defined, read by nothing) and `ObservabilityConfig.capture_cuda_trace` (defined, read by nothing, documented nowhere else) — and uses environment overrides for the rest, following the `NANOQUANT_PROFILE` precedent.

## 2. Goals

- Watch device memory live on a managed or resident run with `nanoquant logs latest --follow`, without attaching a debugger or adding prints.
- Attribute VRAM movement to blocks, layers, tuning epochs, and probes from the event stream alone.
- Distinguish the four failure shapes — live-tensor leak, allocator-pool growth/fragmentation, external device pressure, and genuine working-set overflow — from recorded numbers, not guesswork (§8).
- Make VRAM regressions in GPU tests a declared budget with a reusable fixture, replacing the hand-rolled patterns in `test_device_batches.py` and `test_resident_batching.py`.
- Capture actionable forensics at the moment of a CUDA OOM, before fallback or unwind destroys the evidence.
- Compare planned device budgets (resource planning) against measured peaks.
- Add no CUDA stream synchronization and no new config schema fields at default settings.

## 3. Non-goals

- Per-allocation or per-kernel event emission. High-cardinality allocator data goes into snapshot artifacts, never into `events.jsonl` (V3 field contract).
- An NVML/pynvml or `nvidia-smi` dependency. `torch.cuda.mem_get_info` already reports driver-level free/total without a subprocess or a new package, on Windows and POSIX.
- Changing batching, planning, or fallback *policy*. This design measures; existing policies keep their inputs (they may additionally emit what they saw).
- Cross-GPU or cross-process accounting beyond what `device_used_bytes` implies about other consumers of the same device.
- Replacing or restructuring the profiler. Phase memory counters already work ([profiling.py](../src/nanoquant/infrastructure/profiling.py), `_MemoryMetricAggregate`); this design positions them and shares their vocabulary.

## 4. The meters

All instruments read the same meters, defined once. The CUDA side is exactly what `_runtime_memory_sample()` in [profiling.py](../src/nanoquant/infrastructure/profiling.py) already captures; that function moves to a new shared home, `infrastructure/device_memory.py`, and the profiler delegates to it.

| Field | Source | What it means | Primary question it answers |
|---|---|---|---|
| `cuda.allocated_bytes` | `memory_stats()` current | Bytes in live tensors | Leak detection: this growing while work is steady-state is a leak |
| `cuda.reserved_bytes` | `memory_stats()` current | Bytes held by the caching allocator (pool) | What other CUDA work *cannot* use; the number OOM cares about |
| `cuda.peak_allocated_bytes` / `cuda.peak_reserved_bytes` | `memory_stats()` peak | High-water marks since last reset | Window maxima for blocks/probes/tests |
| `cuda.device_free_bytes` / `cuda.device_used_bytes` / `cuda.device_total_bytes` | `mem_get_info()` | Driver-level truth, including the CUDA context and *other processes* | External pressure; real headroom |
| `cuda.allocation_count` / `cuda.free_count` | `memory_stats()` | Allocator churn | Fragmentation risk correlates with churn |
| `host.working_set_bytes` / `host.private_bytes` (+ peaks) | [resource_usage.py](../src/nanoquant/infrastructure/resource_usage.py) | Process host memory | Offload/pinned-buffer growth that shadows VRAM changes |

Facts the design relies on:

- Reading `memory_stats()` and `mem_get_info()` does **not** synchronize a CUDA stream. Sampling is cheap enough for a periodic default.
- The `getattr(torch.cuda, "_initialized", False)` guard (already in the sampler) means sampling never *forces* CUDA initialization: on CPU-only environments and in unit tests the sample simply contains only `host.*` fields.
- The derived quantity **`reserved − allocated`** is the caching/fragmentation gap; **`device_used − reserved`** is context-plus-external pressure. Consumers compute these; they are not stored redundantly.
- Field names are identical in events, `profile.json`, OOM reports, and test reports, so evidence cross-correlates exactly. All values are integers; units live in the name per the observability design.

### 4.1 Peak-counter ownership

`torch.cuda.reset_peak_memory_stats` is process-global, and today three different actors reset it: run setup, per-block capture ([quantization_stages.py:195](../src/nanoquant/application/quantization_stages.py:195)), and the resident batch probes — which already have to fold their observed peak back into the run-level number (`max(probe_peak, ...)` at [resident_quantization.py:707](../src/nanoquant/resident_quantization.py:707)). That folding rule becomes the documented protocol instead of a local fix:

1. Peak windows are **hierarchical**: run ▸ block ▸ probe/test. Only the orchestrator that owns a window may reset peaks at its start.
2. An inner window that resets peaks **must fold** its observed peak into every enclosing window's tracking before the enclosing window reads its own peak.
3. **Read-only instruments never reset**: the periodic sampler, the profiler's phase counters, and OOM forensics only read.
4. `empty_cache()` changes `reserved_bytes`, never `allocated_bytes`, and never peaks. Code that calls it (tuning cooldown, [tuning.py:234](../src/nanoquant/application/tuning.py:234); post-phase cleanups) is expected to show as a reserved-bytes drop in the next sample — that visible sawtooth is a feature, not noise.

A small `PeakWindow` helper in `device_memory.py` implements reset-and-fold so future windows cannot get rule 2 wrong.

## 5. Instruments

### 5.1 Periodic resource sampler (live runs)

A daemon thread owned by the V3 run session emits one `resource.sample` info event per interval:

```text
2026-07-14T18:40:12.031Z 0000412 INFO    resource resource.sample cuda.allocated_bytes=18734252032 cuda.device_free_bytes=2147483648 cuda.device_total_bytes=25757220864 cuda.device_used_bytes=23609737216 cuda.reserved_bytes=21474836480 host.working_set_bytes=41875931136
```

On Windows, the standard sample also includes `wddm.dedicated_bytes`, `wddm.shared_bytes`, and their process-lifetime
peaks. These counters are not interchangeable with PyTorch allocator bytes: pinned CPU allocations remain
GPU-addressable and are charged as WDDM shared GPU memory even after their tensors die if PyTorch retains the blocks
in its pinned-host cache. Recording both vocabularies prevents a bounded CUDA allocator from hiding system-memory
paging pressure.

- **Cadence:** the existing `ObservabilityConfig.record_resource_interval_seconds` (default 5.0) finally becomes effective. Values ≤ 0 disable the sampler; validation (`OBS004`) rejects non-finite values and warns below 1.0 s to keep volume bounded. At the default, a 40-hour resident run adds ~29k events ≈ 10 MB of JSONL — negligible next to the evidence already retained.
- **Threading:** the V3 `EventRouter` is thread-safe under its per-process lock, so a sampler thread may emit directly. The sampler must never touch the `Profiler`, which is thread-confined by design.
- **Failure policy:** mirrors V3's optional-destination quarantine. The first sampling exception stops the thread and emits one `observability.sampler_disabled` warning with `error_type`; the run is untouched. Sampler emit failures are already covered by router policy.
- **Resident and managed parity:** because the sampler lives in `open_run_session`, resident quantization, the factorization slice, and managed runs all get it with no per-entry-point code.
- Peaks are included in the sample (`cuda.peak_*`) but interpreted against the ownership protocol: they reflect the current innermost reset window, and the block-boundary events (§5.2) are the authoritative per-window numbers.

This is the instrument that makes `nanoquant logs latest --follow` a live VRAM monitor, and `nanoquant logs RUN --json | <analysis>` a post-hoc timeline without any new file format.

### 5.2 Boundary checkpoint events (attribution)

The V3 Phase-4 diagnostic enrichment gains a standard rule: **every existing lifecycle info event at a memory-relevant boundary carries the standard meters**, using the shared sampler:

- `block.completed` — plus `cuda.window_peak_allocated_bytes` / `cuda.window_peak_reserved_bytes` for the block's peak window, and, when resource planning supplied a device budget for the block, `planned_device_bytes` and `budget_utilization` (measured window peak ÷ plan). Planning already reads `mem_get_info` ([resource_planning.py:53](../src/nanoquant/infrastructure/resource_planning.py:53)); this closes the loop between what it promised and what happened.
- tuning epoch summaries at existing commit/cooldown boundaries — plus meters, so the reserved-bytes sawtooth from cooldown `empty_cache` is attributable.
- batch probe results — the probes measure peaks already; they now emit `probe.completed` debug events with the measured peak, chosen batch size, and folded run peak, instead of keeping those numbers internal.
- `run.started` / `run.resumed` / terminal events — plus meters, giving every run a baseline and a final state (the CUDA context cost is `device_used − reserved` at `run.started`).

No new event cadence is created — these are fields on events the logging design already emits. Domain code stays pure: all sampling happens in application/orchestration code that already owns these boundaries (V3 §11.1).

### 5.3 Phase memory counters (aggregate attribution) — existing, unchanged

`profiling.memory_counters: true` already samples the meters at phase enter/exit and aggregates `first/last/min/max/net_change/positive_delta` per phase into `profile.json` and the "Run memory counters" table in `profile.md`, with `PERF004` degradation on sampler failure. This design changes only its plumbing (delegating to `device_memory.py`) so the vocabulary is shared.

Division of labor, to prevent duplicate evidence (V3 §11.3): the **sampler** is time-based and answers "when"; **phase counters** are structure-based and answer "which code path, on aggregate"; **checkpoint events** answer "which block/epoch, exactly". The event stream never carries the profiler's raw per-phase samples.

### 5.4 Allocator history trace (deep diagnosis, opt-in)

When enabled, the run session calls `torch.cuda.memory._record_memory_history(max_entries=N)` (bounded; default 100k entries) at open, and dumps `_dump_snapshot()` pickles as `state/vram-history-<n>.pickle`:

- automatically on CUDA OOM (§5.5) and at terminal transition;
- on demand from a resident tool flag.

Snapshots open in the PyTorch memory visualizer (`pytorch.org/memory_viz`) and show every live allocation with its Python stack — the tool that turns "allocated grows 40 MB per block" into a file and line number.

- **Gate:** `ObservabilityConfig.capture_cuda_trace: true` — this design assigns the currently dead knob its meaning — or the `NANOQUANT_VRAM_HISTORY=1` environment override for resident tools and ad-hoc test runs, mirroring `NANOQUANT_PROFILE`.
- **Never default:** history recording adds allocator-path overhead and host memory, and snapshot files are large. It is exempt from the performance budgets and excluded from parity gates.
- **Private-API containment:** `_record_memory_history`/`_dump_snapshot` are underscore APIs. `device_memory.py` wraps them behind a version-checked adapter; if the running torch doesn't provide them, the session emits one `observability.vram_history_unavailable` warning and continues — a diagnostics gap must never fail a run.
- **Retention:** snapshots are derivative diagnostics under [Docs/14](14-artifact-retention-and-disk-usage.md) rules — safe to delete, never referenced by resume.

### 5.5 OOM forensics

Today a CUDA OOM produces a Python traceback and, at best, a fallback event (`runtime.oom_unrecoverable`/`RES002`, `calibration.oom_*`/`CAL005-6`) with no memory numbers — the allocator state that explains the failure is gone by the time anyone looks. New behavior at the existing fallback/unwind boundaries, before any `empty_cache` or retry:

1. Sample the meters and emit `resource.oom_snapshot` (severity `error`) with the standard fields plus `requested_bytes` parsed from the OOM message when present, and the active stage/block/layer context the boundary already has.
2. Write `torch.cuda.memory_summary()` to `state/oom-report-<n>.txt` and reference it from the event by path — the multi-kilobyte text block goes into an artifact, not event fields, per the V3 field contract.
3. If history tracing is active, dump a snapshot pickle and reference it too.

All three steps are best-effort inside a `try` that swallows its own failures (the profiler's established pattern): forensics must never mask the original OOM or change fallback behavior. The existing fallback events remain unchanged and now have a sibling event with the evidence.

## 6. VRAM during testing

### 6.1 The `vram_budget` fixture

GPU tests currently hand-roll the same sequence — `empty_cache`, record baseline reserved, `reset_peak_memory_stats`, run, assert peak increment ([test_device_batches.py:31](../tests/unit/test_device_batches.py:31), `test_resident_batching.py`). That becomes one fixture in `tests/support`:

```python
def test_streamed_block_forward_fits(vram_budget):
    with vram_budget(peak_increment_bytes=512 * 2**20) as window:
        run_streamed_forward(...)
    assert window.peak_increment_bytes <= window.budget  # raised by the fixture with a diagnostic
```

Behavior:

- baseline: `empty_cache()`, read `reserved_bytes`, open a `PeakWindow` (§4.1 — reset-and-fold, so a fixture nested inside a larger measured scope stays correct);
- on exit: compute peak increments (allocated and reserved), fail the test with a rendered meter table if the declared budget is exceeded;
- **report:** append one JSONL line per measured test — test id, budget, measured peaks, device name, torch version — to a path given by `--vram-report PATH` (or `NANOQUANT_VRAM_REPORT`). The report uses the standard field names, so the same analysis tooling reads test evidence and run evidence. Budgets live at the test site as explicit bytes: a budget is an assertion, and assertions belong in the test that owns them.
- skips cleanly when CUDA is unavailable, like the existing GPU tests.

### 6.2 Diagnosing a failing GPU test

The run instruments work under pytest because they are gated by environment, not by run plumbing: `NANOQUANT_PROFILE=macro` with `memory_counters` gives per-phase aggregates for the code under test, and `NANOQUANT_VRAM_HISTORY=1` makes the fixture dump an allocator snapshot on budget failure — the investigation path from "this test regressed by 300 MB" to "these allocations with these stacks are new" is two environment variables, no code changes.

### 6.3 Unit-test seams

Everything goes through `device_memory.sample()` (injectable, as `Profiler` already accepts `memory_sampler`) so existing monkeypatch-based tests (`test_tuning.py`, `test_resident_batching.py`, `test_profiling.py`) keep working and new sampler/fixture tests need no GPU. The CPU tiny pipeline emits host-only samples via the `_initialized` guard — CI without CUDA exercises the full sampler path minus the `cuda.*` fields.

## 7. Fit into the logging system

Mapping onto [LoggingRefactorV3](LoggingRefactorV3.md) explicitly:

- **Producers sit at application/orchestration boundaries** (V3 §11.1). `device_memory.py` is infrastructure with no event knowledge; sessions, stages, and fallbacks sample it and emit.
- **Severities:** `resource.sample` and enriched checkpoints are `info` (part of the "monitor an hours-long run" stream, V3 §11.2); probe details are `debug`; `resource.oom_snapshot` is `error`; self-diagnostics (`observability.sampler_disabled`, `observability.vram_history_unavailable`) are `warning`. All obey `event_level`.
- **Field contract compliance:** every field is a bounded integer, float, or short string; `memory_summary()` text and history pickles are artifacts referenced by path, never inline (V3 §6.4).
- **Failure isolation:** the sampler thread and OOM forensics follow the same doctrine as optional destinations — degrade once, warn once, never alter stage results, commits, or the progress journal.
- **Discovery and reading:** live monitoring is `nanoquant logs latest --follow`; post-hoc analysis is `logs RUN --json` piped to analysis, plus a small reader-side `nanoquant runs vram SELECTOR` that folds `resource.sample` and checkpoint events into a summary table (run peak, baseline, per-block window peaks, budget utilization, empty-cache sawtooth count). Consistent with V3's computed-views principle, it stores nothing.
- **Manifest synergy:** `PYTORCH_CUDA_ALLOC_CONF` and `CUDA_VISIBLE_DEVICES` are already in the environment-capture allowlist, so every run's manifest records the allocator configuration its VRAM numbers were measured under — required context when comparing runs across `expandable_segments` changes.

## 8. Reading the numbers: diagnosis playbook

| Symptom (from samples/checkpoints) | Meaning | Next tool |
|---|---|---|
| `allocated` grows monotonically across blocks at steady-state work | Live-tensor leak | History trace on; diff two snapshots in memory_viz |
| `allocated` flat, `reserved` grows; gap widens | Allocator pool growth / fragmentation | Check `allocation_count` churn; try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (recorded in manifest); confirm cooldown `empty_cache` drops reserved |
| `device_used − reserved` large or growing | CUDA context cost or another process on the GPU | Compare `run.started` baseline; check the host |
| OOM with `reserved ≫ allocated` | Fragmentation, not true exhaustion | `oom-report` block table; allocator config; history snapshot |
| OOM with `allocated ≈ reserved ≈ device_total` | Genuine working-set overflow | `budget_utilization` on recent blocks; batch probe events; reduce batch/enable existing fallback |
| Block `budget_utilization` ≫ 1 | Planning underestimates that block shape | Compare `planned_device_bytes` vs window peak across blocks; fix the planner's model |
| Test budget failure, `peak_reserved` up but `peak_allocated` flat | Allocation pattern change, same live memory | Usually benign; adjust budget consciously or investigate churn |

## 9. Performance budgets

- Default configuration (sampler at 5 s, checkpoint fields on): covered by the V3 budgets — ≤ 2%/10 ms on the tiny CPU pipeline, < 1% on a resident attempt, **zero added stream synchronizations**. The benchmark matrix in V3 §11.4 gains a sampler-on/off pair; results and any rejected candidates land in [Docs/16](16-behavior-preserving-optimizations.md).
- One meter sample is microseconds of host-side work; the 5 s cadence makes it unmeasurable. The interval validation floor exists so a misconfigured 0.01 s cadence cannot silently melt the budget.
- History tracing and OOM forensics are explicitly outside the budgets: opt-in, and active only when something is already wrong.

## 10. Configuration

The diagnostics implementation subsequently added one opt-in execution guard required by the 4B canary:

- `CompressionQualityExperiment.maximum_wddm_shared_gib` — operator guard used by the 4B canary without changing
  semantic compression identity. The sampler latches a transient violation and compression, distillation, or
  quality evaluation raises `VRAM001` at a safe point.

The remaining fields are unchanged from V3 §5:

- `observability.record_resource_interval_seconds` — existing; becomes effective. `≤ 0` disables; `OBS004` validates finiteness and warns under 1.0 s.
- `observability.capture_cuda_trace` — existing dead knob; defined here as the allocator-history gate.
- `profiling.memory_counters` — existing; unchanged.
- Environment overrides (no schema impact): `NANOQUANT_VRAM_HISTORY=1`, `NANOQUANT_VRAM_REPORT=PATH`.
- Resident, distillation, and quality requests receive the resolved byte ceiling as a non-semantic execution option
  from the numbered experiment workflow.

## 11. Test matrix

- **device_memory:** sample shape with and without CUDA initialized; injectable sampler; `PeakWindow` reset-and-fold correctness for nested windows (run ▸ block ▸ probe), including the existing probe-folding case; no reset performed by read-only consumers.
- **Sampler:** cadence honored with a fake clock; disable at `≤ 0`; first failure quarantines and emits `sampler_disabled` exactly once; thread stops at session close in success/failure/interrupt paths; events carry only allowed field types; deterministic fixture run with sampler fault injection produces byte-identical committed artifacts and journal (V3 acceptance pattern).
- **Checkpoints:** block/tuning/probe events carry meters and window peaks; `budget_utilization` present exactly when planning supplied a budget; no meters added to any debug-suppressed path when `event_level=info` excludes it.
- **Forensics:** simulated OOM produces `resource.oom_snapshot`, a summary artifact, and (when tracing) a snapshot dump; forensics-path exceptions never mask the original OOM; fallback decisions are unchanged with forensics on/off.
- **History adapter:** graceful `vram_history_unavailable` on a torch stub missing the private APIs; bounded `max_entries` passed through.
- **Fixture:** budget pass/fail with monkeypatched meters; JSONL report lines use standard field names; nested-in-window correctness; CUDA-absent skip.
- **CLI:** `runs vram` folds a synthetic event stream into the documented summary; works on legacy JSONL-only directories via `--path`.

## 12. Implementation phases

1. **`device_memory.py`.** Extract the shared sampler from `profiling.py`; add `PeakWindow`; delegate the profiler to it. Pure refactor plus one new helper — independently landable now.
2. **Test fixture.** `vram_budget` + report writer; migrate `test_device_batches.py` and `test_resident_batching.py` to it. Independent of the logging work.
3. **Sampler + checkpoints.** Rides [LoggingRefactorV3](LoggingRefactorV3.md) Phase 4 (diagnostic enrichment): sampler thread in `open_run_session`, `OBS004` validation, meter fields on boundary events, probe events, budget-vs-actual. Requires V3 Phases 1–2.
4. **Forensics + history.** OOM snapshot events/artifacts, history adapter, `capture_cuda_trace` activation, `runs vram` command (requires V3 Phase 3 CLI plumbing).

## 13. Rejected alternatives

- **Per-allocation events in `events.jsonl`** — unbounded cardinality, duplicates the allocator's own history mechanism, and violates the V3 field contract. Allocator detail belongs in snapshot artifacts.
- **NVML/pynvml or `nvidia-smi` polling** — a new dependency or subprocess for numbers `mem_get_info` already provides in-process; per-process attribution from NVML adds little when NanoQuant owns the device via the existing lease.
- **A separate `vram.jsonl` stream** — a second stream needs its own discovery, rotation, and correlation story; `resource.sample` events in the canonical stream get follow, filtering, and rebuild semantics for free.
- **Inlining `memory_summary()` into event fields** — multi-KB text blobs in events break the size contract and tailing; it is an artifact.
- **Default-on allocator history** — allocation-path overhead and large artifacts for evidence needed only during active investigations.
- **Sampler resetting peak counters to get per-interval peaks** — a read-only instrument mutating process-global state would corrupt every window owner below it; interval peaks are derivable from window events plus samples.
- **A new `vram_interval` config field** — `record_resource_interval_seconds` already exists with exactly this meaning; giving dead configuration its documented meaning beats growing the schema.

## 14. Definition of done

A researcher watching `nanoquant logs latest --follow` sees the VRAM sawtooth of a resident Gemma run in real time; every block-completion line states its window peak and budget utilization; a GPU test that regresses peak memory fails with a meter table and one line of JSONL evidence; an OOM leaves behind an event, a summary report, and (when tracing) a clickable allocator snapshot instead of only a traceback; and all of it shares one field vocabulary across `events.jsonl`, `profile.json`, and test reports — with no new configuration fields, no CUDA synchronization, and no change to committed artifacts or resume behavior.
