Here is a design document addressing these developer experience (DX) and observability shortcomings. It follows the established Architecture Decision Record (ADR) and design specification format used in the project.

---

# Design Doc: Ergonomic Logging and Run Discoverability

**Status:** Proposed  
**Audience:** Maintainers, Tooling Engineers, Algorithm Researchers  

## 1. Problem Statement

The rewrite successfully transitioned NanoQuant from ad-hoc `print()` statements to a rigorous, structured event system (`events.jsonl`) and content-addressed artifact store. While this ensures machine-readability, auditability, and deterministic resumes, it has severely degraded the human developer experience:

1. **Buried Information:** Logs and journals are hidden inside deeply nested, UUID-suffixed directories (e.g., `runs/run_20260714T090123_a1b2c3/events.jsonl` or `.nanoquant/artifacts/sha256/ab/abcd.../`). Finding the log for the *current* or *latest* run requires manual directory sorting and traversal.
2. **Hostile Formats:** JSONL is excellent for reporting engines but hostile to developers reading via `tail` or `cat`.
3. **Lack of Context:** The current `ConsoleRenderer` is too terse. Crucial algorithmic context (e.g., why ADMM is plateauing, tensor shapes causing OOMs, or specific scale-fit regressions) is either missing entirely or stripped out to keep the console clean.
4. **No Global Ledger:** There is no single place to see a history of what experiments have been run on a machine, making it hard to track long-running sweeps.

## 2. Goals

*   Make finding the logs for the most recent run instant and deterministic.
*   Provide a human-readable, traditional text log containing deep diagnostic information without breaking the structured JSONL contract.
*   Provide CLI tooling to list, search, and tail runs without touching the filesystem directly.
*   Add `debug`/`trace` level events to the core algorithm without cluttering the standard `info` console output.

## 3. Proposed Solutions

### 3.1. The `LATEST` Pointer and Global Ledger
We will stop forcing developers to memorize or `ls -tr` UUID folders. 

*   **`runs/LATEST` Symlink:** Whenever a new run is created, the system will update a `LATEST` symlink (or a `.LATEST.json` pointer file on Windows to avoid admin privilege requirements) in the `run_root`. 
*   **`runs/nanoquant_history.log`:** A flat, global, append-only CSV/JSONL ledger at the root of the output directory. It records: `timestamp, run_id, experiment_number, status, command_line`. This allows developers to instantly see what happened recently across *all* runs.

### 3.2. Dedicated Human-Readable Text Logs (`run.log`)
`events.jsonl` will remain the canonical machine-readable audit trail. However, the `EventSink` will gain a new attached renderer: `TextFileRenderer`.

*   Every run directory will contain a `run.log` file.
*   This file will format the structured events into a traditional `[TIMESTAMP] [LEVEL] [STAGE] message` format.
*   Complex fields (like shape tuples, nested dictionaries, or multiline tracebacks) will be pretty-printed across multiple lines for readability.

### 3.3. CLI Tooling for Observability
We will add new CLI commands to `src/nanoquant/cli/main.py` that abstract away the filesystem structure:

```bash
# List all runs in the run_root chronologically
nanoquant runs list --limit 10

# Tail the human-readable log of the most recent run
nanoquant logs LATEST --follow

# Tail the log of a specific run or experiment number
nanoquant logs run_20260714T090123_a1b2c3
nanoquant logs exp:19

# Print the exact path to the latest journal for quick inspection
nanoquant inspect-run LATEST --path-only
```

### 3.4. Expanded Verbosity Levels (`DEBUG` and `TRACE`)
Currently, `ObservabilityConfig` supports `console_level` and `event_level`. We will instrument the domain code with high-cardinality debugging information that defaults to `DEBUG`.

*   **ADMM Traces:** Emit per-iteration primal/dual residuals to `DEBUG` so they appear in `run.log` but not the console.
*   **Retry Mechanics:** Emit exact marginal utility scores and budget math to `DEBUG` when deciding rank allocations.
*   **Memory State:** Emit allocator high-water marks and fragmentation states to `DEBUG` before and after large block allocations.
*   The `ConsoleRenderer` will default to `INFO`, keeping the terminal clean, while `TextFileRenderer` (writing to `run.log`) will default to `DEBUG`.

## 4. Architecture & Implementation

### 4.1. Configuration Changes
No schema migrations are strictly required, but the defaults in `ObservabilityConfig` will conceptually map as:
*   `console_level`: `"info"`
*   `file_level`: `"debug"` (New field, or derived implicitly for `run.log`)

### 4.2. Infrastructure Updates (`src/nanoquant/infrastructure/events.py`)

We will update the `JsonlEventSink` architecture to support multiple independent observers (Renderers) with their own log-level filters.

```python
class TextFileRenderer:
    def __init__(self, path: Path, level: str = "debug"):
        self.handle = path.open("a", encoding="utf-8")
        self.level_int = _level_to_int(level)

    def __call__(self, event: Event) -> None:
        if _level_to_int(event.severity) < self.level_int:
            return
        
        # Format: [2026-07-14 09:12:33] [DEBUG] [factorize-attempt] Retrying rank 32 -> 64
        base = f"[{event.timestamp[:19].replace('T', ' ')}] [{event.severity.upper():5}] [{event.stage}] {event.name}"
        
        # Pretty print extra fields
        if event.fields:
            fields_str = " | " + " ".join(f"{k}={v}" for k, v in event.fields.items())
            base += fields_str
            
        self.handle.write(base + "\n")
        self.handle.flush()
```

### 4.3. Run Lifecycle Updates (`src/nanoquant/infrastructure/runs.py`)
Modify `initial_manifest` and `RunDirectory` creation to:
1. Append to `nanoquant_history.log`.
2. Update the `LATEST` pointer.
3. Instantiate the `TextFileRenderer` alongside the `JsonlEventSink` and `ConsoleRenderer`.

## 5. Backward Compatibility & Safety

*   **S0 (Behavior-Preserving):** This is purely an infrastructure and CLI change. It does not alter the mathematical domain, artifact hashes, or the `events.jsonl` audit trail.
*   **Performance:** String formatting for `run.log` will only occur for events that pass the verbosity filter. `DEBUG` events will be completely skipped if `file_level` is set to `INFO`, preserving the strict profiling budgets defined in `15-performance-profiling.md`.

## 6. Execution Plan

1.  **Phase 1:** Implement `nanoquant runs list` and `nanoquant logs LATEST` CLI commands using basic filesystem traversal. Update `RunDirectory` to maintain the `LATEST` pointer and global ledger.
2.  **Phase 2:** Implement `TextFileRenderer` and wire it into the `JsonlEventSink` composite observer. Ensure `run.log` is generated for all new runs.
3.  **Phase 3:** Audit domain code (`factorization.py`, `retry_loop.py`, `scale_fit.py`) and inject rich `context.events.emit(..., severity="debug")` calls to expose internal algorithmic decisions.