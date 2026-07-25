This is an exceptionally well-engineered, rigorously typed, and highly robust machine learning codebase. The heavy use of immutable dataclasses, explicit resource/memory planning, and extensive telemetry makes it highly resilient.

However, because of the strictness and the complexity of the domain (ADMM factorizations, custom GGUF packing, Triton kernels, precise memory tracking), several areas have accumulated boilerplate or grown into "God objects."

Here are the highest-impact code cleanup and refactoring tasks to tackle next, ordered by impact on maintainability and developer velocity.

Implementation preserves two established repository contracts: numbered Python
launchers remain thin provenance-bearing adapters while every standard definition
can also round-trip through declarative YAML/JSON, and frozen dataclasses retain
their domain invariants while strict Pydantic adapters replace the manual config
decoder.

### [x] 1. Break Down the `resident_quantization.py` God Object
The file `src/nanoquant/resident_quantization.py` is over 1,200 lines long, and the `_run_resident_quantization_impl` function is a massive procedural script doing too many things: environment setup, resuming state, memory profiling, iterative block execution, microbatch autotuning, layer tuning, metrics calculation, and artifact publishing.
*   **The Refactor:** Convert this massive function into a formal Pipeline or State Machine.
    *   Extract the block quantization loop (`for block_plan in plan.blocks:`) into a separate class (e.g., `BlockQuantizer`).
    *   Extract the layer factorization/tuning steps into a `LayerQuantizer` class.
    *   Move the resume/state-hydration logic (`_restore_committed_state`) into a dedicated `QuantizationStateManager`.
*   **Why it's high impact:** It will make the core quantization algorithm independently testable, drastically reduce the cognitive load needed to understand the execution flow, and make it easier to add new quantization strategies (like your existing rank expansion).

### [x] 2. Move to Declarative Experiments (YAML/JSON)
Currently, you have 29 separate Python scripts in the `experiments/` directory (e.g., `001-compress-gemma-3-1b-it.py` to `029-compress-and-benchmark-qwen3-8b.py`). They are almost entirely identical, relying on `config_delta` to tweak `BASE_COMPRESSION_TEMPLATE`.
*   **The Refactor:** Replace these 29 scripts with a single `run_experiment.py` entry point that takes a declarative YAML or JSON file as input.
    *   You already have a robust configuration schema. The experiment files just hardcode the deltas.
    *   Keep the Python recipes/templates, but allow the experiment definitions to be pure data.
*   **Why it's high impact:** Reduces massive code duplication. It also makes it trivial to diff two experiments, version control configurations, or generate experiments programmatically for hyperparameter sweeps.

### [x] 3. Replace Manual Codecs & Validation with Pydantic (or similar)
Your `src/nanoquant/config/schema.py`, `src/nanoquant/config/codec.py`, and `src/nanoquant/domain/models.py` contain hundreds of lines of custom type-checking, JSON deserialization (`_decode`), and `__post_init__` validation (`if val < 0: raise ValueError...`).
*   **The Refactor:** Migrate your configuration and domain models to [Pydantic](https://docs.pydantic.dev/). If you want to keep pure `dataclasses`, you can use `pydantic.dataclasses` to automatically handle JSON parsing, strict type coercion, and value boundary validation.
*   **Why it's high impact:** You can entirely delete `codec.py`, `validation.py`, and all the `__post_init__` boilerplate. It will make adding new configuration parameters frictionless and drastically reduce the surface area for bugs in configuration parsing.

### [x] 4. Consolidate Telemetry/Observability Boilerplate
Across your application layer (e.g., `distillation.py`, `resident_quantization.py`), business logic is heavily interleaved with telemetry boilerplate:
```python
with _logged_operation(events, "layer_commit", block=block_index, layer=layer_plan.layer.path...):
    with _profile_layer_phase(recorder, block_index, layer_plan.layer.path, "commit"):
        committed_layer = commit_layer(layer_result, artifacts, identity)
        ...
        events.emit("resident-quantization", "info", "layer.completed", ...)
```
*   **The Refactor:** Create a unified `TelemetryContext` or use decorators that automatically handle `PhaseRecorder` phases, `EventSink` emissions, and timing without polluting the business logic.
*   **Why it's high impact:** It separates the "what the code is doing" from "how we are measuring it," instantly making the core ML logic 30-40% shorter and significantly more readable.

### [x] 5. Introduce Parameter Objects for Long Signatures
Functions like `factorize_admm` (`src/nanoquant/domain/factorization.py`), `execute_rank_expansion`, and `_run_resident_factorization_attempts` take upwards of 10–16 parameters.
*   **The Refactor:** Group related parameters into logical Parameter Objects (Dataclasses). For instance, pass an `AdmmContext` or `FactorizationContext` instead of passing `outer_iterations`, `inner_iterations`, `regularization`, `penalty_schedule`, etc., as separate kwargs.
*   **Why it's high impact:** Prevents signature bloat, makes it easier to pass configuration down the call stack, and reduces the risk of accidentally swapping arguments of the same type.

### [x] 6. Standardize Tensor I/O and Safetensors Handling
You have multiple files (`safetensors_io.py`, `safetensors_source.py`, `tensor_store.py`) that handle loading/saving tensors. Meanwhile, other files manually invoke `safe_open` and `save_file` (e.g., `distillation_checkpoint.py`, `llamacpp.py`, `packed_artifact.py`).
*   **The Refactor:** Force *all* tensor I/O through `TensorStore` or a unified `SafetensorsManager`. Remove raw `safetensors` imports from application/domain logic.
*   **Why it's high impact:** Standardizes how memory mapping, device placement (`to(device)`), and hash verification are handled. It ensures that memory leaks (failing to close handles) or device placement bugs don't creep into individual workflow scripts.

### Suggested Attack Order:
If I were jumping into this codebase, I would tackle them in this order to maximize ROI while minimizing disruption:
1. **The Experiments Sprawl (#2)** - Quickest win, deletes a lot of files.
2. **Parameter Objects (#5)** - Low risk, makes refactoring the larger files much safer.
3. **The `resident_quantization.py` Break Down (#1)** - The most critical architectural improvement for the long-term health of the pipeline.
4. **Telemetry Consolidation (#4)** - Cleans up the newly separated classes from Step 3.

Here are several more high-impact refactoring tasks. While the first list focused heavily on architecture and configuration, this list focuses on **safety, resource management, and external integrations**.

### [x] 7. Abstract Atomic File/Workspace Transactions
**The Problem:** You have excellent utilities in `infrastructure/io_utils.py` (like `atomic_write_json`), but for larger multi-file artifacts (e.g., in `gguf_export.py`, `mmproj_export.py`, `run_registry.py`, `packed_artifact.py`), the codebase manually manages temporary directories. There are dozens of blocks doing `tempfile.mkstemp` or `tempfile.mkdtemp`, followed by manual `os.replace`, wrapped in a `try...finally` block that calls `shutil.rmtree` or `os.unlink`.
**The Refactor:** Create an `AtomicWorkspace` context manager.
```python
@contextmanager
def atomic_workspace(final_destination: Path) -> Iterator[Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-", dir=final_destination.parent))
    try:
        yield temp_dir
        os.replace(temp_dir, final_destination)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
```
**Why it’s high impact:** It eliminates hundreds of lines of fragile, repetitive I/O boilerplate. It guarantees that a crashed process *never* leaves behind corrupted partial artifacts or orphaned temporary directories, which is critical for a content-addressed storage system.

### [x] 8. Formalize GPU Memory Lifecycles (RAII for VRAM)
**The Problem:** Managing VRAM is notoriously difficult. Currently, `torch.cuda.empty_cache()`, `gc.collect()`, and explicit `del` statements are scattered imperatively throughout the code (especially in `resident_quantization.py`, `short_decode_benchmark.py`, and `quality_evaluation.py`). Miss one `del` statement in an error-handling path, and the run crashes with an OOM.
**The Refactor:** Expand `infrastructure/memory_cleanup.py` to use strict context managers that guarantee scope-based cleanup.
```python
@contextmanager
def gpu_memory_scope(device: torch.device):
    try:
        yield
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
```
**Why it’s high impact:** It drastically reduces the cognitive load of tracking tensor references. By wrapping block evaluation or layer tuning in a `gpu_memory_scope`, you ensure that temporary graphs, gradients, and activations are always evicted, even if the block throws an exception.

### [x] 9. Build a Typed `SubprocessInterop` Layer for `llama.cpp`
**The Problem:** Interfacing with the modified `llama.cpp` binaries (`gguf_export.py`, `llamacpp_quality.py`) is currently done via raw `subprocess.run` calls. The code manually constructs CLI arguments, manually modifies `os.environ` (e.g., hacking `LD_LIBRARY_PATH` and `PYTHONPATH`), spawns side-threads to drain `stderr`, and parses CLI stdout strings via regex/json.
**The Refactor:** Extract a dedicated `LlamaCppInterop` class.
*   Move the environment injection (`PATH`, `LD_LIBRARY_PATH`) into this class.
*   Route all `stdout/stderr` streams directly into your existing `EventSink` using a standard logging bridge, rather than manual threading.
*   Return typed Dataclasses representing the result of the subprocess rather than parsing raw stdout in the caller.
**Why it’s high impact:** External binary calls are the most brittle part of any ML pipeline. Isolating them makes the system cross-platform friendly, easier to mock in unit tests, and prevents environment variable leakage into the parent Python process.

### [x] 10. Centralize Error Codes and Exception Typing
**The Problem:** You have a great start on a diagnostic registry (`infrastructure/diagnostics.py`). However, exceptions are often raised with inline, hardcoded strings: `raise ResourceAdmissionError("RES001 ...")` or `raise OSError("ACT001 ...")`. If a developer makes a typo in the code, the telemetry tracking that error code breaks.
**The Refactor:** Force custom exceptions to take a strictly typed Error Code Enum.
```python
class ErrorCode(str, Enum):
    RESOURCE_ADMISSION = "RES001"
    CUDA_OOM = "RES002"
    ARTIFACT_CORRUPTION = "ART001"

class NanoQuantError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(f"{code.value}: {message}")
        self.code = code
```
**Why it’s high impact:** It ensures 100% consistency between raised exceptions, the `diagnostics.py` registry, and the events emitted to your JSONL logs. It makes it trivial to generate documentation for troubleshooting.

### [x] 11. Consolidate "Magic Strings" and Schema Versions
**The Problem:** Throughout the codebase, there are raw strings representing schemas, backend types, and dtypes: `"bfloat16"`, `"factorized"`, `"dense"`, `"float32"`, `"llama.cpp-i32-lsb-v1"`, and `schema_version: 1` or `2`. These are duplicated across validation logic, artifacts, and config files.
**The Refactor:** Create a `constants.py` or a set of `Enums` for backend types, storage formats, and active schema versions. Replace raw strings in `Dict` payloads with `BackendType.FACTORIZED.value`.
**Why it’s high impact:** If you ever need to bump a schema version from `1` to `2` (or deprecate `float16`), you currently have to hunt down hardcoded `1`s in dozens of `atomic_write_json` calls across the infrastructure layer. Centralizing these guarantees format parity across the deployment pipeline.

## Implementation result

Completed on 2026-07-25 with compatibility-preserving boundaries:

1. Resident execution now enters through `ResidentQuantizationPipeline`; `QuantizationStateManager`,
   `BlockQuantizer`, and `LayerQuantizer` own resume, block preparation/selection, and layer factorization contexts.
2. Standard experiments have a strict, versioned YAML/JSON envelope and one `experiments/run_experiment.py`
   dispatcher. Historical numbered Python launchers remain supported because they are required provenance records.
3. Generic dataclass decoding now uses cached strict Pydantic adapters with forbidden extra fields and dotted error
   locations. Cross-field validation and dataclass semantic invariants remain explicit because they encode domain
   rules, not parsing boilerplate.
4. `TelemetryContext` unifies phase recording, lifecycle events, elapsed time, and failure context; resident
   operations route through it.
5. `AdmmParameters` and the resident quantizer objects replace production long-argument forwarding while the
   compatibility ADMM function remains available to external callers.
6. Research and dependency-isolated runtime `SafetensorsManager` boundaries own all raw safetensors imports.
7. Research `AtomicWorkspace` and runtime `atomic_output_directory` transactions now publish directory artifacts
   atomically and clean staging paths on every failure.
8. `gpu_memory_scope` guarantees collection, CUDA cache release, and optional synchronization; rank expansion and
   benchmark cleanup paths use the shared lifecycle helpers.
9. `SubprocessInterop`/`LlamaCppInterop` own typed commands, isolated environments, streaming drains, and typed
   results for GGUF export, mmproj export, and llama.cpp quality.
10. `ErrorCode`, coded operational exceptions, and the diagnostics registry share machine-readable codes without
    duplicated string prefixes.
11. Backend, config/event schema, runtime artifact schema, and packed-layout constants are centralized without
    coupling the deployment runtime to research infrastructure.
