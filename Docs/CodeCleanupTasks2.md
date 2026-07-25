Here are the highest-value code cleanup and refactoring opportunities to improve maintainability, readability, and DRY (Don't Repeat Yourself) principles.

### 1. Break Up `_run_resident_quantization_impl` (High Impact)
- [x] Break up `_run_resident_quantization_impl` into focused orchestration helpers.

**Location:** `src/nanoquant/resident_quantization.py`

This function is currently a **~600-line monolith** (spanning from line ~610 to ~1250). It handles state recovery, tensor loading, memory bound checks, calibration, rank planning, ADMM factorization loops, bias correction, low-rank patching, and artifact committing.

**Refactoring Strategy:**
Extract this into a dedicated orchestration class (e.g., `ResidentQuantizer`) or break it into logical helper functions.
*   Extract `_initialize_and_calibrate(...)`
*   Extract `_plan_ranks_and_objectives(...)`
*   Extract `_process_shared_input_groups(...)`
*   Extract `_process_standard_layers(...)`
*   Extract `_finalize_and_assemble_model(...)`

This will drastically reduce the cognitive load required to understand the quantization loop and make it much easier to write isolated tests.

### 2. De-duplicate Experiment Boilerplate
- [x] De-duplicate experiment launcher boilerplate with a shared execution entry point.

**Location:** `experiments/001...` to `experiments/029...`

You have almost 30 experiment files that share identical boilerplate for bootstrapping the workflow. Every file repeats the `if __name__ == "__main__": raise SystemExit(...)` block.

**Refactoring Strategy:**
Move to a registry or decorator-based pattern. You can define experiments declaratively and have a single CLI entry point run them.

*Before:*
```python
EXPERIMENT = define_compression_quality_experiment(...)

if __name__ == "__main__":
    raise SystemExit(run_compression_quality_experiment(...))
```

*After:*
Create an `experiments/registry.py` or use a decorator:
```python
@register_experiment(number=3, name="compress-and-benchmark-gemma-3-4b-it")
def gemma_4b_experiment():
    return define_compression_quality_experiment(...)
```
Then, your CLI (`nanoquant run exp:3`) dynamically loads the definition and automatically handles the `run_compression_quality_experiment` boilerplate. You can delete hundreds of lines of execution boilerplate across the `experiments/` directory.

### 3. Abstract Explicit Memory Cleanup
- [x] Introduce and adopt a reusable explicit-memory-cleanup abstraction.

**Location:** Throughout `application/` and `infrastructure/` layers.

There are dozens of instances of this exact block:
```python
del working_block
gc.collect()
if request.device.startswith("cuda"):
    torch.cuda.empty_cache()
```

**Refactoring Strategy:**
Abstract this into a reusable context manager. It centralizes the logic and ensures cleanup happens reliably even if exceptions are raised.

```python
from contextlib import contextmanager
import gc
import torch

@contextmanager
def explicit_memory_cleanup(device: str | torch.device):
    try:
        yield
    finally:
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage:
with explicit_memory_cleanup(request.device):
    working_block = environment.adapter.load_block(...)
    # ... do work ...
```

### 4. Flatten Heavily Nested Context Managers
- [x] Flatten deeply nested context managers in the identified workflow and infrastructure paths.

**Location:** `resident_quantization.py`, `compression_export_workflow.py`, `hf_language_model.py`

There are places where code indents 6 to 8 levels deep just to manage recording phases and tensor reads.
```python
with _logged_operation(events, "calibration_persist", ...):
    with recorder.phase("calibrate"):
        with recorder.phase("persist"):
            calibration = persist_calibration(...)
```

**Refactoring Strategy:**
Python 3.9+ allows multiple context managers in a single `with` statement.
```python
with _logged_operation(events, "calibration_persist", ...), \
     recorder.phase("calibrate"), \
     recorder.phase("persist"):
    calibration = persist_calibration(...)
```
Or, add a helper to `PhaseRecorder` that accepts multiple phases to yield a combined context. This will push your code back to the left side of the screen and greatly improve readability.

### 5. Replace "Magic Strings" with Enums for Artifact Types
- [x] Replace artifact-type string constants with a strongly typed string enum.

**Location:** `src/nanoquant/domain/models.py`

Currently, `ArtifactTypes` is a class holding string constants:
```python
class ArtifactTypes:
    LAYER_RESULT = "layer-result"
    SHARED_INPUT_GROUP_RESULT = "shared-input-group-result"
    # ...
```
Because these are strings, type checkers cannot enforce that a function taking an `artifact_type: str` is receiving a valid artifact type.

**Refactoring Strategy:**
Make this a `str, Enum` (just like `RunStatus` and `ExecutorKind`).
```python
from enum import Enum

class ArtifactType(str, Enum):
    LAYER_RESULT = "layer-result"
    SHARED_INPUT_GROUP_RESULT = "shared-input-group-result"
    # ...
```
Then update functions like `LocalArtifactStore.begin_write(..., artifact_type: ArtifactType)` to benefit from strict type-checking and IDE auto-completion.

### 6. Remove `cast(Any, model)` by Defining a `HuggingFaceModel` Protocol
- [x] Define and adopt a typed Hugging Face model protocol in model execution paths.

**Location:** `src/nanoquant/infrastructure/model_adapters.py`, `streamed_language_model.py`

To bypass type checkers when accessing Hugging Face specifics, the code frequently does this:
```python
cast(Any, model).config.use_cache = False
logits = cast(Any, self.teacher)(input_ids=batch, use_cache=False).logits
```

**Refactoring Strategy:**
Define a strict `Protocol` for what you expect a model shell to provide. This maintains your strict typing without relying on `Any`.

```python
from typing import Protocol
from transformers import PretrainedConfig

class HFModelShell(Protocol):
    config: PretrainedConfig
    def __call__(self, input_ids: torch.Tensor, use_cache: bool = False, **kwargs) -> Any: ...

# Usage:
hf_model = cast(HFModelShell, model)
hf_model.config.use_cache = False
```

### 7. Refactor Chunked Tensor Reductions
- [x] Centralize and adopt reusable chunked tensor reductions.

**Location:** `domain/metrics.py`, `resident_quantization.py`, `application/kl_budget.py`

There are several places where you manually chunk tensors to avoid OOM during reductions (e.g., calculating MSE or KL divergence).
*Example:*
```python
for start in range(0, prediction_rows.shape[0], 256):
    stop = min(start + 256, prediction_rows.shape[0])
    error = prediction_rows[start:stop].float()
    # ... math ...
```

**Refactoring Strategy:**
Create a reusable utility for "batched reduction" in `nanoquant/domain/linear_math.py`.
```python
from typing import Callable

def chunked_reduce(
    tensor: torch.Tensor,
    chunk_size: int,
    reduction_fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    total = torch.zeros((), device=tensor.device)
    for start in range(0, tensor.shape[0], chunk_size):
        stop = min(start + chunk_size, tensor.shape[0])
        total += reduction_fn(tensor[start:stop])
    return total
```
This isolates the chunking math and makes the business logic (MSE, KL divergence) much more declarative.

Here are several more high-value refactoring and cleanup opportunities. These focus on reducing boilerplate, catching errors at compile-time instead of runtime, and standardizing common patterns across the codebase.

### 1. Strongly Typed Event Payloads (High Impact)
- [x] Add strongly typed payloads and methods for well-known events.

**Location:** `src/nanoquant/ports/event_sink.py` and everywhere `events.emit(...)` is used.

Currently, the `EventSink` protocol accepts arbitrary keyword arguments for its payload:
```python
def emit(self, stage: str, severity: str, name: str, **fields: object) -> Event | None: ...
```
This is heavily used throughout the pipeline (e.g., `events.emit("resident-quantization", "info", "block.completed", block=block_index, actual_bits=...)`). Because `**fields` is untyped, type checkers cannot catch typos in metric names, missing required fields, or type mismatches. Since your downstream diagnostics and reports rely on these exact JSON keys, this is fragile.

**Refactoring Strategy:**
Use `TypedDict` or specific `dataclass` payloads for well-known events.
*Before:*
```python
events.emit(
    "resident-quantization", "info", "layer.committed",
    block=block_index, layer=layer_plan.layer.path, rank=layer_result.frozen_state.rank
)
```
*After:*
```python
from typing import TypedDict

class LayerCommittedPayload(TypedDict):
    block: int
    layer: str
    rank: int
    # ...

events.emit_layer_committed(LayerCommittedPayload(
    block=block_index, layer=layer_plan.layer.path, rank=layer_result.frozen_state.rank
))
```
Alternatively, define an `EventPayload` protocol and allow `events.emit(payload: EventPayload)`.

### 2. Consolidate Identity & Semantic Hashing (Medium Impact)
- [x] Consolidate canonical semantic hashing behind one shared utility.

**Location:** Scattered across `domain/models.py`, `application/diagnostics.py`, `infrastructure/cache.py`, etc.

The codebase repeatedly re-implements the exact same SHA-256 canonical JSON hashing logic:
```python
return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
```
There are dozens of variations of this (sometimes encoding to `utf-8`, sometimes checking string lengths, etc.).

**Refactoring Strategy:**
Centralize this in a single utility (e.g., in `nanoquant.config.codec` or a new `nanoquant.domain.identity` module).
```python
def semantic_hash(payload: object) -> str:
    """Produce a stable sha256 semantic hash from a canonicalized object."""
    encoded = canonical_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
```
Then replace all inline `hashlib.sha256(...)` calls that deal with object identity.

### 3. Unify Runtime Manifest Marshaling (Medium Impact)
- [x] Introduce a lightweight shared runtime manifest decoder and migrate manual parsing.

**Location:** `src/nanoquant/runtime/bundle.py` and `src/nanoquant/runtime/packed_artifact.py`

Because the `runtime/` module is deliberately isolated from the `config/` module (to ensure inference deployments don't require research dependencies), you've manually re-implemented JSON unmarshaling using `_mapping()`, `_sequence()`, `_string()`, and `_integer()`.
This results in highly verbose, repetitive parsing functions like `_manifest_from_payload`.

**Refactoring Strategy:**
Extract a lightweight, standalone deserializer for the `runtime` package. You don't need the full reflection power of `config/codec.py`, but a simple generic dictionary unmarshaler using `dataclasses.fields` can eliminate hundreds of lines of error-prone, manual dictionary lookups.

*Instead of:*
```python
RuntimeBundleMember(
    _string(member.get("path"), f"manifest.members[{index}].path"),
    _integer(member.get("bytes"), f"manifest.members[{index}].bytes"),
    _string(member.get("sha256"), f"manifest.members[{index}].sha256"),
)
```
*Create a minimal `runtime.codec` that handles type-safe instantiation of runtime dataclasses based on their type hints.*

### 4. Flatten Deeply Nested Config Mutations
- [x] Replace deeply nested recipe config mutations with dotted-path overrides.

**Location:** `src/nanoquant/experiments/recipes/base_compression.py`

Because Python dataclasses lack a native "lens" or deep-replace mechanism, modifying nested configurations looks like this:
```python
_TEMPLATE = config_delta(
    LARGE_MODEL_COMPRESSION_TEMPLATE,
    model=config_delta(
        LARGE_MODEL_COMPRESSION_TEMPLATE.model,
        source=MODEL_SOURCE,
        revision=MODEL_REVISION,
    ),
    block_tuning=config_delta(
        LARGE_MODEL_COMPRESSION_TEMPLATE.block_tuning,
        microbatch_size=1,
    ),
)
```
This is hard to read and scales poorly as configurations get deeper.

**Refactoring Strategy:**
You already have a powerful `apply_overrides(config, overrides: dict)` function in `nanoquant/config/codec.py` that accepts dotted paths. Use it directly in your experiment templates!

*After:*
```python
_TEMPLATE = apply_overrides(LARGE_MODEL_COMPRESSION_TEMPLATE, {
    "model.source": MODEL_SOURCE,
    "model.revision": MODEL_REVISION,
    "block_tuning.microbatch_size": 1,
    "runtime.block_forward_batch_size": 1,
})
```
This massively cleans up the experiment definition files and makes intent instantly readable.

### 5. Abstract Safetensors Loading Patterns
- [x] Centralize repeated safetensors loading and device-placement patterns.

**Location:** `infrastructure/hf_calibration_dataset.py`, `infrastructure/safetensors_source.py`, `runtime/packed_artifact.py`, etc.

The pattern of opening a safetensors file, fetching specific keys, and transferring them to a device is repeated everywhere:
```python
with safe_open(path, framework="pt", device="cpu") as handle:
    left = handle.get_tensor("left")
    right = handle.get_tensor("right")
```

**Refactoring Strategy:**
Add a small helper to your `TensorStore` or create a new context manager that yields a dictionary of requested tensors, handling the `safe_open` context and device placement internally.

```python
def load_tensors(path: Path, keys: tuple[str, ...], device: str = "cpu") -> dict[str, torch.Tensor]:
    result = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in keys:
            tensor = handle.get_tensor(key)
            result[key] = tensor if device == "cpu" else tensor.to(device)
    return result
```

### 6. Standardize "Device / DType" Mapping
- [x] Standardize dtype parsing and device/dtype mapping across research and runtime paths.

**Location:** Scattered.

There are many ad-hoc dictionaries mapping string types to torch types:
```python
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
```
This appears independently in `runtime/benchmark.py`, `runtime/bundle.py`, `runtime/validation.py`, `runtime/cuda_backend.py`, `resident_calibration.py`, etc.

**Refactoring Strategy:**
Move this to a shared location. For the `runtime/` boundary, put it in `nanoquant.runtime.logical` (where `canonical_torch_dtype` already exists). For the broader app, put it in `nanoquant.domain.linear_math`. Provide `parse_torch_dtype(name: str) -> torch.dtype` to ensure consistent error handling ("unsupported dtype...") everywhere.
