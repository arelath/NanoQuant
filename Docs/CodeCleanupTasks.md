Based on a static analysis of the codebase you provided, the architecture is remarkably clean—dependency rules point inward, immutable configuration is strictly enforced, and you've successfully avoided mutable global state. 

However, because the codebase has grown rapidly to achieve 1B parity and implement complex profiling/resume logic, a few "God functions" and leaky abstractions have emerged. Here are the highest-value, behavior-preserving (S0) refactoring and code cleanup tasks you should tackle:

### [ ] 1. Decompose the `_run_resident_quantization_impl` God Function
**Location:** `src/nanoquant/resident_quantization.py`
**Problem:** This function is currently over 450 lines long. It handles model loading, prefix capture, preprocessing, calibration, planning, journal discovery, block restoration, block loops, tuning, freezing, commits, and reporting. It has up to 7 levels of indentation inside nested `recorder.phase()` contexts.
**Action:** Extract the major phases into private helper functions.
*   Extract the `for block_plan in plan.blocks:` loop body into a `_process_resident_block(...)` function.
*   Extract the initialization logic into `_setup_resident_environment(...)` (loading the model, capturing prefix, routing calibration).
*   Extract the resume logic into `_restore_committed_state(...)`.
*   *Why:* This will vastly improve readability and make it much easier to introduce the distributed or streaming executor variants later without duplicating massive blocks of orchestration code.

### [x] 2. Unify Device Memory High-Water Accounting
**Location:** Scattered across `quantization_stages.py`, `distillation.py`, `resident_calibration.py`, etc.
**Problem:** Multiple files do ad-hoc VRAM checks like:
`peak = int(torch.cuda.max_memory_allocated(request.device)) if request.device.startswith("cuda") else 0`
However, in `resident_quantization.py` you correctly realized that PyTorch's *reserved* memory is the actual metric that prevents OOMs, creating the `_peak_device_memory_bytes()` helper.
**Action:** 
*   Move `_peak_device_memory_bytes()` to `src/nanoquant/infrastructure/resource_usage.py` (next to `peak_process_memory_bytes`).
*   Replace all inline `max_memory_allocated` calls across the pipeline with this unified function. 
*   *Why:* Ensures VRAM accounting is exactly consistent across all phases and correctly represents true board-level memory pressure.

### [x] 3. DRY Up Optimizer State Hydration/Dehydration
**Location:** `src/nanoquant/application/tuning.py` and `src/nanoquant/application/distillation.py`
**Problem:** The logic to save and restore the optimizer state (extracting `exp_avg`, `exp_avg_sq`, Kahan compensation, step counts, and `CosineAnnealingLR` state) is nearly identical in both files. It spans ~40 lines of boilerplate in each file.
**Action:** 
*   Extract this logic into utility functions within `src/nanoquant/application/parity_adamw.py` (e.g., `capture_optimizer_state(optimizer) -> list[TuningOptimizerState]` and `restore_optimizer_state(optimizer, states)`).
*   *Why:* Reduces the size of the massive `tune()` and `distill_topk()` functions and centralizes the complex Kahan/BF16 tensor moving.

### [x] 4. Plug Hugging Face Abstraction Leaks using `ModelAdapter`
**Location:** `src/nanoquant/resident_quantization.py` and `src/nanoquant/resident_calibration.py`
**Problem:** The application layer is meant to be architecture-agnostic via `ModelAdapter`. However, the orchestration code still does manual Hugging Face topology traversals, such as:
*   `text_model = getattr(model, "model", model)`
*   `layer_container = getattr(getattr(model, "model", None), "layers", None)`
*   `cast(Any, text_model)(input_ids=tokens[:1], use_cache=False)`
**Action:** 
*   Add methods to `ModelAdapter` for `get_decoder_layers(model: nn.Module) -> nn.ModuleList` and `run_full_forward(model: nn.Module, tokens: Tensor)`.
*   Replace the `getattr` hacks in the orchestration files with these clean adapter calls.
*   *Why:* Keeps Hugging Face's changing internal API architectures securely quarantined inside `infrastructure/model_adapters.py`.

### [ ] 5. Type-Safe Validation in `validate_resident_run.py`
**Location:** `tools/validate_resident_run.py`
**Problem:** `_commit_payload` and `_committed_metrics` manually parse dictionaries with deeply nested `.get()` and `isinstance()` checks, spanning ~150 lines.
**Action:** 
*   You already have a robust, strict dataclass decoder (`from_dict` in `nanoquant.config.codec`). 
*   Create lightweight dataclasses for the expected JSON shapes of `BlockResult` and `LayerResult` (if the domain ones are too strict) and parse the JSON directly using `from_dict(BlockResultPayload, payload)`.
*   *Why:* Eliminates fragile string-key lookups and redundant type-checking logic, offloading the work to your existing codec infrastructure.

### Suggested Order of Execution
If you decide to take these on, I recommend doing **#2 (Memory Accounting)** and **#3 (Optimizer State)** first as they are isolated, purely mechanical S0 changes. Then tackle **#4 (Adapter Leaks)** before taking on the larger surgical extraction required for **#1 (The God Function)**.


----------------


Here are four more high-value, behavior-preserving (S0) refactoring opportunities. These target areas where duplicated logic or "magic strings" could eventually cause subtle bugs as the codebase evolves, particularly as you build out the deployment runtime.

### [ ] 6. Consolidate Factorized Reconstruction Math
**Location:** `src/nanoquant/application/layers.py` and `src/nanoquant/domain/scale_fit.py`
**Problem:** The core NanoQuant math—multiplying `left @ right` with `pre`, `mid`, and `post` scales, plus outlier masking and addition—is currently duplicated across:
*   `TrainableFactorizedLinear.forward`
*   `TrainableFactorizedLinear.dense_weight`
*   `FrozenReferenceLinear._materialize_dense_weight`
*   `FactorizedReferenceLinear.forward`
*   `domain.scale_fit.reconstruct()`

While the trainable modules use the `_SignSTE` wrapper and the frozen ones do not, the core algebraic sequence is identical. If you discover a faster way to fuse these operations (as planned in M7), you currently have to update it in 5 different places.
**Action:** 
*   Extract pure functional helpers into a new `nanoquant.domain.linear_math` module (e.g., `functional_factorized_linear(x, left, right, pre, mid, post, outliers)` and `functional_dense_reconstruction(...)`).
*   Refactor the `nn.Module` classes to act as thin state-holding wrappers that just call these pure functions.
*   *Why:* Guarantees that the training graph, scale-fitting, and inference references are mathematically locked together, preventing divergence bugs.

### [ ] 7. Deduplicate Calibration Hook Generators
**Location:** `src/nanoquant/application/calibration.py`
**Problem:** The logic that registers PyTorch hooks for statistic accumulation is almost entirely duplicated between `calibrate_causal_model()` (full model passes) and `calibrate_block()` (single block passes). Both functions define `forward_hook`, `backward_hook`, `profile_forward`, and `profile_backward` as nested closures, and manage their registration/removal. This accounts for ~100 lines of highly dense, repetitive code.
**Action:**
*   Extract these into factory functions, e.g., `_register_fisher_hooks(module, input_acc, output_acc, recorder) -> RemovableHandle`.
*   Use a context manager like `with _apply_calibration_hooks(modules, inputs, outputs, recorder):` to safely guarantee hook removal in a `finally` block.
*   *Why:* Simplifies the massive calibration entry points and makes it much easier to add new experimental hook types (like Activation-Aware Weight Quantization (AWQ) stats) later.

### [x] 8. Centralize Magic Artifact Type Strings
**Location:** Scattered across `commits.py`, `planning.py`, `artifact_gc.py`, `cleanup_run_activations.py`, and `validate_resident_run.py`.
**Problem:** Strings like `"layer-result"`, `"block-result"`, `"activation-generation"`, and `"quantization-plan"` are hardcoded throughout the codebase. In a content-addressed storage architecture, a typo in one of these strings doesn't throw a standard Python `AttributeError`—it silently creates an unreferenced artifact, breaks garbage collection, or causes validation to fail.
**Action:**
*   Create a simple namespace or Enum in `nanoquant.domain.models` (e.g., `class ArtifactTypes: BLOCK_RESULT = "block-result"...`).
*   Replace all raw string literals identifying artifacts with these constants.
*   *Why:* Enforces type-safety at the Python level for artifact schema identities, preventing invisible data orphaning.

### [x] 9. Centralize I/O and Hashing Boilerplate
**Location:** `safetensors_source.py`, `artifacts.py`, `activation_store.py`, `progress.py`
**Problem:** The exact same `_hash_file` function (reading a file in 1MB chunks to compute a SHA256) is defined independently in three different files. Similarly, the "write to temporary file, sync, then atomically replace" pattern is repeated manually in `RunDirectory.write_manifest`, `LocalArtifactWriter.commit`, `MmapGenerationWriter.commit`, and various checkpoint tools.
**Action:**
*   Create a `src/nanoquant/infrastructure/io_utils.py` module.
*   Extract `hash_file(path) -> str` and `atomic_write_json(path, payload)`.
*   Move the Windows-specific retry loop for `os.replace` (currently hiding inside `LocalArtifactStore._persist_validation`) into a generalized `safe_replace(src, dst)` helper in this new module.
*   *Why:* Removes boilerplate, standardizes file hashing, and ensures that cross-platform file locking edge cases (especially on Windows) are handled uniformly across the whole application.
