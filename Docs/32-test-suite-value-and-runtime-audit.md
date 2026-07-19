# Test Suite Value and Runtime Audit

Date: 2026-07-19  
Scope: `tests/`, pytest configuration, and test-only execution paths  
Status: baseline audit plus an implementation checklist; completed items are checked below

## Executive summary

The suite has substantial high-value coverage, especially around artifact integrity, resume behavior, numerical
parity, model adapters, and runtime export. The main problem is not that most tests are useless. It is that the
suite currently has no enforceable separation between fast CPU tests, real CUDA tests, external-data checks,
subprocess contracts, and multi-run integration scenarios.

The current inventory is:

| Measure | Observed |
| --- | ---: |
| Test modules | 147 |
| Test source lines | 21,514 |
| Explicit `test_*` functions | 657 |
| Collected cases after parametrization | 762 |
| Full-suite collection time with CUDA hidden | 25.06 seconds |

The highest-priority findings are:

1. The default suite can access the network and physical CUDA devices even though
   [the testing design](09-testing-and-quality.md) says default PR tests should do neither.
2. One optional pinned-dataset test can hang instead of quickly skipping offline.
3. Two cleanup CLI tests spend about 37 seconds starting Python and importing Torch five times.
4. The tiny-pipeline smoke test spends 28 seconds running 400 ADMM iterations for each of 12 layers even though
   its assertions test orchestration, artifacts, logging, and profiling rather than factorization convergence.
5. Six resident integration scenarios account for about 76 seconds and repeatedly construct, quantize, interrupt,
   resume, and reload nearly identical tiny Gemma models.
6. The exhaustive CUDA matrix repeats dimensions already covered by focused tests and performs 540 combinations
   inside one collected test, giving poor failure isolation and potentially large Triton compilation cost.
7. Numbered-experiment tests preserve useful audit contracts, but their current form repeats helpers and common
   assertions. One supposedly general test stops at Experiment 012 while the repository now has Experiment 019.

No complete test module can be proven to have literally zero value from a source and timing audit alone. Mutation
testing or defect-history data would be needed for that conclusion. This report therefore distinguishes safe
removals of ineffective assertions, strong consolidation candidates, and slow tests that should remain.

## Method

The audit used static inventory, collection, source inspection, call-site searches, and pytest duration reporting.
No production or test source was modified.

Measured runs:

| Run | Result | Pytest time | Important exclusions or limitations |
| --- | --- | ---: | --- |
| Unit + contract, CUDA hidden, cleanup CLI tests excluded | 671 passed, 43 skipped | 69.79 s | Measures the CPU-oriented body of the current “unit” tier |
| Unit + contract with CUDA visible while the device lease was owned | 677 passed, 39 setup errors | 105.16 s | Errors were lease conflicts, not product failures; useful evidence that default collection touches CUDA |
| Integration CPU subset | 42 passed, 2 deselected | 129.28 s | Excluded pinned external data, explicit CPU-offload CUDA test, and names containing `cuda` |
| Full collection, CUDA hidden | 762 cases | 25.06 s | Collection only |

Prior full-suite attempts in this worktree timed out at 120 seconds and 600 seconds while reaching
`test_local_pinned_datasets_and_gemma_tokenizer_match_retained_legacy_samples`. CUDA benchmarks were not forced
through the active device lease for this audit.

Timing is machine- and cache-dependent, but the concentration of cost is sufficiently large to guide cleanup.

## Priority 0: make the default suite deterministic

### 1. The documented markers and tiers do not exist

`Docs/09-testing-and-quality.md` says markers distinguish `cpu`, `cuda`, `slow`, `network`, `performance`, and
`large_model`, and that default tests do not unexpectedly download models or require a GPU. In practice,
`pyproject.toml` registers no markers and its only pytest option is `-ra`.

Real CUDA tests currently live under `tests/unit` and `tests/integration`, primarily in:

- `tests/unit/test_runtime_cuda_backend.py`
- `tests/unit/test_runtime_cuda_matrix.py`
- `tests/unit/test_device_batches.py`
- `tests/unit/test_resident_batching.py`
- `tests/unit/test_tuning.py`
- `tests/integration/test_cpu_offload_calibration_device.py`
- `tests/integration/test_quantization_stages.py`
- `tests/integration/test_resident_quantization.py`

With CUDA hidden, 43 unit/contract cases skip. With CUDA visible while another resident job owns the device, the
module-scoped autouse lease fixtures in the two CUDA runtime modules turn 39 cases into setup errors. Even
capability-only tests in `test_runtime_cuda_backend.py` acquire the physical-device lease because the fixture is
autouse for the entire module.

Recommendation:

- Register `cuda`, `slow`, `external_data`, `subprocess`, and `performance` markers.
- Mark CUDA tests explicitly instead of relying only on `skipif(torch.cuda.is_available())`.
- Make the normal CPU command exclude CUDA and external-data tests.
- Put capability checks that require no device in a separate CPU module so they do not acquire the CUDA lease.
- Run the CUDA tier only after the repository's process inspection, device lease, and `nvidia-smi` checks pass.
- Consider `tests/cuda/` for discoverability, matching the documented design.

This is a reliability fix, not merely a runtime optimization.

### 2. The “local pinned datasets” test is not local-only

`tests/integration/test_pinned_multiple_choice_inputs.py::test_local_pinned_datasets_and_gemma_tokenizer_match_retained_legacy_samples`
has high parity value, but its current failure behavior is poor:

- `AutoTokenizer.from_pretrained(snapshot, local_files_only=False)` permits network access despite receiving a
  resolved local snapshot.
- `load_pinned_multiple_choice_documents(...)` uses its default `local_files_only=False`, so a cache miss can start
  a Hugging Face dataset download.
- `except Exception` converts any implementation defect into a skip, not only an absent optional cache.
- The first missing task skips the entire test, so later retained task fixtures are not assessed.

This test is the identified full-suite stall candidate, and its code paths explain why it can wait on restricted
network access. It should not be removed because its retained text/token hashes are important parity evidence.

Recommendation:

- Pass `local_files_only=True` to both tokenizer and dataset loading.
- Catch `FileNotFoundError` (or a dedicated cache-miss exception), not `Exception`.
- Mark it `external_data` and keep the default suite offline.
- Report cached and missing tasks individually, or preflight all six caches before starting assertions.

## Priority 1: dominant runtime reductions

### Measured integration hotspots

The following eight tests consumed 113.57 seconds, or about 88% of the measured 129.28-second CPU integration run:

| Test | Time | Assessment |
| --- | ---: | --- |
| `test_tiny_pipeline_runs_entirely_on_rewrite_components` | 28.08 s | Valuable smoke path, but excessive numerical work for what it asserts |
| `test_resident_tuning_recipe_refits_blocks_and_resumes_exactly` | 25.83 s | High value, but repeats too many full resident runs in one test |
| `test_resident_quantization_commits_complete_transformers_model` | 18.12 s | High value, but is a multi-contract “mega-test” with repeated quantizations |
| `test_rolling_retention_keeps_only_latest_resume_generation` | 10.47 s | High-value recovery contract |
| `test_continuous_multiblock_run_reloads_committed_activation_boundary` | 9.05 s | Useful but overlaps the preceding two-block execution setup |
| `test_complete_frozen_run_can_be_distilled_committed_and_reloaded` | 9.02 s | High-value end-to-end KD contract; retain |
| `test_resident_quantization_factorizes_qkv_as_one_shared_input_group` | 7.81 s | Unique architecture behavior; retain |
| `test_reconstruction_rank_probe_covers_every_physical_unit_before_fitting` | 5.19 s | Unique planning/resume behavior; retain |

### 3. Tiny pipeline runs production-like ADMM work unnecessarily

`run_tiny_pipeline` calls `FactorizationAttemptStage()` without an explicit ADMM configuration. The stage default is
400 outer iterations. The fixture has two blocks and six quantized layers per block, so the single smoke test runs
approximately 4,800 outer iterations before checking pipeline composition, artifact commits, logging, reporting,
and profiler shape.

The test does not assert convergence quality against a retained numerical value. A separate collection of math and
factorization tests already owns numerical behavior.

Recommendation:

- Give the tiny pipeline a test recipe with one or two outer/inner iterations.
- Keep one deterministic factorization-quality test elsewhere with the production iteration policy if needed.
- Remove `assert result.elapsed_seconds < 600`; it is too loose to catch the present 28-second regression and is
  environment-dependent. Use a separately marked benchmark if a performance gate is desired.
- Reassess whether `src/nanoquant/tiny_pipeline.py` remains a supported composition root. Repository searches found
  no caller outside this one test and documentation. If the resident pipeline fully supersedes it, retire both the
  test and the test-only production path after a deliberate compatibility decision.

Expected benefit: likely tens of seconds without reducing the orchestration coverage.

### 4. Cleanup CLI tests pay for five full interpreter startups

The two tests in `tests/unit/test_cleanup_logical_artifact.py` took 20.83 and 16.31 seconds. Together they call the
script through `subprocess.run` five times. Each process imports `nanoquant.runtime`, which imports Torch and the
runtime package before argument validation.

The safety behavior is important; the repeated process startup is not.

Recommendation:

- Refactor `tools/cleanup_logical_artifact.py` into a callable `main(argv)` or a pure `cleanup(...)` function.
- Test dry-run, malformed hash, mismatch, missing hash, and apply behavior in-process.
- Retain one marked subprocess smoke test to prove CLI parsing and exit-code wiring.
- Optionally validate malformed/missing apply arguments before importing the runtime artifact loader.

Expected benefit: roughly 25–35 seconds per full CPU run on this machine.

### 5. Resident quantization integration tests repeat full workflows

`tests/integration/test_resident_quantization.py` is 826 lines. Its slowest tests combine many independently useful
contracts and invoke `run_resident_quantization` repeatedly:

- The complete-model test combines base compression, CPU-offload behavior, artifact validation, stale-journal
  handling, frozen loading, interrupted resume, live reports, profiling, and layer replay.
- The tuning/resume test performs a control run, layer-interrupted run and resume, then multiple epoch-interrupted
  runs before final resume.
- Rolling retention and continuous-boundary tests each build and compress a two-block Gemma fixture.

Focused unit tests already cover tuning resume equivalence, tuning checkpoint serialization, commit discovery,
activation retirement, and frozen loading. The integration layer needs to prove that these components connect, but
does not need to re-prove every branch through multiple full quantizations.

Recommendation:

- Keep one uninterrupted resident smoke and one interruption/resume equivalence scenario.
- Keep one end-to-end epoch-checkpoint resume scenario; leave optimizer/checkpoint branch coverage in
  `test_tuning.py` and `test_tuning_checkpoint.py`.
- Share immutable saved tiny-model snapshots through module-scoped fixtures.
- Reuse a completed resident fixture for read-only loader, export, replay, and distillation checks. Give mutating
  tests a copied fixture directory rather than recompressing the source model.
- Combine rolling-retention and committed-boundary observations if one interrupted two-block run can prove both.
- Move the pure helper tests listed later out of this integration module.
- Remove `assert replay.elapsed_seconds < 60`; it duplicates a broad design target without stable benchmark
  controls.

The unique shared-QKV and reconstruction-rank-probe scenarios should remain even if their setup is refactored.

### 6. The CUDA matrix duplicates focused dtype coverage

`tests/unit/test_runtime_cuda_matrix.py::test_cuda_backend_full_declared_dtype_outlier_workload_matrix` loops over:

- 3 factor dtypes;
- 3 scale dtypes;
- 5 outlier encodings;
- bias on/off;
- 3 input dtypes;
- prefill/decode.

It executes 540 cases inside one pytest item. `tests/unit/test_runtime_cuda_backend.py` separately parametrizes
outlier/token-shape parity, all factor dtypes, all scale/input dtype pairs, salient tiling, bias behavior, prefill,
decode, and generation. The full Cartesian test can still detect interactions, but it repeats the main effects at a
high device/compilation cost and reports any failure under one node ID.

Recommendation:

- Retain focused boundary and kernel tests.
- Replace the full Cartesian product with a pairwise covering array plus a small number of hand-selected risky
  combinations (INT8 outliers, word tails, bias, BF16, prefill, and decode).
- If a full matrix is required for releases, mark it `slow` and `cuda`, parametrize cases as pytest IDs, and exclude
  it from normal development runs.

## Priority 2: repeated and low-value coverage

### 7. Numbered-experiment tests are over-specified and partially stale

The suite has individual modules for Experiments 003–019, plus `test_promoted_recipes.py`,
`test_base_compression_recipe.py`, `test_recipe_deltas.py`, and architecture tests that inspect all numbered
launchers.

This is not valueless: numbered experiment definitions are auditable research inputs and accidental drift matters.
The duplication comes from how the contract is expressed:

- `_diff` is copied into `test_experiment006.py`, `007`, `010`, `011`, and `012`.
- Many modules repeat output-path, block-count, WDDM limit, quality backend, task-count, and publication assertions.
- `test_numbered_launchers_own_their_concrete_definitions` contains an exact 19-name manifest and then checks
  identity/name for every launcher.
- `test_each_numbered_launcher_owns_its_concrete_definition` dynamically checks the same identity ownership.
- `test_templates_are_unnumbered_and_concrete_configs_are_numbered` only loads Experiments 001–012. Its broad name
  is obsolete now that 013–019 exist, and several assertions intentionally do not apply to the newer
  reconstruction/shared-input cohort.

Recommendation:

- Create one shared `config_diff_paths` helper.
- Use a declarative table for common launcher contracts: number, name, model, block count, output paths, runtime
  family, and publication destination.
- Keep focused per-experiment tests only for the novel hypothesis/delta introduced by that experiment.
- Replace the exact filename manifest with dynamic checks for unique, contiguous numbers and identity/name
  agreement. Preserve an explicit manifest only if it is intended as an immutable publication ledger.
- Rename the 001–012 cohort test to state its historical scope, or replace it with template-family-specific
  assertions covering all current experiments.

This consolidation should reduce maintenance and collection/import work while preserving the historical audit
contract.

### 8. Three workflow provenance tests repeat one generic contract

These tests have almost identical arrange/execute/assert structure:

- `test_quality_workflow_records_config_and_launcher_provenance`
- `test_benchmark_workflow_records_config_and_launcher_provenance`
- `test_short_decode_workflow_records_config_and_launcher_provenance`

Each workflow still needs a contract proving it attaches resolved config and launcher provenance. Removing two would
permit one implementation to drift unnoticed. Consolidate the pattern rather than the coverage: use a shared helper
or parametrized adapter contract, while leaving workflow-specific output and publication assertions in their own
modules.

Runtime savings will be small; the main benefit is lower test-code duplication.

### 9. Pure unit tests are collected as integration tests

The following tests do no integration work and should move to unit modules:

- `test_numerical_batch_shapes_invalidate_resume_identity`
- `test_epoch_cooldown_skips_initial_loss_and_sleeps_after_training_epochs`
- `test_forward_metadata_clone_isolates_nested_tensor_mutation`
- `test_global_distillation_rejects_invalid_cooldown` (six parametrized validation cases)

Moving them does not directly reduce total runtime, but it makes tier selection honest and lets the integration tier
mean “constructs multiple real components or crosses a persisted boundary.”

### 10. Weak elapsed-time assertions add little or no signal

The assertions below are not controlled performance tests:

- `test_tiny_pipeline_runs_entirely_on_rewrite_components`: `elapsed_seconds < 600`
- `test_resident_quantization_commits_complete_transformers_model`: replay `elapsed_seconds < 60`

They depend on host load and are too loose to catch the measured hotspots. Remove them from correctness tests. If
the targets matter, create marked performance tests with warmups, named hardware/protocol, synchronization where
required, repetitions, and an explicit regression policy.

### 11. Golden “Experiment 019” naming is now ambiguous

`tests/unit/test_golden_reports.py` and `tests/golden/experiment019-block1-reconstruction.md` refer to a frozen legacy
Experiment 019 under `evidence/m0/20260712T052926Z`. The active numbered Experiment 019 is now the Llama 3.2 1B
compression launcher. The golden report remains valuable, but its name now suggests the wrong experiment.

Recommendation: rename the helper, test, and golden file to identify the frozen legacy provenance or phase-one
reconstruction fixture rather than “Experiment 019.” Keep the manifest/hash checks.

### 12. A few slow tests should be re-tiered, not removed

These tests are relatively expensive but defend important boundaries:

- `test_runtime_import_does_not_load_research_packages` took 7.85 seconds because it starts a clean interpreter and
  imports the Torch-backed runtime. A subprocess is required to prove module isolation. Mark it `subprocess` or
  `slow`; do not replace it with an already-contaminated in-process check.
- The architecture AST tests took about 2.6 seconds together. They enforce dependency direction and recipe
  placement and should remain.
- The Windows cross-process device-lease test took about 1.1 seconds. It protects a critical long-running-job safety
  invariant and should remain.
- Global distillation, shared QKV, rank-probe resume, artifact corruption, and resume-equivalence tests are expensive
  but exercise failure modes that small pure units cannot establish.

## Proposed suite structure

The directory layout need not change immediately; enforceable markers and commands are the important part.

| Tier | Contents | Default? | Suggested target |
| --- | --- | --- | ---: |
| `fast` | Pure CPU units, schema/contracts, small tensor math | Yes | under 30–45 s |
| `integration_cpu` | Resident smoke, resume, stores, frozen loading, KD | Yes in CI; optional local command | under 60–90 s after fixture reuse |
| `subprocess` | CLI parsing, clean-import and cross-process contracts | CI | separately reported |
| `cuda` | Kernel, transfer, Triton, CUDA runtime and resident-device tests | No | serialized behind device lease |
| `external_data` | Pinned tokenizer/dataset cache parity | No | strictly offline unless an explicit fetch job runs |
| `performance` | Runtime/throughput/memory regression protocols | No | dedicated hardware only |

One possible command policy is:

```powershell
# Normal local feedback; never uses CUDA or the network.
.\.venv\Scripts\python.exe -m pytest -q -m "not cuda and not external_data and not performance and not slow"

# CPU integration and process contracts.
.\.venv\Scripts\python.exe -m pytest -q -m "integration_cpu or subprocess"

# Explicit device run after lease/process/GPU checks.
.\.venv\Scripts\python.exe -m pytest -q -m cuda

# Cached parity inputs only; loaders must use local_files_only=True.
.\.venv\Scripts\python.exe -m pytest -q -m external_data
```

The exact default marker expression should be agreed with CI, but `pytest -q` must not opportunistically claim an
available GPU or contact Hugging Face.

## Recommended implementation order

- [x] **Fix default reliability:** pinned tokenizer/dataset loading is now strictly offline; markers are registered;
  default pytest excludes CUDA, external-data, and performance tests; all physical CUDA tests are marked; and the
  two CPU-only CUDA capability checks no longer live behind an autouse device lease.
- [x] **Remove avoidable work:** the tiny smoke uses an explicit one-iteration ADMM recipe, four of five cleanup CLI
  subprocess calls are now in-process, and the two uncontrolled elapsed-time assertions were removed. Focused
  timing reduced the tiny smoke from 28.08 seconds to 6.53 seconds and the cleanup pair from 37.14 seconds to
  4.45 seconds on the audit machine.
- [ ] **Refactor resident integration fixtures:** one base compression, one resume scenario, reusable immutable
  snapshots/completed runs, and fewer repeated end-to-end branches. This remains a larger follow-up because it
  changes fixture ownership across the highest-value recovery tests.
- [ ] **Reduce the CUDA Cartesian matrix:** the current 540-case matrix is now explicitly `cuda` and `slow`, so it
  cannot run in the default suite, but pairwise reduction remains to be implemented and validated on an idle GPU.
- [ ] **Consolidate experiment contracts and provenance helpers:** the repeated config-diff helper and exact launcher
  filename ledger were consolidated, and the Experiment 001–012 cohort test was renamed to state its scope. The
  three workflow provenance contracts still need a shared parametrized helper.
- [x] **Rename stale legacy Experiment 019 golden fixtures and move pure unit tests out of integration:** the golden
  fixture now uses legacy-phase-one naming; resident helper tests and global-distillation validation cases now live
  under `tests/unit`.
- [x] **Re-run timing and establish a clean post-pass baseline:** the default offline/CPU suite completed with
  715 passed and 47 deselected in 155.19 seconds. The strictly offline external-data tier completed with one passed
  and 761 deselected in 17.95 seconds. The CUDA tier collected 46 cases without executing them because the device
  was not idle. Ruff and strict mypy both passed. The default suite is deterministic now, but its 2:35 runtime
  confirms that resident fixture reuse remains the next performance target. Do not turn the former loose elapsed
  assertions into hard gates without a controlled protocol.

## Expected outcome

The first two implementation steps should remove obvious hangs and save approximately 50–60 seconds on this
machine. Resident fixture reuse and reduced repeated workflows offer the largest additional CPU saving. The result
should be a fast, deterministic feedback tier plus explicit high-value CUDA, external-data, process, and full
integration tiers—not a smaller suite that silently gives up parity or recovery coverage.
