# Testing and Quality Gates

## 1. Testing philosophy

The quantization tools are experimental, but their behavior must not be mysterious. Research conclusions are only as trustworthy as the factorizer, model adapters, checkpoint writer, evaluator, and inference runtime that produced them.

Tests optimize for three properties:

1. **Locality:** failures identify one contract or invariant.
2. **Realism:** captured shapes and end-to-end scenarios cover actual behavior.
3. **Recoverability:** interrupted and corrupted work fails safely.

Line coverage is useful but not the definition of confidence. Contract, branch, shape, dtype, failure-mode, and artifact-version coverage matter more.

## 2. Test layout

```text
tests/
  unit/
    domain/
    application/
    runtime/
  property/
  contract/
    adapters/
    factorizers/
    stores/
    backends/
  integration/
    pipeline/
    resume/
    evaluation/
  cuda/
    kernels/
    runtime/
  performance/
  fixtures/
    generated/
    captured/
  golden/
```

Markers distinguish `cpu`, `cuda`, `slow`, `network`, `performance`, and `large_model`. Default PR tests do not unexpectedly download models or require a GPU.

## 3. Mathematical unit tests

Pure tensor tests cover:

- SVID/rank-one approximation and zero-sign convention;
- ADMM solve steps and scheduler boundaries;
- diagonal and Hessian-weighted reconstruction objectives;
- whitening and unwhitening algebra;
- scale-pre/mid/post least-squares fits and rollback;
- Fisher and residual outlier selection;
- rank allocation, rounding, caps, BPW accounting, and retry budgets;
- raw and normalized error calculations;
- int8 outlier and embedding quantization;
- binary pack/unpack and padding.

Cases include zero, constant, rank-one, ill-conditioned, rectangular, minimum-rank, maximum-rank, non-contiguous, FP32, BF16, and boundary-aligned tensors. Expected failures such as non-finite weights or impossible ranks have exact error codes.

Tests use small CPU tensors whenever GPU behavior is not the subject. This keeps the core suite fast and debuggable.

## 4. Property tests

Generated tests validate invariants over many shapes and seeds:

- pack then unpack preserves logical signs and original shape;
- serialization round trips preserve state and metadata;
- effective bit accounting equals serialized component sizes within declared container overhead;
- accepted scale fitting does not worsen its declared objective;
- retry ranks are monotonic, aligned, capped, and budgeted;
- deterministic seed derivation is independent of execution order;
- reconstructed packed and frozen logical states agree within tolerance;
- objective normalization is invariant to permitted global scaling;
- activation-store batch partitioning does not change concatenated output;
- cache identities change for every semantic input and remain stable for presentation-only changes.

When a property framework is used, minimized failing examples are saved as regression fixtures.

## 5. Contract tests

### Model adapters

Every adapter must prove:

- correct block count and ordering;
- complete and unique source tensor mapping;
- quantizable-layer discovery and canonical identity;
- prefix/block/suffix execution agreement with the source model;
- input capture without permanent model mutation;
- batch and sequence metadata correctness;
- tied-weight handling;
- resident and streamed block loading equivalence;
- clear rejection of unsupported architecture variants.

### Factorizers and policies

Every factorizer returns required factors/metrics, honors its generator, preserves input tensors, validates shapes, and serializes its configuration/version. Allocators always produce a valid plan under declared budget rules.

### Artifact and activation stores

All store implementations pass the same read/write/hash/commit/corruption suite. GPU, RAM, and mmap activation stores must yield equivalent batch values.

### Runtime backends

Every backend declares capabilities, rejects unsupported workloads with a reason, and passes reference parity for every declared shape/dtype/layout class.

## 6. Integration tests

The core integration fixture is a deterministic tiny causal transformer designed for fast CPU or small-GPU execution. Tests exercise:

1. resolve recipe;
2. prepare data;
3. calibrate;
4. plan;
5. quantize all blocks;
6. pack;
7. validate and load;
8. run inference;
9. evaluate smoke metrics;
10. render and compare reports.

Pinned tiny external model fixtures may complement this test, but the primary pipeline suite must work offline.

Additional integration scenarios:

- no outliers versus BF16/INT8 outliers;
- diagonal versus approximate/full Hessian;
- tuning disabled and enabled;
- resident versus streaming executor equivalence;
- CPU/RAM versus mmap activation stores;
- source checkpoint split across shards;
- optional global KD;
- loading a prior supported artifact version through migration;
- strict inference mode refusing an unsupported backend layout.
- a numbered zero-argument experiment runfile producing the same resolved config as direct Python/YAML construction;
- Experiment 019-style per-layer reconstruction and final-block/pre-KD report generation.

## 7. Resume and failure injection

The test executor can inject failure immediately before and after every commit step:

- event append;
- tensor write;
- hash calculation;
- commit descriptor write;
- atomic rename;
- run-reference update;
- next-layer/block activation commit.

For each injection point:

1. run until the injected failure;
2. terminate without graceful cleanup where appropriate;
3. resume the run;
4. compare the final artifact and metrics with an uninterrupted control;
5. verify no incomplete artifact was treated as committed;
6. verify completed units were not repeated.

Other failure cases:

- CUDA OOM and batch-size fallback;
- host OOM preflight rejection;
- disk full during temporary and commit writes;
- corrupt cached artifact;
- changed source file under the same path;
- expired worker lease;
- evaluator task failure after other tasks completed;
- cancellation during a long factorization attempt.

## 8. Kernel and runtime tests

The CUDA shape matrix is generated from supported real model shapes plus boundaries:

- ranks below/at/above tile multiples;
- padded input/output dimensions;
- batch 1 decode and multi-token prefill;
- supported accumulation/input dtypes;
- no outliers and each supported outlier encoding;
- bias/no bias;
- contiguous and supported-stride inputs;
- smallest and largest declared dimensions.

For each case:

- logical reconstruction and factorized PyTorch references agree;
- packed backend output meets absolute/relative error tolerance;
- guard regions or sanitizers detect out-of-bounds access in debug builds;
- repeated calls do not grow memory;
- unsupported cases take the documented path;
- deterministic mode is repeatable where promised.

Generation tests validate KV-cache positions, batched padding, stopping, greedy output parity, sampling seeds, long-enough decode to expose leaks, and compiled/eager agreement.

## 9. Evaluator tests

Evaluators are tested with synthetic logits and labels where the expected metric is exact. Tests cover:

- causal shift;
- BOS/EOS and padding masks;
- partial batches and final windows;
- batching invariance;
- sample-limit determinism;
- task prompt rendering;
- cached preprocessing;
- distributed reductions;
- paired comparison and confidence interval code;
- incomplete task results;
- baseline comparability checks.

## 10. Golden tests

Golden fixtures are used sparingly for stable, high-value outputs:

- resolved recipe serialization;
- run manifest and event schema examples;
- known tiny factorization result/metrics under deterministic mode;
- packed artifact inventory;
- run summary and comparison reports;
- numbered-runfile provenance and final-block error tables;
- migration of every supported prior schema.

Goldens store tolerances and producer versions. Updating a golden requires a review that explains the semantic change; `--update-goldens` is not an acceptable unexplained fix.

## 11. Performance tests

Performance tests do not run on noisy generic CI workers. Designated hosts record:

- host identity and health checks;
- GPU clocks/power mode and temperature range;
- driver, CUDA, compiler, and package versions;
- warm-up and repetition policy;
- raw sample distributions;
- accepted baseline revision.

Gates include:

- critical kernel latency/bandwidth;
- layer and block latency;
- time to first token and decode throughput;
- peak runtime VRAM;
- calibration/factorization replay timing;
- streaming I/O throughput;
- resume overhead.

Statistical tests or robust thresholds distinguish noise from regressions. A major environment change creates a new baseline series.

## 12. CI lanes

### Pull request CPU lane

Target: under 10 minutes.

- formatting, lint, types, import-boundary checks;
- domain unit and property tests;
- config/schema/migration tests;
- store tests using temporary local files;
- tiny pipeline smoke without CUDA;
- report generation.

### Pull request CUDA lane

Target: under 20 minutes on an available runner.

- selected kernel shape matrix;
- reference parity;
- tiny resident quantization;
- resident/streaming equivalence;
- short generation test.

### Nightly lane

- full unit/property seeds;
- all adapter fixtures;
- complete CUDA shape matrix;
- resume failure-injection matrix;
- standard evaluation on a small accepted artifact;
- artifact compatibility suite.

### Performance lane

- designated benchmark host;
- kernel/layer/generation comparisons;
- prototype-speed and streaming-I/O baselines;
- alerts on statistically significant regressions.

### Large-model canary

Periodic rather than per-PR:

- real sharded source reader;
- bounded-memory streaming plan;
- selected blocks or a complete run as budget permits;
- resume after intentional interruption;
- packed artifact load and inference.

## 13. Release gates

A release requires:

- no failing required CI lane;
- no unexplained reference-parity difference;
- no unresolved critical warning in canary runs;
- legacy comparison within approved tolerances;
- successful clean-install runtime inference;
- successful interruption/resume equivalence;
- readable manifests from every supported schema version;
- published quality and performance comparison against the previous release;
- captured comparison against the modified llama.cpp NanoQuant runtime for the agreed workload.

## 14. Bug workflow

Every correctness or performance bug should leave behind the smallest useful regression asset:

- a generated tensor case;
- a captured layer/block fixture;
- an artifact manifest;
- an event sequence;
- or a benchmark workload.

The test should fail before the fix and pass afterward. Large full-run logs alone are not a regression test.
