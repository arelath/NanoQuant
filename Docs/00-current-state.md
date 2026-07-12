# Current-State Assessment

This document records why a rewrite is justified and what must be preserved. It is not a complete code review; detailed algorithm and optimization findings remain in the existing project documentation.

The current NanoQuant codebase is located at `D:\dev\research\NanoQuant-OfficalCode\`. The modified NanoQuant-capable llama.cpp reference is located at `D:\dev\research\llama.cpp\`; its CUDA implementation is `D:\dev\research\llama.cpp\ggml\src\ggml-cuda\nanoquant.cu`. These are current local anchors, not portable artifact identities.

## 1. Current workflow

The primary flow is approximately:

```text
CLI / numbered experiment script
  → construct a large flat configuration
  → NanoQuantModel.from_pretrained_quantize
  → load calibration model
  → prepare/tokenize calibration data
  → collect and shrink statistics
  → load a second full-precision teacher
  → sequential block reconstruction
      → tune unquantized remainder
      → collect/reuse Hessian
      → ADMM factorization and retries
      → outlier selection and scale fitting
      → factorized tuning
      → optional post-block refit
  → optional model-level KD
  → custom compressed state_dict
  → evaluation and decode benchmark scripts
```

This flow embodies valuable research behavior, but its responsibilities are not represented as stable component boundaries.

## 2. Structural problems motivating the rewrite

### Duplicated configuration

Defaults and fields are repeated across CLI dataclasses, `NanoQuantConfigDataclass`, and a dictionary factory. The CLI manually copies values into another configuration object, while much of the pipeline converts it back to a mutable dictionary. This creates drift risk and allows runtime counters/output rows to be inserted beside user intent.

Rewrite response: [one hierarchical configuration schema](03-configuration-reference.md), immutable resolved recipes, and separate plan/state/results.

### Monolithic block compression

Block compression currently owns model movement, activation caching, layer discovery, rank planning, Hessian collection, tuning, factorization, retry decisions, model mutation, memory cleanup, progress text, CSV writes, and block diagnostics.

Rewrite response: pure factorization/objective/policy components, a typed block application workflow, an executor that owns resources, and event/report infrastructure.

### Architecture knowledge is distributed

Model-family branches appear in model loading, decoder discovery, embedding handling, calibration, cached inputs, and distillation helpers. Adding a model can require edits across unrelated modules.

Rewrite response: one `ModelAdapter` contract and a shared adapter test suite.

### Mutable training and runtime representation

The custom linear module switches among latent training tensors, hardened signs, buffers/parameters, salient outlier training state, packed state, and kernel execution behavior. This makes state transitions and serialization hard to reason about.

Rewrite response: separate trainable, frozen logical, and immutable packed types with explicit validated conversions.

### Orchestration inside integration wrappers

The Hub model wrapper performs calibration, teacher loading, compression, tuning, checkpoint save/load, and publication behavior. A second auto-model path implements overlapping orchestration.

Rewrite response: application services own use cases; Hugging Face and Hub behavior are infrastructure adapters.

### Experiment chronology and zero-argument runfiles are useful, but mechanics are duplicated

Numbered Python files make experiment order obvious, and running a file without parameters is an effective executable record. Those are strengths to preserve. The current files also capture repeated environment loading, logging, paths, and invocation mechanics; filenames, printed labels, and active model/settings can then drift, and comparing two runs requires manual investigation.

Rewrite response: retain thin numbered zero-argument runfiles that construct the canonical typed configuration and invoke one shared application service. Add purpose/hypothesis/baseline, immutable run manifests, launcher hashes, semantic config diffs, and generated reports. See [ADR-0005](adr/0005-numbered-zero-argument-runfiles.md).

### Persistence is final-output-oriented

The existing checkpoint path primarily saves/loads a complete compressed state dictionary. There is no general atomic stage/layer commit protocol, semantic cache validation, or deterministic resume cursor.

Rewrite response: content-addressed stage artifacts, layer/block commit boundaries, logical seeds, and failure-injected resume tests.

### Large-model execution assumes too much residency

`device_map=auto` can help model loading, but the workflow later creates a separate full-precision teacher and the block pipeline still reasons about two model objects. A 70B BF16 model is roughly 140 GiB, so two copies are already roughly 280 GiB before working data.

Rewrite response: block-aligned source streaming, disk/RAM/GPU activation stores, resource preflight, and execution strategies independent of algorithms.

### Runtime and benchmark concerns overlap

The decode test file contains generation, sampling, cache management, compilation, monitoring, CLI, sweep execution, formatting, and benchmark aggregation. This makes it difficult to know whether observed throughput represents kernel speed, generation-loop overhead, fallback, synchronization, or benchmark bookkeeping.

Rewrite response: deployment-only runtime, explicit backend planner, separate generation engine, reference parity, and layered kernel/block/end-to-end benchmark suites.

### Logging is text/file oriented

Important decisions are printed and selected metrics are appended to CSV/Markdown within compression code. A failed run can require reading long logs and source code to infer why a rank, retry, or fallback occurred.

The block-final-versus-pre-quantization table in `D:\dev\research\NanoQuant-OfficalCode\outputs\019-phase1-weight-errors.md` is nevertheless a high-value diagnostic and must be carried forward.

Rewrite response: structured decision events, stable diagnostics, result artifacts, and reports generated entirely from structured data, including an equivalent block-final table before and after model-level KD.

### Evaluation is not yet a decision ladder

Perplexity, task evaluation, block diagnostics, and decode benchmarking exist, but they are not unified into cheap-to-expensive promotion tiers with comparable baselines, uncertainty, and cost.

Rewrite response: smoke/quick/standard/full evaluation specifications and predefined promotion gates.

## 3. Strengths to preserve

The rewrite should not discard the lessons already earned:

- validated NanoQuant and DBF factorization math;
- sequential tune-then-factorize block behavior;
- diagonal importance and Hessian experimentation;
- rank sensitivity, retry, and bit-budget work;
- salient outlier representations and tuning;
- least-squares scale fitting and rollback behavior;
- memory-conscious block batching/offload improvements;
- custom binary CUDA kernels and packing knowledge;
- current supported model-family behavior;
- optimizer correctness/performance fixes;
- existing focused tests for Hessian, allocation, retry, phase recovery, optimizer behavior, and outlier training;
- detailed prior correctness and optimization reviews.

These become parity fixtures, domain tests, adapter contracts, and historical decisions rather than being reimplemented from memory.

## 4. Behavior that needs an explicit decision

Some existing behavior should not be copied accidentally. Before cutover, decide and record:

- whether DBF remains supported or becomes an external/legacy factorizer;
- which calibration strategies are productized versus research-only;
- whether model-level KD is part of the default pipeline;
- which Hessian approximations are supported at each resource class;
- exact BPW accounting, including indexes, embeddings, padding, and metadata;
- deterministic guarantees across devices and compiler versions;
- checkpoint compatibility expectations for existing `.pt` files;
- remote-code trust policy;
- supported runtime backends and model shapes;
- minimum supported Python, PyTorch, CUDA, and GPU architectures.

These decisions should become ADRs rather than hidden defaults.

## 5. Baseline evidence required before replacement

Capture at least:

- three layer fixtures covering attention, MLP, and a difficult reconstruction case;
- two block fixtures from early and late model positions;
- one tiny complete pipeline artifact;
- one representative 1B run with stage timings and memory;
- current 20 tokens/second workload with an end-to-end profile;
- the compatible llama.cpp workload near 400 tokens/second with the same protocol where possible;
- checkpoint size/load and logical reconstruction data;
- current perplexity/task baselines;
- interruption behavior and amount of work currently lost.

Without these, the rewrite can become cleaner while silently changing the algorithm or failing to improve the actual bottleneck.
