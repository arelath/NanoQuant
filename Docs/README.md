# NanoQuant Rewrite Design

Status: proposed rewrite specification  
Audience: maintainers, algorithm researchers, runtime/kernel engineers, and reviewers

This directory defines a clean-sheet design for the next NanoQuant implementation. It is intentionally more than a refactoring plan. The rewrite separates mathematical research, experiment execution, model integration, artifact storage, evaluation, and high-performance inference so that each can evolve without destabilizing the others.

The current implementation proved the algorithm and accumulated useful optimizations, but it now mixes concerns in ways that make changes expensive to validate:

- configuration is represented in several places;
- orchestration lives partly in model/Hub wrappers;
- block compression combines policy, math, model mutation, memory management, logging, and file output;
- training and packed inference states share one mutable linear-module implementation;
- numbered Python programs preserve useful experiment chronology and zero-argument reproducibility, but currently also repeat orchestration that can drift;
- the inference benchmark script also contains substantial runtime behavior;
- a normal run is too expensive to use as the first feedback loop;
- loading a calibration model and a separate full-precision teacher does not scale cleanly to 70B models.

The rewrite must preserve the validated NanoQuant mathematics while making the surrounding system easier to understand, test, resume, profile, and extend.

## Goals and success measures

The values below are initial engineering targets. They must be recorded against named hardware in the benchmark baseline before implementation begins.

| Goal | Rewrite success measure |
| --- | --- |
| Cleaner architecture | Mathematical packages do not import CLI, Hugging Face Hub, filesystem reporting, or model-specific traversal code. Dependency rules are enforced in tests. |
| Faster prototyping | A cached single-layer replay completes in under 60 seconds when the factorizer itself permits it; a tiny-model smoke run gives a useful signal in under 10 minutes on the reference development GPU. |
| Fast execution | Packed inference reaches at least 70% of the best compatible reference implementation on the same hardware and workload, or produces a profile that attributes the remaining gap. Decode, prefill, and end-to-end generation are measured separately. |
| Resume support | A terminated quantization run resumes from the last committed layer or block and loses no more than one unit of work. Completed stages are never silently rerun. |
| 1B-to-70B scaling | The same pipeline supports fully resident GPU execution and bounded-memory block streaming. Streaming peak model-weight memory is proportional to one block plus workspace, not total model size. |
| Actionable logging | Failures and quality regressions identify a stage, block/layer, relevant inputs, thresholds, resource state, and suggested next diagnostic. |
| Self-documenting runs | Every run produces an immutable manifest, resolved recipe, environment snapshot, stage timings, artifacts, comparisons, and a human-readable report. |
| Evaluation | A promotion ladder supplies cheap smoke signals, representative intermediate decisions, and publication-grade full evaluation. Comparisons include uncertainty and cost. |
| Testing | Pure math, adapters, storage, resume behavior, kernels, and end-to-end workflows have automated tests. Release candidates pass reference-versus-packed numerical parity and interruption/restart tests. |

The 20 tokens/second currently observed in the test framework and the approximately 400 tokens/second observed in the optimized llama.cpp implementation are starting measurements, not directly comparable conclusions. The rewrite first establishes an identical workload protocol and then optimizes the measured bottlenecks.

## Design principles

1. **One canonical representation per concept.** One configuration schema, one run manifest schema, one artifact schema, and one owner for device placement.
2. **Pure math at the center.** Factorization and reconstruction objectives accept tensors and typed values and return typed results. They do not traverse models or write reports.
3. **Policies are replaceable.** Rank allocation, outlier selection, retry rules, Hessian approximation, tuning, evaluation, and inference backends are explicit strategies.
4. **Execution is resource-aware.** GPU, CPU, pinned memory, and disk are planned resources rather than ad hoc `.cuda()`, `.cpu()`, and cleanup calls.
5. **Every expensive boundary is cacheable.** Calibration, activation capture, layer plans, factorization, packing, and evaluation have content-addressed inputs and outputs.
6. **Research and deployment representations differ.** Trainable latent factors and immutable packed runtime weights are separate types with explicit conversion.
7. **A run is an auditable object.** Reproducing a result must not depend on remembering which script, environment variable, or local file was used.
8. **Reference paths come first.** A slow, clear PyTorch implementation defines correctness for every optimized kernel and packed format.
9. **Failure is expected.** Long-running work commits progress atomically and can survive process death, OOM, machine restart, and evaluation interruption.
10. **Performance claims require protocols.** Throughput without model, hardware, prompt/decode lengths, batch size, cache policy, sampling, and synchronization details is not a result.
11. **Preserve useful research ergonomics.** Numbered experiment files and zero-argument launch remain first-class; the rewrite centralizes their mechanics rather than removing the convention.

## Documentation map

Read these in order for a full rewrite plan:

1. [Current-state assessment](00-current-state.md)
2. [Requirements and acceptance criteria](01-requirements.md)
3. [System architecture](02-architecture.md)
   - [Concrete domain objects and stage input/output contracts](02-domain-and-stage-contracts.md)
4. [Configuration, recipes, and run identity](03-configuration-and-runs.md)
   - [Concrete hierarchical configuration schema and migration map](03-configuration-reference.md)
5. [Execution, checkpoints, resume, and 1B-to-70B scaling](04-execution-and-scaling.md)
6. [Fast research and prototyping workflow](05-fast-prototyping.md)
7. [High-performance inference runtime](06-inference-runtime.md)
8. [Observability, diagnostics, and run reports](07-observability-and-reporting.md)
9. [Evaluation strategy](08-evaluation.md)
10. [Testing and quality gates](09-testing-and-quality.md)
11. [Artifact formats and compatibility](10-artifacts-and-compatibility.md)
12. [Delivery and migration plan](11-delivery-roadmap.md)
13. [Lessons carried forward and local reference implementations](12-lessons-carried-forward.md)
14. [Implementation task list and milestones](13-implementation-task-list.md)
15. [Artifact retention and disk usage](14-artifact-retention-and-disk-usage.md)
16. [Performance profiling and micro-profiling](15-performance-profiling.md)
17. [Behavior-preserving optimization catalog](16-behavior-preserving-optimizations.md)
18. [Glossary](glossary.md)
19. [Legacy numbered experiment migration inventory](22-legacy-experiment-migration-inventory.md)

The [architecture decision record directory](adr/README.md) records decisions that future contributors may otherwise be tempted to reverse without understanding their context.

## Proposed top-level commands

The CLI is a thin adapter over the same application API used by tests and Python callers:

```text
nanoquant inspect-recipe recipe.yaml
nanoquant calibrate recipe.yaml
nanoquant plan recipe.yaml
nanoquant quantize recipe.yaml
nanoquant resume runs/<run-id>
nanoquant pack runs/<run-id>
nanoquant evaluate runs/<run-id> --tier quick
nanoquant benchmark runs/<run-id> --suite decode
nanoquant compare runs/<candidate> runs/<baseline>
nanoquant report runs/<run-id>
```

Commands do not contain unique business logic. They validate input, invoke an application service, stream structured progress, and select an exit code.

Promoted research experiments also retain a numbered, zero-argument runfile:

```text
python experiments/020_low_rank_hessian_replay.py
```

The runfile constructs the same canonical `RunConfig` and calls the same application service. It contains no copied orchestration and its path/hash are recorded in the manifest. See [Lessons Carried Forward](12-lessons-carried-forward.md#2-preserve-numbered-experiment-files).

## Definition of done for the rewrite

The rewrite is complete when:

- the old and new pipelines produce equivalent layer results for a frozen reference recipe within documented numerical tolerances;
- a 1B model can use the resident executor and a 70B model can use the streaming executor without changing algorithm code;
- interrupted runs pass automated resume equivalence tests;
- packed checkpoints are loadable without importing training code;
- the inference runtime passes reference parity and the agreed performance gates;
- quick, standard, and full evaluation tiers produce comparable reports;
- all supported model families pass adapter contract tests;
- a run directory alone is sufficient to explain what was attempted, why, with which inputs, what happened, and how the result compared with its baseline.
