# Lessons Carried Forward

The rewrite should correct structural problems without discarding practices that made the current research workflow effective. This document records those practices as requirements.

## 1. Reference locations

These are the current local reference locations for the rewrite effort:

| Purpose | Current location |
| --- | --- |
| NanoQuant research/application codebase | `D:\dev\research\NanoQuant-OfficalCode\` |
| Experiment 019 block/weight error report | `D:\dev\research\NanoQuant-OfficalCode\outputs\019-phase1-weight-errors.md` |
| Modified llama.cpp with NanoQuant support | `D:\dev\research\llama.cpp\` |
| Modified llama.cpp NanoQuant CUDA implementation | `D:\dev\research\llama.cpp\ggml\src\ggml-cuda\nanoquant.cu` |

Convenience links from this document:

- [Experiment 019 error report](../../outputs/019-phase1-weight-errors.md)
- [Modified llama.cpp NanoQuant CUDA implementation](../../../llama.cpp/ggml/src/ggml-cuda/nanoquant.cu)

Absolute paths document the current workstation layout. New manifests and artifacts must use repository identities, revisions, hashes, and portable logical references; correctness must not depend on these paths remaining unchanged.

## 2. Preserve numbered experiment files

Numbered experiment files provide a simple chronological research record. The number answers “what was tried next?” even before opening a database or report:

```text
019_phase1_diagonal_baseline.py
020_low_rank_hessian_replay.py
021_residual_outlier_followup.py
```

The rewrite retains this convention.

### Rules

1. Numbers increase monotonically and are never reused for a semantically different experiment.
2. A completed experiment file is immutable. A changed hypothesis or numerical setting receives the next number.
3. The number is a human chronology label, not the run's globally unique identity. Retries/resumes retain the same experiment number and run ID; a changed configuration creates a new run/fork according to policy.
4. The filename includes a short descriptive slug; the file also contains purpose, hypothesis, and baseline.
5. Sweeps may use one numbered file whose explicit experiment definition produces named child runs.
6. Reports display both the experiment number and immutable run ID.

Chronology remains understandable in a directory listing, while the run manifest supplies exact identity and lineage.

## 3. Preserve zero-argument runfiles

A runfile that requires no command-line parameters is valuable because it is an executable record of the intended run:

```text
python experiments/020_low_rank_hessian_replay.py
```

There is no shell history to reconstruct and no risk that a required flag was omitted from a note.

The rewrite's numbered runfile should be thin:

```python
from nanoquant.application import run_experiment
from nanoquant.config import (
    IntentConfig,
    ModelConfig,
    ObjectiveConfig,
    ObjectiveKind,
    RunConfig,
)


CONFIG = RunConfig(
    model=ModelConfig(
        source="google/gemma-3-4b-it",
        revision="0123456789abcdef",
    ),
    intent=IntentConfig(
        experiment_number=20,
        name="low-rank-hessian-replay",
        purpose="Test a bounded-memory covariance approximation on difficult layers.",
        hypothesis="The approximation reduces held-out block loss at acceptable cost.",
        baseline_run="run_01J_BASELINE",
    ),
    calibration=CalibrationConfig(
        objective=ObjectiveConfig(
            kind=ObjectiveKind.LOW_RANK_DIAGONAL,
            low_rank=256,
        ),
    ),
)


if __name__ == "__main__":
    raise SystemExit(run_experiment(CONFIG, launcher_path=__file__))
```

The exact imports may be simplified by a builder, but the design properties are mandatory:

- no `argparse` or required CLI flags;
- no local implementation of model loading, logging, checkpointing, or evaluation;
- no copied default values merely to match the framework;
- experiment-defining choices are visible in the typed `RunConfig`;
- credentials and host resource discovery stay outside the experiment definition;
- the launcher path and content hash are captured in the run manifest;
- the framework validates that `intent.experiment_number` agrees with the number in the launcher filename;
- execution still writes the complete resolved recipe, including canonical defaults.

YAML remains useful for generated sweeps, interoperability, and inspection. It does not replace the option to keep a numbered, zero-argument Python runfile as the durable human experiment record. A runfile may load a colocated YAML recipe when that is clearer, provided the pair is treated as one hashed launcher definition.

## 4. Preserve final block error versus its baseline

The current [Experiment 019 error report](../../outputs/019-phase1-weight-errors.md) demonstrates a particularly useful diagnostic. It contains:

1. normalized objective-weighted weight reconstruction error for every target layer by block;
2. a `Final Error Before Model-Level KD` snapshot;
3. per-layer residual block-loss increase relative to the loss immediately before that layer was quantized;
4. a `Block final` value relative to the block-entry pre-quantization loss, after all layer tuning/refitting/finalization;
5. positive/negative/n-a semantics that make regressions, improvements, and near-zero denominators visible.

The rewrite must preserve this information as canonical structured results and render an equivalent human-readable table.

### Required block loss snapshots

For every block, record:

- `source_reference`: original base-model block evaluated on original teacher inputs/targets;
- `block_entry`: working block on compressed inputs before quantizing any target layer;
- `after_layer`: one snapshot after each accepted, tuned, and frozen layer;
- `after_post_block_refit`: when configured;
- `final_frozen_pre_kd`: exact finalized state propagated to the next block;
- `final_post_kd`: when global KD later changes the block/model.

### Required comparisons

Render at least:

```text
final_frozen_pre_kd - block_entry
(final_frozen_pre_kd - block_entry) / max(abs(block_entry), configured_floor)

final_frozen_pre_kd - source_reference
final_post_kd - final_frozen_pre_kd
```

When the denominator is below the configured floor, the relative value is `n/a`; the absolute values remain available. Baselines must be named in field names and column labels—never just `before` or `base`.

The pre-KD table is retained even after KD completes. Otherwise a global improvement can hide which blocks were weak at the end of block quantization.

The implementation retains the local objective-weighted snapshots in every immutable `BlockResult`. Model-level KD
adds a separate, versioned block-output probe to `GlobalTuningResult`: a bounded deterministic calibration slice is
run through the base model, the final frozen pre-KD model, and the final post-KD model, with each student compared to
the same base-model hidden outputs. The probe stores BF16 references in pageable host memory one sequence at a time
and accumulates MSE in FP32, so it does not reintroduce full-logit or full-model-overlap VRAM growth. Reports label
the local pre-KD value and the probe pre/post-KD values separately; only the two probe values form the named
`final_post_kd - final_frozen_pre_kd` comparison.

### Why this matters

Per-weight reconstruction error can look acceptable while the block's behavior is poor, and later blocks can sometimes compensate for earlier error. The final block comparison shows whether the complete local procedure—allocation, ADMM, outliers, scale fitting, layer tuning, and post-block refit—actually preserved the teacher behavior at that boundary.

It is therefore used for:

- finding the first block where error becomes material;
- distinguishing poor factorization from poor tuning recovery;
- selecting representative replay fixtures;
- deciding where more rank/outlier budget is worthwhile;
- comparing pre-KD and post-KD recovery;
- stopping obviously bad runs early;
- validating resume equivalence at block commits.

## 5. Preserve the modified llama.cpp implementation as a runtime reference

The modified tree at `D:\dev\research\llama.cpp\` is not merely an external competitor. It is a NanoQuant-capable implementation containing learned optimization work that should inform the rewrite.

The CUDA implementation at [nanoquant.cu](../../../llama.cpp/ggml/src/ggml-cuda/nanoquant.cu) currently demonstrates techniques including:

- decode-specific warp kernels;
- packed 32-bit sign-word loads;
- aligned vector loads of multiple sign words;
- lane-zero load plus warp broadcast;
- branchless sign application through the floating-point sign bit;
- sign-aware fused multiply-add in the first factor stage;
- multiple accumulators to hide dependency latency;
- specialized fast paths for contiguous F32 inputs/scales and explicit F16/BF16 handling;
- fused first-stage processing for multiple projections;
- separate optimization work for decode and prefill, reflected in benchmark/profile artifacts in the repository root.

The rewrite should study and, where licenses/interfaces permit, reuse or port the proven layout and execution ideas rather than starting kernel design from the current Python test runtime alone.

### Required reference captures

Before runtime redesign changes either codebase, record:

- llama.cpp source revision and dirty patch hash;
- NanoQuant CUDA source hash;
- conversion script and GGUF format version;
- exact model artifact hash;
- benchmark command, JSON output, prompt/decode sizes, repetitions, and hardware;
- Nsight traces or summarized kernel profiles for representative prefill and decode;
- output parity/quality checks against the research representation.

The reported approximately 400 tokens/second is a useful target observation, while repository benchmark files also represent intermediate measurements under different workloads and revisions. The shared benchmark protocol must identify which result is being compared rather than selecting an unlabeled maximum.

## 6. Required changes to the rewrite plan

These lessons modify the earlier clean-sheet plan in important ways:

- numbered experiment files are retained, not replaced wholesale;
- zero-argument runfiles are a supported first-class launcher alongside YAML and CLI commands;
- the architectural rule is “no duplicated orchestration in runfiles,” not “no runfiles”;
- block-final error tables are required report outputs and promotion signals;
- pre-KD and post-KD snapshots are separate durable artifacts;
- the modified llama.cpp NanoQuant runtime is a named source, correctness, packing, profiling, and performance reference;
- repository paths are documented as current local anchors while manifests use portable identities.
