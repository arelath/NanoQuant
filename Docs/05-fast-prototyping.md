# Fast Research and Prototyping

## 1. Objective

A full quantization run taking hours must be the final confirmation step, not the first time an idea is exercised. The rewrite creates a sequence of increasingly expensive experiments, each answering a narrower question with cached, replayable inputs.

## 2. Feedback ladder

| Level | Input | Typical output | Question answered |
| --- | --- | --- | --- |
| Unit | generated tiny tensors | invariants and exact values | Is the local implementation correct? |
| Layer replay | captured weight, statistics, plan | reconstruction metrics and timing | Does the algorithm improve the target layer? |
| Block replay | captured block and activations | block-output loss and timing | Does the local gain survive surrounding operations? |
| Pipeline smoke | synthetic/tiny model | complete artifact and smoke metrics | Do all stages integrate? |
| Representative subset | selected blocks/layers/data | directional quality/cost evidence | Is a full run justified? |
| Full run | complete target recipe | candidate model and full evaluation | Does the change improve the deliverable? |

Every level produces a normal run manifest and comparable metrics. A replay is not an untracked debugging script.

## 3. Numbered zero-argument experiment workflow

Numbered runfiles remain the normal human-facing chronology for new promoted experiments. The active chronology was
reset on 2026-07-15 after the legacy lessons were absorbed into shared services, tests, recipes, and retained
evidence. The `experiments/` directory therefore contains only the current Experiment 001:

```text
python experiments/001-compress-gemma-3-1b-it.py
```

New experiments should compose the shared workflow modules under `src/nanoquant`; do not restore or copy the old
experiment scripts. Experiment 001 is a thin example: it declares canonical configuration and material outputs,
then delegates compression, validated GGUF export, and BF16-versus-NanoQuant evaluation to the framework. It takes
no experiment-defining parameters. Logging, checkpoint, model loading, evaluation, and resume mechanics stay in
shared code.

After an experiment produces a committed run, changing its hypothesis or semantic settings means creating the next
number and expressing the delta in canonical configuration. Resume does not create a new
experiment number; a changed recipe does. Historical experiments remain evidence and lessons, not templates or a
migration backlog. See [Lessons Carried Forward](12-lessons-carried-forward.md#2-preserve-numbered-experiment-files)
and [ADR-0005](adr/0005-numbered-zero-argument-runfiles.md).

## 4. Captured replay fixtures

An expensive run can opt into capturing a bounded diagnostic fixture:

```text
fixture/
  fixture.json
  source-weight.safetensors
  calibration-stats.safetensors
  objective.safetensors
  block-inputs.safetensors
  target-outputs.safetensors
  source-block.safetensors
  expected.json
```

Fixture metadata includes:

- source model and tensor hash;
- adapter and layer identity;
- shape, dtype, and layout;
- calibration and objective identities;
- sample indices or anonymized sample hashes;
- capture code and schema versions;
- baseline metrics;
- whether fixture data may be shared.

Fixtures must not silently include private prompt text. Token tensors and activations inherit dataset handling rules.

Commands:

```text
nanoquant capture-layer runs/<id> --block 12 --layer self_attn.v_proj
nanoquant replay-layer fixtures/<id> --recipe recipes/change.yaml
nanoquant replay-block fixtures/<id> --compare runs/<baseline>
```

## 5. Stage caching

Calibration is often more expensive than the policy being changed. Stage-specific semantic hashes allow common experiments to reuse:

- tokenized calibration samples;
- first-block inputs;
- calibration statistics;
- Hessian/objective artifacts;
- source block fixtures;
- teacher outputs;
- quantization plans;
- packed reference states;
- evaluator task preprocessing.

Cache reuse is visible in the plan and run report. Each hit states the producing run, content hash, validation result, and time saved.

## 6. Experiment specification

Every experiment should declare:

- the problem observed;
- one primary hypothesis;
- the baseline run;
- the exact independent variable;
- metrics expected to move;
- regression guards;
- the cheapest experiment capable of falsifying the hypothesis;
- promotion criteria for the next tier;
- maximum compute/time budget;
- conclusion after completion.

Example:

```yaml
intent:
  experiment_number: 20
  purpose: Test low-rank-plus-diagonal Hessian approximation on difficult v_proj layers.
  hypothesis: Rank-256 covariance correction reduces block residual loss by at least 5% relative without more than 20% factorization-time overhead.
  baseline_run: run_01J...
  promotion:
    metric: block.residual_loss
    relative_improvement_min: 0.05
    guards:
      - metric: factorization.wall_seconds
        relative_regression_max: 0.20
  budget:
    max_wall_minutes: 30
```

The run report fills in whether promotion criteria were met.

## 7. Representative case selection

Fast subsets must be representative rather than merely convenient. The evaluation registry maintains cases such as:

- early, middle, and late transformer blocks;
- attention q/k/v/o and MLP gate/up/down projections;
- smallest, median, and largest matrix shapes;
- historically easy and difficult reconstruction layers;
- layers with and without outliers;
- models from each adapter family;
- short and long context activation captures.

Case selection is versioned and uses baseline evidence. Changing the representative set changes the benchmark identity.

## 8. Debug and reduced recipes

Named profiles make speed-versus-fidelity tradeoffs explicit:

```text
debug:          tiny tensors, minimal iterations, no quality claim
smoke:          tiny model, all stages, catches integration failures
quick-research: selected real layers/blocks, reduced samples/iterations
representative: fixed subset with normal algorithm settings where possible
full:           complete production recipe
```

A report carries a prominent evidence tier. Results from `debug` or `smoke` cannot be displayed as model-quality results.

Reduced ADMM iterations can test code paths and gross trends but may change method rankings. Where possible, a representative replay should use production factorizer settings on fewer layers rather than weakened settings on the whole model.

## 9. Comparison-first workflow

Researchers should not inspect unrelated logs side by side manually. `nanoquant compare` aligns candidate and baseline by semantic layer identity and produces:

- changed configuration fields;
- reused versus recomputed artifacts;
- per-layer reconstruction deltas;
- block-output loss deltas;
- effective BPW and storage deltas;
- calibration/factorization/tuning timing deltas;
- memory and I/O deltas;
- warnings unique to either run;
- evaluation deltas with uncertainty;
- promotion-gate outcome.

Large per-layer tables sort by the most actionable regression and link back to the exact attempt events.

## 10. Profiling workflow

Performance changes follow the same ladder:

1. microbenchmark the affected operation;
2. profile a captured layer/block with synchronized stage timers;
3. measure a representative subset;
4. confirm end-to-end impact.

Each profile records warm-up, repetitions, synchronization points, tensor shapes, dtypes, allocator state, and hardware. CUDA traces are optional artifacts linked from the run rather than embedded in console logs.

## 11. Reproducibility modes

Two modes are useful:

- **deterministic research mode:** deterministic algorithms where available, logical seed derivation, fixed selections, and strict cache identities;
- **performance mode:** permits documented nondeterministic kernels and reports repeated-run variability.

The mode is part of the recipe. A candidate cannot claim bit-for-bit reproducibility if performance mode was used.

## 12. Preventing prototype code from becoming architecture

Exploration may begin in notebooks or scratch programs, but promotion requires:

- strategy implementation behind an existing typed interface;
- unit and fixture replay tests;
- serializable versioned configuration;
- registered metrics and events;
- no model-family branching outside adapters;
- a comparison report against the baseline;
- promotion into a thin numbered zero-argument runfile, with the scratch entry point removed or archived.

This keeps fast experimentation and chronological numbered files while preventing those files from reimplementing orchestration.
