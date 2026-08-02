# Experiment 047: Fixed Epoch-3 Capability Confirmation

## Status

Completed. All four gates passed for the fixed Experiment 044 trajectory. The
epoch-3 correction is a validated retained candidate, but it is not a
transferable production recipe. The exact checkpoint-arm evaluator was
committed in `29bc4b5` and did not change during the primary gate.

## Question

Experiment 046 produced exploratory evidence that a 96-step one-sided
correction may repair part of Experiment 044's remaining capability loss. Its
selection sequence was procedurally invalid, so it cannot promote a model.
Experiment 047 asks the narrower confirmation question: does that one fixed
checkpoint improve capability on gates that played no role in choosing it?

This experiment confirms only the retained Experiment 044 trajectory. It does
not establish that epoch 3, 96 steps, weight 2.0, or the 0.8 ratio transfers to
a fresh factorization.

## Frozen arms

- baseline: Experiment 044 tail-aware epoch-8 global tuning,
  `sha256-eb4a8f736d51b94b8b8b561d3dc2a0a48970be9f359b508e8c0d4140c46233cf`,
  observed at exactly 256 optimizer steps;
- candidate: Experiment 046 correction epoch 3, exactly 96 correction steps,
  artifact
  `sha256-d885f823e0651314d78f1c4c7b98edd66470344bb1957f5e27aa959ac0624a8d`;
- candidate checkpoint directory:
  `evidence/046/experiment046-weight2-ratio0p8-correction`;
- frozen factorization:
  `evidence/044/044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it`;
- no further optimization, checkpoint selection, final-norm fold, block refit,
  coefficient change, or calibration is permitted.

Selected top-64 mass is diagnostic only. NLL, full KL, and task behavior are
the capability criteria.

## Gate order

The gates open in this order. Failure stops the experiment; later gates remain
unopened.

1. **Untouched C4 confirmation.** Use C4 validation offset 296, 48x512, token
   hash
   `sha256:98ab9ec7c492025b6ae03d8706a96c80262d4504b726bd9df03b58a734629402`.
   Relative to the Experiment 044 baseline, the paired 95% upper bounds must
   be below zero for both NLL and full KL, and both point deltas must be
   negative. This slice is reserved as
   `experiment047-c4-validation296-48x512` and is retired immediately after
   its first model-dependent result.
2. **Artifact materialization.** After the independent C4 gate passes,
   materialize epoch 3 as an immutable global-tuning artifact. Reload it and
   verify exact selected tensor equality, the same frozen factor graph,
   unchanged represented factor bytes, unchanged rank, and unchanged
   effective BPW. Materialization is representation validation, not candidate
   promotion, and may not alter any value selected by the gate.
3. **Canonical WikiText/task-200 benchmark.** Run the retained factorized
   quality protocol at 64x128 WikiText test windows and the six canonical tasks
   at 200 examples. This reused protocol is a stable deployment benchmark, not
   fresh selection data. The candidate must improve on Experiment 044
   perplexity `180.3641328653` and must be no worse than the Experiment 042 2%
   ceiling `175.3083146187`. Task-200 values are reported but do not bind
   because the final task gate is powered at 1,000 examples.
4. **Full task guardrail.** Evaluate the six canonical tasks at 1,000 examples
   without changing the candidate. Compare paired item outcomes with both
   Experiment 044 and the Experiment 042 packed candidate. A regression is
   established only when the task-stratified paired 95% upper bound on the
   candidate-minus-reference mean accuracy is below zero. Any established
   regression rejects the candidate.

## Interpretation

Passing all four gates establishes that the fixed checkpoint repairs this one
retained tail-aware trajectory. It justifies designing an adaptive correction
policy and testing that policy in a fresh complete compression campaign. It
does not justify inserting the fixed checkpoint number or budget into the
general compression recipe.

Failure rejects the fixed epoch-3 hypothesis. It does not reject one-sided
correction in general.

## Primary-gate result

The committed source identity was `28c5543`. The evaluator observed the exact
256-step baseline, 96-step candidate, common frozen identity, 48 aligned
sequences, and reserved token hash. The result was:

| Arm | C4 NLL | Full KL |
| --- | ---: | ---: |
| Experiment 044 tail-aware baseline | 4.912287 | 1.006619 |
| Correction epoch 3 | **4.872377** | **0.984313** |

Candidate-minus-baseline paired intervals both pass:

- NLL: `-0.039910`, 95% interval `[-0.047543, -0.032233]`;
- full KL: `-0.022307`, 95% interval `[-0.027980, -0.016931]`.

An overlapping continuation briefly started the later canonical benchmark
before this gate. It was stopped before producing an output artifact and was
not used to select, alter, or judge the candidate. The C4 worker then ran alone
under the CUDA lease. The reserved slice is permanently retired with evidence
at `evidence/047/experiment047-c4-validation296-48x512.json`.

## Materialization result

Epoch 3 materialized as global-tuning artifact
`sha256-9eae64909f2533e35655ecbcfa341eb68415684f36b0ba7673ba8d109c52c0e5`
under `evidence/047/experiment047-correction3-derived-run`. A hash-verifying
reload found all 677 selected tensors, zero missing tensors, zero mismatches,
and maximum absolute difference 0. The protocol hash, token hash, and 96-step
receipt match the durable correction checkpoint.

Strict resident validation audited 713 artifacts and all 26 blocks. Rank sum
remains 111,744 and effective BPW remains `1.024417665448784`. The correction
adds no tensors, represented factor bytes, or inference operations.

## Deployment benchmarks

The canonical 64x128 WikiText result is PPL `166.451651`, passing both binding
conditions:

- Experiment 044: `180.364133` (candidate is 7.71% lower);
- Experiment 042: `171.870897` (candidate is 3.15% lower and comfortably
  inside the 2% non-inferiority ceiling of `175.308315`).

The non-binding task-200 mean is `0.4600`, between Experiment 044's `0.4592`
and Experiment 042's `0.4617`.

At 1,000 examples per task, the candidate scores are:

| PIQA | ARC-E | ARC-C | HellaSwag | WinoGrande | BoolQ | Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.588 | 0.378 | 0.216 | 0.400 | 0.503 | 0.624 | **0.4515** |

Paired task-stratified comparisons establish no regression:

- versus Experiment 044: delta `+0.002167`, 95% interval
  `[-0.002667, +0.007000]`;
- versus Experiment 042: delta `-0.007000`, 95% interval
  `[-0.017667, +0.003667]`.

The benchmark loader applied the durable checkpoint directly. The exact
677-tensor reload audit above proves that it is behaviorally the same state as
the derived global-tuning artifact; no fold or other transformation was
applied.

Evidence:

- `evidence/047/experiment047-correction3-strict-validation.json`;
- `evidence/047/experiment047-correction3-canonical-quality.json`;
- `evidence/047/experiment047-correction3-tasklimit1000-quality.json`;
- `evidence/047/experiment047-correction3-vs-experiment044-tasklimit1000-paired.json`;
- `evidence/047/experiment047-correction3-vs-experiment042-tasklimit1000-paired.json`.

## Decision

The fixed epoch-3 checkpoint repairs the retained Experiment 044 candidate
enough to beat both Experiments 044 and 042 on factorized WikiText PPL without
an established task regression. This is evidence that a capability-oriented
post-KD correction can complement tail-aware primary KD.

It does not justify transplanting 96 steps, weight 2.0, or ratio 0.8 into a
fresh run. The next work is to define an adaptive stopping policy whose monitor
and uncertainty are committed before training, then test that policy in one
fresh complete compression campaign with normal packed reload, GGUF export,
C4, WikiText, and task gates.
