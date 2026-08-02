# Experiment 047: Fixed Epoch-3 Capability Confirmation

## Status

Primary C4 confirmation passed. Artifact materialization and the conditional
deployment benchmarks remain pending. The exact checkpoint-arm evaluator was
committed in `29bc4b5` and did not change during the gate.

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
