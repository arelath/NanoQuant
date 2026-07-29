# Experiment 033: Gemma 3 1B covariance-refined D2 compression

## Status

**Prepared on 2026-07-29; complete run pending.**

- Model: `google/gemma-3-1b-it`
- Launcher:
  `experiments/033-covariance-refined-d2-compress-and-benchmark-gemma-3-1b-it.py`
- Baseline: Experiment 022

## Question

Does the same-format covariance-aware binary refinement promoted by Documents
43–49 improve the complete Experiment 022 D2 result after resident tuning,
global distillation, packing, GGUF export, and retained quality evaluation?

## Controlled change

Experiment 033 is structurally identical to Experiment 022 except for the
calibration objective that explicitly enables the resident refinement:

| Setting | Experiment 022 | Experiment 033 |
| --- | --- | --- |
| Objective kind | `diagonal` | **`dense_hessian`** |
| Covariance fit rows per block | none | **8,192** |
| Refinement | none | **32 left-coordinate steps, 16 right batches** |

The selected refinement settings are fixed production constants. They were
chosen before this run from the independent sample-size and depth screens,
not tuned against Experiment 033 quality.

All other recipe settings remain matched to Experiment 022: fused QKV,
0.6-shrunk diagonal Fisher importance, fresh same-campaign exact-unit KL
measurement, calibration-weighted measured rank responses, D2 allocation,
per-layer and factorized tuning, post-block refit, global distillation, export,
and the 64-by-128 retained WikiText protocol.

The refinement changes neither rank nor factor storage. It updates the binary
signs and three diagonal scales of the already accepted factorization before
factorized tuning. Fused QKV, O, gate, and up projections are eligible.
Gemma's 6,912-wide down projections remain unchanged because that dense solve
was outside the validated memory and numerical screen.

## Prior evidence

The final independent 26-block static screen at exactly 0.999472370 factor
BPW reported:

- held-out covariance error −24.06%;
- joint KL −12.85%;
- NLL −0.455772 nats/token;
- 104/104 refined groups and all 26 block aggregates improved;
- all three retained block-output comparisons improved;
- 20.74 seconds of refinement versus 684.23 seconds of ADMM.

Those results justify this complete run but do not predict its outcome.
Experiment 032 showed that a strong static functional result can reverse
after allocation, tuning, and distillation.

## Success criteria

- the fresh uniform control, exact-unit KL profile, and D2 rank-response
  profile complete under the covariance-refined recipe;
- all 26 candidate blocks and the complete transitive artifact graph pass
  strict validation;
- logical, packed, checkpoint, GGUF, and export-summary outputs complete;
- the retained WikiText protocol and token identity match Experiment 022;
- effective BPW does not exceed Experiment 022's 1.024494712;
- candidate perplexity improves on Experiment 022's 228.550618;
- task accuracy, ranks, block losses, runtime, peak memory, and artifact bytes
  are reported regardless of the primary quality outcome.

## Interpretation guard

The same-campaign uniform control is also covariance-refined, so the exact-unit
KL anchors compare projection removals at the candidate's operating point.
Measured rank responses remain the ordinary calibration-weighted response
model used by Experiment 022; covariance refinement is deliberately treated
as a post-factorization optimizer, not silently substituted for the D2
allocation objective.

No promotion decision will be made from block losses alone. The decisive gate
is protocol-matched retained quality after the complete compression lifecycle.
