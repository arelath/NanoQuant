# Fisher Importance Shrinkage Functional Probe

**Date:** 2026-07-29
**Status:** completed positive result; promote raw Fisher to a complete resident canary
**Model:** pinned `google/gemma-3-1b-it` revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`

## Question

The base compression recipe linearly shrinks every diagonal Fisher importance
vector 60% toward its mean:

`I_s = (1 - s) I + s mean(I)`.

This is a regularization choice, not a storage-format requirement. It is free at
runtime because the diagonal weighting is absorbed into the fitted scales. The
question is whether `s=0.6` protects against calibration overfitting or discards
useful channel sensitivity.

The answer on the pinned Gemma workload is that it discards useful sensitivity.
Raw Fisher (`s=0`) reduces held-out whole-model KL by **13.13%** at identical
physical BPW when all 26 blocks are reconstructed.

## Reproducible harness

`tools/probe_importance_shrinkage.py` reconstructs complete decoder blocks with:

- the adopted fused Q/K/V factorization plus separate O, gate, up, and down
  groups;
- corrected CCE Fisher state from
  `evidence/m4/gemma-cce-fisher-state`;
- identical 1.0-BPW targets, rank calculation, scale storage, ADMM settings,
  scale-fit passes, and seed for every shrinkage arm;
- 400 outer ADMM iterations, 5 inner iterations, cubic penalty scheduling,
  regularization 0.03, and two alternating scale-fit passes;
- held-out WikiText evaluation on 12 sequences of 512 tokens;
- paired 10,000-resample sequence bootstrap intervals;
- isolated complete-block output measurements on four held-out sequences.

The harness retains dense reconstructions only in memory. Its JSON is
analysis evidence, not a compression or runtime artifact.

The primary ignored results are:

- `evidence/m4/importance-shrinkage-probe/blocks-0-12-24.json`;
- `evidence/m4/importance-shrinkage-probe/uniform-control-corrected.json`;
- `evidence/m4/importance-shrinkage-probe/full-0-vs-0.6.json`.

## Representative sweep

The first screen reconstructed every projection in blocks 0, 12, and 24. All
arms use exactly 0.999472 actual BPW.

| Shrinkage | Joint KL vs teacher | NLL | Weighted-objective RMSE | Original-space RMSE |
|---:|---:|---:|---:|---:|
| **0.0 (raw)** | **0.350002** | **4.034924** | **0.300126** | 0.618977 |
| 0.3 | 0.391503 | 4.044267 | 0.412953 | 0.582159 |
| 0.6 (recipe) | 0.494634 | 4.085792 | 0.487333 | 0.563146 |
| 0.8 | 0.685044 | 4.183160 | 0.517787 | 0.554479 |
| 0.9 | 0.938034 | 4.417080 | 0.526803 | **0.550990** |

Raw Fisher improves KL by 0.144632 nats/token, or **29.24%**, relative to 0.6.
The paired 95% interval is `[-0.168410, -0.122772]`. Shrinkage 0.3 also beats
0.6, but it leaves 20.85% relative KL improvement on the table versus the
recipe compared with raw Fisher. Both 0.8 and 0.9 fail strongly.

The isolated complete-block output metric gives a more nuanced but compatible
result:

| Block | Raw Fisher | Shrinkage 0.3 | Shrinkage 0.6 | Raw vs 0.6 |
|---:|---:|---:|---:|---:|
| 0 | **0.191388** | 0.202783 | 0.257811 | **-25.76%** |
| 12 | 0.063126 | **0.058447** | 0.059446 | +6.19% |
| 24 | 0.067969 | **0.067748** | 0.069747 | **-2.55%** |

An interior value can be locally preferable, but the paired block-splice KL
still favors raw Fisher in all three blocks: -40.23%, -8.59%, and -9.72%,
each with a 95% interval below zero. The end-to-end selection metric therefore
justifies promoting the raw arm.

## Corrected uniform control

The initial sweep exposed an endpoint bug in `shrink_importance`: the condition
applied shrinkage only for `0 < s < 1`, so `s=1` accidentally returned the raw
vector and duplicated the `s=0` result. That arm is invalid and must not be
interpreted as evidence about uniform weighting.

The implementation now honors the documented closed interval. At `s=1`, every
entry becomes the vector mean. The resident algorithm version is incremented
from 48 to 49 so commits created under the old endpoint semantics cannot be
adopted silently.

The corrected uniform control is decisive:

| Shrinkage | Joint KL | Weighted-objective RMSE | Original-space RMSE |
|---:|---:|---:|---:|
| 0.6 | 0.494634 | 0.487333 | 0.563146 |
| 1.0 (uniform) | 2.565326 | 0.529995 | **0.548356** |
| Uniform relative KL change | **+418.63%** |  |  |

The paired 95% KL-delta interval is `[+1.956944, +2.197412]` nats/token.
Isolated output RMSE also regresses in every representative block: from
0.257811 to 0.545317 in block 0, 0.059446 to 0.091296 in block 12, and
0.069747 to 0.103967 in block 24.

This control rules out the explanation that unweighted fitting is beneficial.
The raw, highly nonuniform Fisher vector is the winning signal.

## Full 26-block gate

The promotion gate reconstructs all seven projections in all 26 decoder
blocks. Both arms store 697,393,632 bits for 697,761,792 source weights:
**0.999472 actual BPW**.

| Arm | KL vs BF16 teacher | NLL | Perplexity | Weighted RMSE | Original RMSE |
|---|---:|---:|---:|---:|---:|
| Raw Fisher | **3.770877** | **6.841914** | **936.28** | **0.349669** | 0.609994 |
| Shrinkage 0.6 | 4.340940 | 7.273607 | 1441.74 | 0.486337 | **0.559164** |
| Raw relative/delta | **-13.13% KL** | -0.431694 | -505.46 | -28.10% | +9.09% |

The raw-minus-0.6 paired KL delta is **-0.570062 nats/token**, with a 95%
interval of `[-0.684003, -0.438269]`. The interval excludes zero comfortably.

Absolute perplexity is intentionally poor because this is a static
analysis-only 1-BPW reconstruction without the complete resident allocation,
outlier, tuning, refit, and export lifecycle. The valid conclusion is the
controlled arm comparison, not that this probe is a releasable model.

The divergence between original-space RMSE and every functional metric is
important. Shrinkage 0.6 spends error more evenly and therefore wins ordinary
Frobenius RMSE. Raw Fisher accepts more error in insensitive coordinates,
reduces the calibrated objective by 28.10%, and produces the lower held-out
KL. Selecting shrinkage by unweighted matrix error would choose the wrong arm.

## Decision

- Promote `calibration.shrinkage=0.0` to the next complete pinned-Gemma
  resident canary.
- Do not change the cross-model base-recipe default solely from this one-model
  analysis probe. A complete resident run must confirm that allocation,
  outlier selection, tuning, post-block refit, artifacts, and final retained
  WikiText quality preserve the gain.
- Reject shrinkage 0.8, 0.9, and uniform weighting on this workload.
- Keep 0.3 as a robustness fallback for another model family, not as the Gemma
  selection.
- Use held-out paired KL or operator/block outputs for future weighting
  selection. Original-space RMSE is not an adequate gate.

## Next gate

Run the exact complete compression workflow with only the shrinkage changed
from 0.6 to 0.0. Validate the resident journal and artifact graph, assemble the
frozen model, and evaluate the retained WikiText protocol. Compare final
quality, allocation, ranks, outlier choices, tuning loss, BPW, memory, and wall
time against the current 0.6 evidence before changing a shared recipe default.
