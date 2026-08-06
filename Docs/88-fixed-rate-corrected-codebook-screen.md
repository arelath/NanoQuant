# Fixed-Rate Corrected-Codebook Screen

**Date:** 2026-08-05

**Status:** completed; lower-rate arms rejected, equal-bit one-correction arm
functionally weaker than the retained two-correction format

## Question

The retained mixed dominant-factor representation stores each coded 32-sign
word as a 10-bit codebook index plus a 9-bit pair of corrected positions. Its
19-bit payload is the equal-bit quality optimum in the existing format sweep,
but that sweep normally reinvests saved storage in rank. This screen instead
holds block, rank, and factor geometry fixed to measure the direct quality cost
of reducing the coded-word payload.

## Protocol

- Model: pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Matrix: block 12 `mlp.down_proj`, shape 1,152 by 6,912.
- Objective: retained corrected-CCE Fisher state with 0.6 shrinkage.
- Free-word control: rank 970.
- Codebook arms: rank 1,344, free left factor, and a variable free right-factor
  prefix followed by corrected coded rows.
- ADMM: 800 outer by five inner iterations, two-pass scale fit, seed zero.
- Every arm receives the retained representation-masked control-then-tabu
  binary-factor search.
- The equal-bit one-correction endpoint receives a matched 48-sequence,
  24,528-target WikiText-2 splice comparison.

The retained 19-bit result is reused. All lower-rate JSON results are fresh and
use the same implementation and protocol.

## Fixed-rate reconstruction curve

| Coded format | Free V rows | Actual BPW | Saving vs retained | Weighted NRMSE | Change vs free words |
| --- | ---: | ---: | ---: | ---: | ---: |
| k10 + two corrections, 19 bits | 256 | 1.000452 | - | 0.527928 | **-0.852%** |
| k10 + one correction, 15 bits | 256 | 0.882397 | 11.800% | 0.565603 | +6.223% |
| k9 + one correction, 14 bits | 256 | 0.850825 | 14.956% | 0.576992 | +8.363% |
| k10 + one correction, 15 bits | 384 | 0.941424 | 5.900% | 0.546338 | +2.605% |
| k10 + one correction, 15 bits | 448 | 0.970938 | 2.950% | 0.536774 | +0.809% |
| k10 + one correction, 15 bits | 512 | 1.000452 | 0.000% | **0.527421** | **-0.947%** |

The lower-rate curve is smooth: each tranche of fully free anchors recovers
quality, but every arm that retains any material storage saving remains worse
than the rank-970 free-word control. At a 2.95% saving the candidate still
regresses by 0.81% NRMSE. At equal bits, reallocating capacity from the second
correction to 256 additional free rows slightly improves the diagonal
reconstruction objective over the retained format.

Masked binary-factor search does not change this conclusion. It improves the
lower-rate candidate NRMSE by only 0.076% to 0.088%, far less than the
representational loss from removing the second correction.

## Equal-bit functional gate

The equal-bit k10/one-correction arm with 512 free rows was the only new arm to
beat free words in reconstruction, so it alone advanced to the paired
48-sequence splice gate.

| Arm | KL nats/token | Change vs free words | Paired 95% interval |
| --- | ---: | ---: | ---: |
| Free words | 0.045156 | - | - |
| Retained 19-bit, free 256 | 0.043403 | -3.88% | [-0.003826, +0.000523] |
| 15-bit, free 512 | 0.044180 | -2.16% | [-0.002769, +0.000796] |

The one-correction point estimate improves over free words but its interval
crosses zero, and it gives back 44.3% of the retained format's KL improvement.
Its small reconstruction lead over the 19-bit format does not transfer to the
held-out model-level signal.

## Decision

Reject the tested 14- and 15-bit fixed-rate arms as storage candidates. Do not
change the mixed overlay, resident algorithm, GGUF, or runtime codec.

The result also rejects a simple use of the new binary-factor search as a way
to compensate for fewer payload bits: its improvement is two orders of
magnitude smaller than the lower-rate reconstruction regression. The equal-bit
one-correction endpoint is a useful negative transfer result, not a promotion
candidate.

A future variable-rate codec needs a genuinely per-word rate-distortion
mechanism rather than a uniform correction-count reduction. It should first
measure zero/one/two-correction and raw-escape choices on disjoint operator
data, charge the mode stream exactly, and advance only if a material lower-rate
point beats the free-word reconstruction control before splice evaluation.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip1-k10-rank1344-free256-binary-search-fixed-rate.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip1-k9-rank1344-free256-binary-search-fixed-rate.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip1-k10-rank1344-free384-binary-search-fixed-rate.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip1-k10-rank1344-free448-binary-search-fixed-rate.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip1-k10-rank1344-free512-binary-search-fixed-rate.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-mixed-flip1-k10-r1344-free512-binary-search-splice-48.json`
