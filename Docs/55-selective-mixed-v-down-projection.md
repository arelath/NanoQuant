# Selective Mixed-V Down-Projection Screen

**Date:** 2026-07-30
**Status:** full reconstruction screen passed; uniform application rejected;
six-block selective policy passed disjoint functional confirmation

**Follow-up:** [Mixed-V Seed and Runtime Acceptance](56-mixed-v-seed-runtime-acceptance.md)
tightens the rule to 0.95%, selects five seed-stable blocks, confirms quality
for three factorization seeds, and accepts load-time packed predecode as the
execution policy.

## Question

[Mixed Free/Coded V Basis](54-mixed-free-coded-v-basis.md) found that a
rank-1,344 factorization with 256 free V rows and 1,088 corrected-codebook
rows beat the rank-970 free-word control on representative blocks.

This experiment asks:

1. Does the reconstruction gain generalize to all 26 Gemma `down_proj`
   matrices?
2. If every block improves locally, should the mixed representation be
   applied everywhere?
3. Can a reconstruction-only selection rule identify a smaller set that
   improves held-out model behavior?

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Matrices: all 26 `mlp.down_proj`, each 1,152 x 6,912
- Importance: retained 256-sample corrected-CCE Fisher state, shrinkage 0.6
- Control: free U and V sign words, rank 970
- Candidate: rank 1,344; U fully free; first 256 V rows free; remaining
  1,088 V rows use a 10-bit codebook and two corrections encoded in 9 bits
- ADMM: 800 outer iterations and five inner iterations
- Scale fit: two alternating passes
- Rank alignment: 32
- Functional data: pinned WikiText-2 test tokens in independent length-512
  windows

Each block is fitted independently with the same deterministic logical seed.
The candidate costs 7,966,224 bits per matrix versus 7,966,624 bits for the
control, so every comparison is equal-budget with 400 bits left unused.

## Full reconstruction screen

The candidate improves Fisher-weighted RMSE in all 26 blocks:

| Block | Free-word RMSE | Mixed RMSE | Change |
| ---: | ---: | ---: | ---: |
| 0 | 0.519717 | 0.514197 | -1.062% |
| 1 | 0.530412 | 0.526488 | -0.740% |
| 2 | 0.509316 | 0.504917 | -0.864% |
| 3 | 0.535451 | 0.531747 | -0.692% |
| 4 | 0.522870 | 0.520131 | -0.524% |
| 5 | 0.543558 | 0.538989 | -0.841% |
| 6 | 0.532845 | 0.529768 | -0.578% |
| 7 | 0.524192 | 0.520133 | -0.774% |
| 8 | 0.521327 | 0.517190 | -0.794% |
| 9 | 0.519056 | 0.515306 | -0.722% |
| 10 | 0.499070 | 0.494056 | -1.005% |
| 11 | 0.529453 | 0.522513 | -1.311% |
| 12 | 0.533293 | 0.528163 | -0.962% |
| 13 | 0.531412 | 0.528679 | -0.514% |
| 14 | 0.527220 | 0.524506 | -0.515% |
| 15 | 0.528314 | 0.526051 | -0.428% |
| 16 | 0.512346 | 0.507728 | -0.901% |
| 17 | 0.535067 | 0.531405 | -0.684% |
| 18 | 0.534286 | 0.531127 | -0.591% |
| 19 | 0.525558 | 0.523193 | -0.450% |
| 20 | 0.520284 | 0.518911 | -0.264% |
| 21 | 0.505091 | 0.503836 | -0.249% |
| 22 | 0.492308 | 0.491167 | -0.232% |
| 23 | 0.499696 | 0.498323 | -0.275% |
| 24 | 0.505094 | 0.503029 | -0.409% |
| 25 | 0.516945 | 0.511357 | -1.081% |

Summary:

- 26/26 improve weighted RMSE;
- mean improvement: 0.672%;
- median improvement: 0.688%;
- best: block 11 at 1.311%;
- weakest: block 22 at 0.232%;
- aggregate weighted RMSE: 0.521062 to 0.517445, a 0.694% reduction;
- aggregate weighted error energy falls 1.384%;
- raw RMSE improves in 22/26 blocks.

Every block uses all 1,024 codebook entries. Empirical index entropy ranges
from 9.9935 to 9.9946 bits out of 10, so the result is not driven by a
collapsed or mostly unused table.

## Uniform 26-block application fails

Despite universal local improvement, replacing all 26 controls is not better
functionally on the first 24-sequence held-out window:

| Arm | KL nats/token |
| --- | ---: |
| Rank-970 free words | 1.450378 |
| Mixed V in all 26 blocks | 1.453150 |

The mixed form changes KL by **+0.19%**. The paired 95% delta interval is
`[-0.017414, +0.023446]`, spanning zero.

This is an important negative result: independent Fisher-weighted matrix
improvements do not compose monotonically through the complete decoder.
Applying a locally better representation everywhere is therefore rejected.

## Reconstruction-selected policies

Selection uses only the calibration-side weighted RMSE gain:

`gain = 1 - mixed_weighted_rmse / free_word_weighted_rmse`

No held-out KL enters the block membership rule. Three coarse thresholds were
screened against the same teacher cache:

| Minimum local gain | Blocks | Free-word KL | Mixed KL | Change | Paired 95% delta interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5% | 19 | 1.171445 | 1.146751 | **-2.11%** | [-0.045176, -0.003802] |
| 0.7% | 12 | 0.698751 | 0.688437 | -1.48% | [-0.032740, +0.012238] |
| 0.8% | 8 | 0.535385 | 0.504408 | **-5.79%** | [-0.052647, -0.008990] |

The 0.5% and 0.8% policies pass. The non-monotonic 0.7% point shows that
block interactions matter; count alone does not predict quality.

### Fine threshold screen

The successful 0.8% boundary was narrowed on the same first 24 held-out
sequences:

| Minimum gain | Blocks | Selected blocks | KL change | Paired 95% delta interval |
| ---: | ---: | --- | ---: | ---: |
| 0.85% | 7 | 0, 2, 10, 11, 12, 16, 25 | -8.05% | [-0.058044, -0.019736] |
| 0.90% | 6 | 0, 10, 11, 12, 16, 25 | **-10.69%** | [-0.067949, -0.029347] |
| 0.95% | 5 | 0, 10, 11, 12, 25 | -10.38% | [-0.058898, -0.020140] |
| 1.00% | 4 | 0, 10, 11, 25 | -10.24% | [-0.052892, -0.017000] |

All four pass, and the 0.90% threshold is best. On this screen it changes KL
from 0.451072 to 0.402836, wins 22/24 paired sequences, and reduces KL by
10.69%.

Because this threshold was selected after examining the first held-out
window, that result is treated as tuning evidence rather than confirmation.

## Disjoint confirmation

The selected six-block rule was rerun on WikiText windows 24 through 47,
which do not overlap the threshold screen:

| Arm | KL nats/token |
| --- | ---: |
| Rank-970 free words in blocks 0, 10, 11, 12, 16, 25 | 0.462460 |
| Mixed V in blocks 0, 10, 11, 12, 16, 25 | 0.409556 |

The mixed policy:

- reduces KL by **11.44%**;
- has paired 95% delta interval `[-0.068443, -0.036520]`;
- wins 21/24 paired sequences.

Combining the two disjoint inventories gives 48 sequences and 24,528
next-token targets:

| Arm | Combined KL nats/token |
| --- | ---: |
| Free words | 0.456766 |
| Six-block mixed V | 0.406196 |

The combined reduction is **11.07%**, with paired interval
`[-0.063377, -0.038139]` and 43/48 sequence wins.

## Compute and storage consequence

The representation remains equal-bit. Across the six selected matrices it
uses 2,400 fewer bits than the free-word controls, which is negligible at
model scale and is not the reason to select it.

Compute is materially better than uniform mixed-V application:

| Policy | Average `down_proj` factor work / dense |
| --- | ---: |
| Rank-970 control in all blocks | 0.982x |
| Rank-1,344 mixed V in all blocks | 1.361x |
| Mixed V in selected six; control in remaining 20 | 1.070x |

Selective use raises average `down_proj` factor work by 8.9% relative to the
control, rather than 38.6% for uniform application. Actual decode overhead
still needs measurement because table lookup and correction application are
not represented by multiply-add count.

## Implementation

`tools/probe_corrected_codebook_splice.py` now supports:

- nested reconstruction-gain thresholds evaluated after one factorization;
- reuse of one immutable teacher-logit cache across all threshold arms;
- explicit WikiText window offsets for disjoint confirmation;
- output schema version 4 with per-selection block inventories, KL metrics,
  and paired intervals.

This remains analysis-only. No persisted compression artifact, GGUF schema,
or runtime contract has changed.

## Decision

Retain the mixed-V format as a selective `down_proj` candidate with the
predeclared rule:

`use mixed V when local weighted RMSE improves by at least 0.9%`

On this model and calibration state, that selects blocks
`0, 10, 11, 12, 16, 25`.

The follow-up seed screen supersedes this exploratory inventory: block 16 is
borderline under seeds 1 and 2, so the accepted rule is tightened to 0.95%
and selects `0, 10, 11, 12, 25`.

The rule passes:

- all-block reconstruction screening;
- a first 24-sequence threshold screen;
- a disjoint 24-sequence confirmation;
- a combined 48-sequence paired analysis.

Do not apply the format uniformly, and do not yet change GGUF or runtime.
The next gates are:

1. verify the selected block inventory under additional factorization or
   calibration seeds;
2. benchmark decoded execution for the six-block hybrid policy;
3. test whether the same one-sided mixed-basis principle transfers to other
   matrix shapes using their own equal-bit allocation and selection rule;
4. define a packed schema only after the quality and throughput gains both
   survive.
