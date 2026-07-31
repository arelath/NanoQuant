# Mixed Dominant-Factor Projection Screen

**Date:** 2026-07-30

**Status:** MLP gate/up transfer demonstrated for one block; attention
projections and broad cross-block application rejected

## Question

[Mixed-V Seed and Runtime Acceptance](56-mixed-v-seed-runtime-acceptance.md)
accepted a compact mixed basis for selected `mlp.down_proj` matrices. This
follow-up asks whether spending additional factorization and assignment
compute can buy the same compressibility in other projection families.

The principle is orientation-independent: transpose tall matrices before
fitting so that the storage-dominant factor is always the coded right factor.
For `gate_proj` and `up_proj`, this coded factor becomes the left factor after
the reconstruction is transposed back. It is therefore a mixed-U
representation in the source matrix orientation, despite using the same
analysis solver as mixed V.

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Importance: retained 256-sample corrected-CCE Fisher state, shrinkage 0.6
- Control: free U and V sign words at the largest aligned rank near 1 BPW
- Candidate: a free prefix in the storage-dominant factor, followed by
  corrected sign-codebook words
- Primary code: 10-bit table index plus two corrected positions in 9 bits
- Primary MLP allocation: rank 1,344 with 256 free dominant-factor rows
  versus the rank-970 control
- Broad screen: block 12, 400 outer iterations
- Confirmation: 800 outer iterations, assignment shortlist sizes 16 and 64,
  seeds 0 through 2, and representative blocks 0, 12, and 24
- Functional data: two disjoint 24-sequence WikiText-2 windows, length 512

The control and primary MLP candidate cost 7,966,624 and 7,966,224 bits,
respectively: 1.000502 and 1.000452 BPW. Candidate factor work is 1.361x
dense versus 0.982x for the control, a 38.6% increase in the factorized
projection.

## Projection-family screen

Every family received its own approximately 1-BPW control and equal-budget
candidate allocation. The table reports the best measured candidate for
each attention projection and the primary allocation for each MLP input
projection.

| Projection | Source orientation | Control rank | Candidate | Weighted RMSE change |
| --- | --- | ---: | --- | ---: |
| `mlp.gate_proj` | transposed | 970 | k10, rank 1,344, free 256 | **-1.77%** |
| `mlp.up_proj` | transposed | 970 | k10, rank 1,344, free 256 | **-0.78%** |
| `self_attn.q_proj` | native | 522 | k8, rank 544, free 416 | +1.79% |
| `self_attn.o_proj` | transposed | 522 | k8, rank 544, free 416 | +1.50% |
| `self_attn.k_proj` | native | 191 | k8, rank 256, free 64 | +5.14% |
| `self_attn.v_proj` | native | 191 | k8, rank 256, free 64 | +1.04% |

Positive changes are regressions. Alternative attention allocations did not
repair the result: q and o remained at least 1.50% worse, k at least 5.14%
worse, and v at least 1.04% worse. Their smaller dimensions leave less room
to amortize a useful table while retaining enough free components.

The attention families are rejected for this representation. Gate and up
advance to the deeper screen.

## MLP allocation

The k10 allocation curve has the same interior optimum for both projections:

| Rank | Free dominant rows | Gate change | Up change |
| ---: | ---: | ---: | ---: |
| 1,216 | 480 | -1.19% | -0.35% |
| 1,280 | 352 | -1.21% | -0.29% |
| 1,344 | 256 | **-1.77%** | **-0.78%** |
| 1,376 | 192 | -1.70% | -0.66% |
| 1,472 | 0 | -1.29% | -0.21% |

This repeats the down-projection result: neither maximum rank nor maximum
freedom wins. A minority of free anchors plus a larger coded basis is the
best measured use of the fixed bit budget.

## Trading offline compute for reconstruction

Increasing the outer iterations from 400 to 800 materially improves both
MLP candidates. At 800 iterations:

| Projection | Control RMSE | Candidate RMSE | Change |
| --- | ---: | ---: | ---: |
| Gate | 0.480072 | 0.469466 | **-2.21%** |
| Up | 0.537961 | 0.531139 | **-1.27%** |

With a 16-entry corrected-assignment shortlist, the candidate fits took
53.6 seconds for gate and 56.6 seconds for up, versus 13.4 and 14.9 seconds
for their controls.

Searching 64 base-codeword candidates instead of 16 roughly doubled
candidate fit time:

| Projection | 16-candidate time/change | 64-candidate time/change |
| --- | --- | --- |
| Gate | 53.6 s / -2.209% | 115.0 s / -2.285% |
| Up | 56.6 s / -1.268% | 109.0 s / -1.290% |

The larger search buys only another 0.076 percentage point for gate and
0.022 point for up. The 16-entry search is the preferred operating point;
64 is an optional offline refinement, not a new default.

## Seed and depth behavior

The 800-iteration, 16-candidate result is stable across factorization seeds:

| Projection | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Gate change | -2.209% | -2.24% | -2.25% |
| Up change | -1.268% | -1.27% | -1.26% |

Representative depths all improve locally:

| Block | Gate change | Up change |
| ---: | ---: | ---: |
| 0 | -1.47% | -1.34% |
| 12 | -2.21% | -1.27% |
| 24 | -1.17% | -0.77% |

As with the down-projection screen, local reconstruction alone is not
sufficient evidence that these replacements compose.

## Gate/up functional interaction

At block 12, replacing either projection alone is ambiguous:

| Replaced projection | Free-word KL | Candidate KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| Gate only | 0.028638 | 0.029148 | +1.78% | [-0.001224, +0.002964] |
| Up only | 0.037760 | 0.036426 | -3.53% | [-0.003602, +0.001270] |

Replacing gate and up together passes on both disjoint windows:

| Window | Free-word KL | Candidate KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| 0 through 23 | 0.063514 | 0.060405 | **-4.89%** | [-0.006227, -0.000392] |
| 24 through 47 | 0.072037 | 0.064676 | **-10.22%** | [-0.012301, -0.002840] |
| Combined 48 | 0.067776 | 0.062541 | **-7.72%** | [-0.008227, -0.002551] |

The combined comparison covers 24,528 next-token targets and wins 34 of 48
paired sequences. The joint result is stronger than either isolated
projection, which is consistent with their multiplicative coupling through
the gated MLP. It is direct evidence that correlated projection groups can
be a better replacement unit than individual matrices.

## Cross-block composition

The same pair does not transfer uniformly across depth:

| Selected blocks | First-window change | Disjoint-window change | Combined change | Combined interval |
| --- | ---: | ---: | ---: | ---: |
| 0, 12 | -3.87% | +1.29% | -1.22% | [-0.013055, +0.006981] |
| 0, 12, 24 | **-4.76%** | +0.59% | -2.07% | [-0.016216, +0.004470] |

The first-window three-block result barely passes, but its disjoint
confirmation reverses. A rule requiring both gate and up to improve local
weighted RMSE by at least 1% selects blocks 0 and 12; that policy also fails
disjoint confirmation. Calibration reconstruction therefore does not yet
provide a robust cross-block selector for this projection pair.

## Decision

Accept the following research conclusion:

- the mixed dominant-factor idea transfers from down projection to the
  jointly evaluated `gate_proj` and `up_proj` pair;
- spending roughly twice the factorization iterations is worthwhile;
- increasing corrected-assignment search from 16 to 64 candidates is not
  cost-effective;
- block 12 passes two independent functional windows at equal storage;
- attention projections and uniform representative-depth composition are
  rejected.

Do not add gate/up layers to the packed overlay yet. The existing
`nanoquant-mixed-v-overlay` stores a compact right factor. A transposed
gate/up fit requires orientation metadata and compact left-factor expansion
after returning to source orientation.

The next useful experiment is a complete 26-block gate/up reconstruction
screen followed by a predeclared group-selection rule and disjoint splice
confirmation. Because local gate/up gains did not predict the representative
cross-block result, that rule should include a group-level calibration
metric, such as block-output error after the gated activation, rather than
only two independent matrix RMSE thresholds.

[Dominant-Factor Format Candidate Search](59-dominant-factor-format-candidate-search.md)
exhausts the nearby code-width, correction-count, banked-table, correction-tier,
and opposite-factor alternatives. It retains k10 plus two corrections and
moves the next search to operator-scope gate/up fitting.
