# Product-Codebook Mixed 1.0-BPW Allocation

**Date:** 2026-08-07

**Status:** reject the unconstrained weighted-energy policy; retain the v3
per-matrix-regression-bounded policy for composed full-policy validation, not
resident or packed-format promotion

## Question

Which mixture of free factors and k16/no-flip product-coded factors gives the
best measured reconstruction under a strict 1.0-bit-per-parameter ceiling over
Gemma's seven quantized linear projections?

The allocator uses all 26-layer gate, up, and down free-row curves. Q, K, V,
and O remain fixed to their matched free-factor controls because the
projection-family screen did not find a functionally reliable replacement.
Every rate includes factor scales, product tables and indices, and Experiment
056's BF16 outlier values and indices.

## Unconstrained optimum

The minimum summed calibration-weighted MLP error policy uses 697,753,234 of
697,761,792 available bits, or 0.999987735 BPW. Its aggregate proxy improves:

| Metric | Free-factor controls | Selected | Change |
| --- | ---: | ---: | ---: |
| Weighted error energy | 1,300.9824 | 1,275.8081 | -1.935% |
| Aggregate proxy NRMSE | 0.510161 | 0.505201 | -0.972% |

The optimizer obtains this result by spending heavily on high-energy early
layers while assigning free576 to many low-energy late gate/up matrices. Those
individual matrices regress even though their absolute contribution to the
additive objective is small.

## Functional rejection of the aggressive late choice

Block 24 represents that risky behavior: the unconstrained policy selects
free576 for both gate and up. Both arms use the exact shared Experiment 056
outliers `(412, 835)`, rank 1,152, 1,200 ADMM iterations, and the retained
binary-factor search.

| WikiText-2 window | Free KL | Free576 KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 0-47 | 0.038786 | 0.039975 | +3.064% | [-0.000843, +0.003445] |
| sequences 48-95 | 0.031862 | 0.032014 | +0.476% | [-0.000876, +0.001198] |
| combined 0-95 | 0.035324 | 0.035994 | +1.897% | [-0.000467, +0.001931] |

Neither interval independently proves harm, but both disjoint point estimates
are unfavorable. The additive raw-energy objective therefore does not provide
adequate protection against sacrificing low-energy layers and is rejected for
promotion.

## Regression-bounded exact policy

Requiring every selected matrix to remain at or below its matched control is
not feasible at 1.0 BPW: the cheapest such policy costs 1.005535 BPW. A 1%
maximum weighted-error-energy regression per matrix is the smallest tested
bound that is feasible; its minimum representation costs 0.998086 BPW before
the allocator spends the remaining budget.

The optimized bounded policy also lands at 0.999987735 BPW. It improves
aggregate weighted error by 0.667% and proxy NRMSE from 0.510161 to 0.508458.
Its allocation is:

| Projection | free640 | free672 | free704 |
| --- | ---: | ---: | ---: |
| Gate | 22 | 3 | 1 |
| Up | 14 | 9 | 3 |
| Down | 1 | 22 | 3 |

The analysis allocator exposes this rule as
`--maximum-matrix-error-regression-fraction`; its versioned receipt records the
constraint explicitly.

## Exact late-layer bounded gate

The bounded policy selects gate free640 and up free672 at block 24. The splice
probe supports projection-specific free-row counts so this combination is
tested exactly, with independent reconstruction cache identities and an
aggregate bit receipt. The pair costs 0.998719 BPW versus 1.028283 for the two
matched free-factor controls.

| WikiText-2 window | Free KL | Gate640/up672 KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 0-47 | 0.038786 | 0.035733 | **-7.872%** | **[-0.004426, -0.001740]** |
| sequences 48-95 | 0.031862 | 0.029989 | **-5.879%** | **[-0.003208, -0.000540]** |
| combined 0-95 | 0.035324 | 0.032861 | **-6.973%** | **[-0.003420, -0.001483]** |

The conservative uniform-free672 bracket also passes both windows and improves
combined KL by 5.977%. The exact selected combination is therefore not relying
on that more expensive proxy and passes the representative late-depth gate on
its own terms.

## Middle-depth functional floor

The original bounded policy selected gate640/up640 at block 12. Its first
window improved by 3.193%, but the second regressed by 2.577%. Combined KL
changed from 0.061293 to 0.060874, only -0.684%, with interval
`[-0.002426, +0.001499]`. This point is neutral and not a robust promotion
gate.

Raising only up to 672 does not rescue the instability. Gate640/up672 improves
the first window by 4.360%, but the second is confidently 4.307% worse;
combined improvement is only 0.592% with a zero-crossing interval. The earlier
exact gate672/up672 receipt is therefore the retained middle-depth boundary:
combined KL improves 3.775%, with interval `[-0.004604, -0.000159]`.

The allocator's group floor records this functional constraint explicitly:
`block-12:gate=672,block-12:up=672`. The revised policy remains at
0.999987735 BPW with 8,558 slack bits. It upgrades those two block-12 matrices
and pays for both by changing only block-3 gate from free704 to free640. The
aggregate weighted-error proxy still improves by 0.528%, with NRMSE changing
from 0.510161 to 0.508813.

## Early-depth payment search

The first functionally floored allocation is not safe. Its compensating
block-3 gate640/up704 choice is worse on both disjoint windows and changes
combined KL from 0.197948 to 0.202763, or **+2.432%**, with interval
`[-0.001481, +0.011180]`.

Forcing block-3 gate back to at least 672 rows makes the exact allocator pay
the same 110,592 bits by reducing block-1 down from free704 to free672. That
trade also points worse on both windows: combined KL changes from 0.095487 to
0.097637, or **+2.252%**, with interval `[-0.002648, +0.007386]`. It is rejected
despite the interval crossing zero because the two independent point estimates
agree on the adverse direction.

The v3 allocation also protects block-1 down at free704. The next exact payment
is block-4 up free704 to free672, making block 4's selected gate/up pair
free640/free672. Its two windows disagree, and their combined result is
effectively neutral:

| WikiText-2 window | Free KL | Gate640/up672 KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 0-47 | 0.199502 | 0.201919 | +1.211% | [-0.003913, +0.008637] |
| sequences 48-95 | 0.179261 | 0.176487 | -1.548% | [-0.008475, +0.003049] |
| combined 0-95 | 0.189382 | 0.189203 | -0.094% | [-0.004493, +0.004084] |

The restored block-3 gate672/up704 pair improves the first two windows and
regresses on the next two. Over the expanded 192-sequence inventory it is also
neutral rather than a statistically supported win:

| WikiText-2 window | Free KL | Gate672/up704 KL | Relative change |
| --- | ---: | ---: | ---: |
| sequences 0-47 | 0.212287 | 0.206864 | -2.555% |
| sequences 48-95 | 0.183609 | 0.176758 | -3.731% |
| sequences 96-143 | 0.195087 | 0.202949 | +4.030% |
| sequences 144-191 | 0.213165 | 0.215127 | +0.921% |
| combined 0-191 | 0.201037 | 0.200425 | -0.304% |

The combined paired interval is `[-0.005435, +0.004197]`. This larger sample
shows why two favorable windows were not enough for promotion.

V3 still uses exactly 697,753,234 bits (0.999987735 BPW, 8,558 slack bits).
Its aggregate calibration-weighted error improves by 0.522%, and proxy NRMSE
changes from 0.510161 to 0.508828. Relative to the first functionally floored
receipt it changes only block-3 gate free640 to free672 and block-4 up free704
to free672, while protecting block-1 down at free704.

## Decision

Do not promote or integrate the unconstrained policy or either rejected early
payment. V3 is the leading exact-1.0 candidate: its middle and late gates pass,
and its tested early gates are neutral overall rather than adverse. However,
the early gates do not prove a quality improvement, and the allocation remains
an additive matrix proxy. The next promotion gate is a composed multi-block or
full-policy KL comparison using the exact v3 selections. Resident and packed
format integration should follow only if that composed test passes.

## Full-policy materialization protocol

`tools/materialize_product_codebook_mixed_policy.py` turns the exact allocation
receipt into the two complete dense overlays required by the existing composed
KL evaluator. It is intentionally an analysis-only bridge, not a resident or
packed-format implementation.

The driver validates all 78 allocation choices against their retained source
sweep receipts. V3 requires 64 sequential jobs: 14 joint gate/up jobs, 24
single gate or up jobs where their exact outliers differ, and 26 down jobs.
Every job preserves its projection's selected free-row count and exact
Experiment 056 outliers, uses the content-keyed reconstruction cache, and
retains the 1,200-iteration binary-search protocol. A versioned root manifest
records the current job and completed receipt inventory, so a stopped campaign
resumes without overwriting valid evidence.

Only after all cache identities and hashes validate does the driver atomically
publish complete 78-matrix `free-words` and `corrected-codebook` overlays. Those
overlays are the inputs to `tools/probe_mlp_overlays_kl.py`; their existence is
not itself a quality result or a promotion gate.

## Evidence

- `evidence/m4/product-codebook-mixed-allocation-1bpw.json`
- `evidence/m4/product-codebook-mixed-allocation-1bpw-max-regression-1pct.json`
- `evidence/m4/product-codebook-mixed-allocation-1bpw-functional-floors.json`
- `evidence/m4/product-codebook-mixed-allocation-1bpw-functional-floors-v2.json`
- `evidence/m4/product-codebook-mixed-allocation-1bpw-functional-floors-v3.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block24-gate-up-free576-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block24-gate-up-free576-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block24-gate-up-free672-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block24-gate-up-free672-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block24-gate640-up672-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block24-gate640-up672-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block12-gate-up-free640-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block12-gate-up-free640-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block12-gate640-up672-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block12-gate640-up672-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block03-gate640-up704-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block03-gate640-up704-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block01-down672-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block01-down672-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block04-gate640-up672-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block04-gate640-up672-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block03-gate672-up704-offset0-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block03-gate672-up704-offset48-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block03-gate672-up704-offset96-48.json`
- `evidence/m4/product-codebook-mixed-allocation-functional-gates/block03-gate672-up704-offset144-48.json`
