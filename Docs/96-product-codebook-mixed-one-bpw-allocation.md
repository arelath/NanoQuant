# Product-Codebook Mixed 1.0-BPW Allocation

**Date:** 2026-08-07

**Status:** reject the unconstrained weighted-energy policy; retain the
per-matrix-regression-bounded policy after its exact late-layer functional gate

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

## Decision

Do not promote or integrate the unconstrained exact-1.0 policy. Retain the 1%
bounded, functionally floored policy as the leading candidate. Its exact middle
and late gates pass. The compensating block-3 gate640/up704 choice now requires
an exact early-depth functional splice before resident or packed-format work.

## Evidence

- `evidence/m4/product-codebook-mixed-allocation-1bpw.json`
- `evidence/m4/product-codebook-mixed-allocation-1bpw-max-regression-1pct.json`
- `evidence/m4/product-codebook-mixed-allocation-1bpw-functional-floors.json`
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
