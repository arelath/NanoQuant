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

## Composed full-policy gate

The resumable v3 materialization completed all 64 jobs and published both
78-matrix overlays. The free-row overlay tensor hash is
`11896abcc46b028e7133e0c78587f5bb04727fcd41cfbb31e3583e2a9a6e289d`; the
corrected-codebook overlay tensor hash is
`3a0ed4a1bcac926797715dc4711cc05e9fb3f9a5774beb259c64c04396735358`.
A fresh completed-driver replay revalidated every retained cache identity and
overlay hash.

The exact composed candidate passes two disjoint held-out windows against the
same-budget free-row control. Without the retained Experiment 056 global
tuning, the comparison is:

| WikiText-2 window | Free KL | Candidate KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 192-239 | 6.842462 | 6.000663 | **-12.303%** | **[-1.056979, -0.624185]** |
| sequences 240-287 | 6.487112 | 5.921244 | **-8.723%** | **[-0.906057, -0.236634]** |
| combined 192-287 | 6.664787 | 5.960954 | **-10.560%** | **[-0.913700, -0.504443]** |

Combined NLL improves by 8.582%, from 9.468939 to 8.656331, with paired
interval `[-1.024439, -0.610561]`.

The relative result survives replay of Experiment 056's retained global-tuning
artifact:

| WikiText-2 window | Free KL | Candidate KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 192-239 | 6.497903 | 5.894639 | **-9.284%** | **[-0.826930, -0.391740]** |
| sequences 240-287 | 6.156348 | 5.831860 | **-5.271%** | **[-0.587304, -0.073409]** |
| combined 192-287 | 6.327125 | 5.863250 | **-7.332%** | **[-0.640150, -0.295720]** |

Combined tuned NLL improves by 5.943%, from 9.081706 to 8.542022, with paired
interval `[-0.712904, -0.372808]`.

This is a decisive representation result, not an absolute-quality result. On
the same combined tuned window, the frozen Experiment 056 baseline has KL
2.140387 and NLL 4.775861. The independently reconstructed candidate remains
173.934% worse in KL and 78.858% worse in NLL. Reusing tuning fitted to a
different representation does not close that gap.

## Candidate-specific zero-bit refit

The accepted candidate was refitted in composed student context using only
separable MLP scales: gate and up output scales plus down input and output
scales. These transformations can be folded into the existing product-factor
scale axes without adding indices, codebook entries, or scale elements. The
current evidence is still a dense analysis overlay; a component-level replay
must prove that zero-bit fold before runtime integration.

The first forward coordinate pass fitted on test sequences 460-467 and
accepted each block only when it improved the disjoint 468-475 selection
window. It accepted 24 of 26 blocks and rolled back blocks 10 and 18. Selection
NLL fell from 8.565156 to 6.105200. On the untouched evaluation inventory:

| WikiText-2 window | Candidate KL | Pass-1 KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 192-239 | 5.894639 | 3.583015 | **-39.216%** | **[-2.443849, -2.182226]** |
| sequences 240-287 | 5.831860 | 3.683527 | **-36.838%** | **[-2.273619, -2.022244]** |
| combined 192-287 | 5.863250 | 3.633271 | **-38.033%** | **[-2.321761, -2.138146]** |

Combined NLL improves by 25.021%, from 8.542022 to 6.404749, with paired
interval `[-2.226181, -2.046707]`.

One bounded second pass was justified because the first pass changed nearly
every downstream student context. It accepted 15 of 26 coordinates and moved
selection NLL from 6.105200 to 6.010986. The independent comparison against
pass 1 also passes both windows:

| WikiText-2 window | Pass-1 KL | Pass-2 KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 192-239 | 3.583015 | 3.514608 | **-1.909%** | **[-0.082485, -0.054718]** |
| sequences 240-287 | 3.683527 | 3.582006 | **-2.756%** | **[-0.120081, -0.083807]** |
| combined 192-287 | 3.633271 | 3.548307 | **-2.339%** | **[-0.096999, -0.073309]** |

Combined pass-2 NLL improves by 1.593% over pass 1, from 6.404749 to
6.302698, with interval `[-0.114390, -0.089985]`. Relative to the original v3
candidate, the two passes improve combined KL by 39.482%, interval
`[-2.407990, -2.222767]`, and NLL by 26.215%, interval
`[-2.327442, -2.149619]`.

The diminishing second-pass gain does not justify a third pass on the reused
selection window. Absolute quality is still insufficient: pass 2 remains
65.779% worse in KL and 31.970% worse in NLL than the frozen Experiment 056
baseline on the same 96 sequences.

## Decision

Do not promote or integrate the unconstrained policy or either rejected early
payment. Promote v3 from allocation candidate to the codebook policy for a
candidate-specific integrated/tuned prototype: the exact full-policy and
two-pass refit comparisons pass independently. Do not promote it as a final
Experiment 056 replacement or publishable 1-BPW model. Its absolute quality is
still far behind the frozen Experiment 056 model. The next gate is exporting
and exactly replaying the accepted separable refits in the product-codebook
components at unchanged bits, then running the full protocol-matched quality
evaluation. Resident and packed-format integration is justified only as the
minimum prototype needed to run that gate, not as a format promotion.

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
- `evidence/m4/product-codebook-mixed-policy-v3-materialization/manifest.json`
- `evidence/m4/product-codebook-mixed-policy-v3-composed-kl-offset192-48.json`
- `evidence/m4/product-codebook-mixed-policy-v3-composed-kl-offset240-48.json`
- `evidence/m4/product-codebook-mixed-policy-v3-composed-kl-tuned-offset192-48.json`
- `evidence/m4/product-codebook-mixed-policy-v3-composed-kl-tuned-offset240-48.json`
- `evidence/m4/product-codebook-mixed-policy-v3-candidate-coordinate-refit.json`
- `evidence/m4/product-codebook-mixed-policy-v3-coordinate-kl-tuned-offset192-48.json`
- `evidence/m4/product-codebook-mixed-policy-v3-coordinate-kl-tuned-offset240-48.json`
- `evidence/m4/product-codebook-mixed-policy-v3-candidate-coordinate-refit-pass2.json`
- `evidence/m4/product-codebook-mixed-policy-v3-coordinate-pass2-kl-tuned-offset192-48.json`
- `evidence/m4/product-codebook-mixed-policy-v3-coordinate-pass2-kl-tuned-offset240-48.json`
