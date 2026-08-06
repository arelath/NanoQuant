# Product-Codebook Projection-Family Screen

**Date:** 2026-08-06

**Status:** retain product coding for `mlp.down_proj` research only; reject
attention and joint gate/up replacement at the tested equal-or-lower rates

## Question

Does the robust down-projection product-code result extend to the other Gemma
projection shapes when both arms receive Experiment 056's exact retained
columns and the candidate does not exceed the free-word control rate?

Wide gate, up, and O matrices are transposed so the product code constrains the
dominant factor, following the repository's established wide-matrix numerical
orientation. An exact source input column becomes an exact row in transposed
factorization space and is transposed back before the model splice. The probe
receipt records this axis and charges the original source output count.

All arms use 1,200 ADMM iterations, two scale-fit passes, the retained 8+8
control/tabu binary search, k16/no-flip product tables, block 12 corrected-CCE
Fisher importance, and the exact two-column Experiment 056 sidecar on both
candidate and control.

## Equal-rate configurations

| Family | Free control rank | Product rank/free rows | Free BPW | Product BPW |
| --- | ---: | ---: | ---: | ---: |
| Gate/up, transposed | 970 | 1,152 / 704 | 1.028283 | 1.019552 |
| Q | 522 | 576 / 352 | 1.027281 | 1.025206 |
| K/V | 191 | 224 / 128 | 1.026496 | 1.026171 |
| O, transposed | 522 | 576 / 352 | 1.030752 | 1.028676 |

More aggressive attention allocations were also screened: Q/O rank 672/free
64 and K/V rank 256/free 64. They save more bits but regress more strongly.

## Representative reconstruction

| Projection | Free NRMSE | Best tested product NRMSE | Change | Gate |
| --- | ---: | ---: | ---: | :---: |
| Gate, transposed | 0.476254 | 0.468457 | **-1.637%** | pass |
| Up, transposed | 0.533903 | 0.527718 | **-1.158%** | pass |
| Q | 0.397696 | 0.403854 | +1.548% | fail |
| K | 0.407682 | 0.415242 | +1.854% | fail |
| V | 0.619514 | 0.617905 | **-0.260%** | pass |
| O, transposed | 0.480672 | 0.484587 | +0.815% | fail |

The aggressive attention points regress by +5.272% for Q, +4.613% for K,
+0.016% for V, and +2.604% for O. Conservative allocations narrow the losses
but only V crosses the reconstruction threshold.

## Joint gate/up held-out result

Gate and up share block 12's exact indices and transposed geometry, so they
were spliced jointly rather than promoted from isolated matrix objectives.

| Window | Free KL | Product KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 0-47 | 0.069284 | 0.066278 | -4.338% | [-0.008130, +0.001995] |
| sequences 48-95 | 0.053303 | 0.056915 | **+6.777%** | **[+0.001078, +0.006249]** |
| combined 0-95 | 0.061293 | 0.061597 | +0.495% | [-0.002690, +0.003268] |

The matrix wins do not compose reliably through the MLP. Joint gate/up product
coding is rejected.

## V-projection depth gate

The conservative rank-224/free-128 V arm passes reconstruction at blocks 0,
12, and 24 by 0.349%, 0.260%, and 0.361%, respectively. Its held-out behavior
does not remain stable across depth.

| Block | Free KL | Product KL | Relative change | Paired 95% interval |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.064542 | 0.061953 | -4.012% | [-0.005735, +0.000531] |
| 12 | 0.040934 | 0.038500 | **-5.947%** | **[-0.003660, -0.001233]** |
| 24 | 0.006117 | 0.006609 | **+8.032%** | **[+0.000360, +0.000619]** |

Both block-24 48-sequence windows are confidently worse. A middle-block gain
does not justify a universal V representation, and an all-block selective-V
campaign is lower priority than the already robust down-projection path.

## Decision

Narrow the next resident prototype to `mlp.down_proj` only:

- use rank 1,152/free 704 product coding where the down-projection policy and
  global allocator select it;
- retain ordinary free factors for Q, K, V, O, gate, and up;
- do not infer functional quality from gate/up reconstruction gains; and
- do not launch a numbered complete run until down-only constrained tuning,
  packed representation, and round-trip contracts exist.

The screen eliminates the need to generalize the first resident implementation
across every projection orientation. Transpose-aware fixed-outlier support is
retained in the analysis probes because it makes the rejection reproducible
and is useful for future rate/allocation research, but it is not a runtime
schema.

## Evidence

Reconstruction receipts are under
`evidence/m4/sign-word-codebook-probe/block12-{gate,up,q,k,v,o}-*.json`.
Representative functional receipts are:

- `block12-gate-up-product-right-k16-r1152-free704-1200-transpose-binary-search-fixed056-outliers2-splice-48.json`
- `block12-gate-up-product-right-k16-r1152-free704-1200-transpose-binary-search-fixed056-outliers2-splice-offset48-48.json`
- `block{0,12,24}-v-product-right-k16-r224-free128-1200-binary-search-fixed056-outliers2-splice-48.json`
- `block{0,12,24}-v-product-right-k16-r224-free128-1200-binary-search-fixed056-outliers2-splice-offset48-48.json`

All are retained under `evidence/m4/sign-word-codebook-probe`.
