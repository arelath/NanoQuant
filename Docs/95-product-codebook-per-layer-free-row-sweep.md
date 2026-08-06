# Product-Codebook Per-Layer Free-Row Sweep

**Date:** 2026-08-06

**Status:** retain a layer-aware 672/704 free-row policy for `mlp.down_proj`;
functional and resident-integration gates remain open

## Question

Does every Gemma `mlp.down_proj` layer need the conservative rank-1,152/free-704
product-code allocation, or can individual layers spend fewer free sign rows
while still improving on the matched rank-970 free-factor control?

The sweep covers all 26 blocks. Each layer uses 1,200 ADMM iterations, the
retained 8+8 control/tabu search, k16/no-flip product tables, seed zero, and its
exact seven Experiment 056 BF16 outlier columns on both the candidate and
control. One free-factor baseline is shared across the five candidates for
that layer.

## Rate ladder

| Free rows | Effective BPW | Saving versus control |
| ---: | ---: | ---: |
| 576 | 0.952431 | 0.064286 |
| 608 | 0.966320 | 0.050397 |
| 640 | 0.980209 | 0.036509 |
| 672 | 0.994098 | 0.022620 |
| 704 | 1.007987 | 0.008731 |
| Free-factor control | 1.016717 | — |

The 576, 608, and 640 allocations regress on every layer. The useful boundary
is between 672 and 704.

## Layer-aware result

Fourteen layers first beat their matched control at 672 rows:

`0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 16, 17, 25`

Twelve layers require 704 rows:

`4, 6, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24`

| Policy | Mean effective BPW | Mean NRMSE change versus control |
| --- | ---: | ---: |
| Smallest improving row count per layer | **1.000508** | **-0.387%** |
| Uniform 704 rows | 1.007987 | **-0.803%** |
| Free-factor control | 1.016717 | baseline |

The layer-aware policy recovers 0.016209 BPW relative to the control while
improving every individual layer's reconstruction. Uniform 704 is more
accurate, but spends 0.007479 additional BPW. Which point is preferable must be
decided by the global allocator and functional quality, not matrix NRMSE alone.

## Interpretation

The required free-row count is depth-dependent. The early and middle model
mostly tolerates 672 rows, while a contiguous late region from blocks 18 through
24 requires 704. This rejects a universal minimum-row rule and supports
carrying a per-layer coded/free allocation decision into the allocator.

This evidence does not yet promote the representation. The next gates are:

1. screen shape-appropriate free-row ladders for gate, up, Q, K, V, and O at
   representative depths;
2. run matched functional splice tests for reconstruction-passing policies;
3. implement constrained resident tuning and packed/runtime round trips; and
4. let the global allocator select between rate and functional quality.

## Evidence

The terminal campaign receipt is
`evidence/m4/product-codebook-free-row-sweep/summary.json`. Per-layer receipts
are `block-00.json` through `block-25.json` in the same directory. The terminal
status is `completed`, all 26 blocks are present, and no failure is recorded.
