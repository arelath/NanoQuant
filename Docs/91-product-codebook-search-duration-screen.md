# Product-Codebook Search-Duration Screen

**Date:** 2026-08-06

**Status:** 1,200 ADMM iterations retained as the strongest k16/no-flip
product-code setting; longer post-fit binary search and 1,600 ADMM iterations
rejected

## Question

Can the compact two-8-bit product code find a better match if its fitting or
post-fit binary search is allowed to run longer?

The representation and bit allocation remain unchanged from the original
screen:

- pinned Gemma block 12 `mlp.down_proj`, shape 1,152 by 6,912;
- rank-970 free-word control;
- rank-1,152 product-code candidate at the physical rank cap;
- free left factor, 672 free right-factor rows, 480 product-coded rows;
- two learned 256 by 16 half-word tables;
- 16-bit word index and no correction stream;
- retained control-then-tabu search over representation-free signs; and
- optional seven exact BF16 Fisher-selected columns.

## Longer post-fit binary search

Holding ADMM at 800 iterations and increasing both binary-search outer stages
from 8 to 24 passes does not materially improve the product code.

| Arm | 8+8 search NRMSE | 24+24 search NRMSE |
| --- | ---: | ---: |
| Matched free words | 0.532465 | 0.532444 |
| k16 product code | 0.531932 | 0.531930 |
| Product plus seven columns | 0.530843 | 0.530843 |

The candidate changes by less than 0.0004% while search time rises
substantially.  The control also improves, so the product code's relative lead
does not increase.  The representation-preserving binary search is saturated.

## Longer joint fit

Increasing ADMM duration continues to lower reconstruction error for both
arms.  The product code improves more than its matched free control through
1,600 iterations.

| ADMM iterations | Free NRMSE | Product NRMSE | Product change | Product + 7 columns | Column-arm change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 800 | 0.532465 | 0.531932 | **-0.100%** | 0.530843 | **-0.305%** |
| 1,200 | 0.530611 | 0.529603 | **-0.190%** | 0.528543 | **-0.390%** |
| 1,600 | 0.529500 | 0.528132 | **-0.258%** | 0.527083 | **-0.456%** |

This matrix-only trend is not sufficient to choose the longest run.  The
held-out operator gate selects a different optimum.

## Held-out splice quality

Each setting is evaluated on two disjoint 48-sequence WikiText-2 windows of
512 tokens.  The table combines all 96 sequences with a paired sequence
bootstrap.

| ADMM iterations | Free KL | Product KL | Relative change | Paired 95% interval | Confident improvement |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 800 | 0.041766 | 0.040858 | **-2.174%** | [-0.002141, +0.000235] | No |
| 1,200 | 0.042036 | 0.040435 | **-3.809%** | **[-0.002826, -0.000346]** | **Yes** |
| 1,600 | 0.040471 | 0.040391 | **-0.198%** | [-0.001206, +0.000962] | No |

Both 1,200-step windows favor the product code:

- sequences 0-47: -3.818%, interval [-0.003853, +0.000424];
- sequences 48-95: -3.797%, interval [-0.002670, -0.000215].

The second window is independently significant, and the combined inventory
excludes zero.  In contrast, the 1,600-step fit improves only the first matrix
objective; one held-out window worsens and the combined functional advantage
nearly disappears.

## Decision

Retain 1,200 ADMM iterations with the original 8+8 post-fit binary search for
the compact k16/no-flip product code.  More post-fit search is wasted work, and
1,600 iterations overfit the reconstruction objective relative to the held-out
functional gate.

This is substantially stronger evidence than the original product-code
screen, but it is not sufficient to promote a resident format or launch a
numbered full-model run.  The next gates remain:

1. repeat 1,200 iterations across representative blocks and factorization
   seeds;
2. evaluate Experiment 056's exact residual-selected columns on matched free
   and product-code controls; and
3. carry the product constraint through resident tuning and a codebook-aware
   global allocator before any complete experiment.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-product-right-k16-rank1152-free672-binary-search24x24-outliers0-7.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-1200-product-right-k16-rank1152-free672-binary-search-outliers0-7.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-1600-product-right-k16-rank1152-free672-binary-search-outliers0-7.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-1200-binary-search-splice-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-1200-binary-search-splice-offset48-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-1600-binary-search-splice-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-1600-binary-search-splice-offset48-48.json`
