# Product-Codebook Exact Experiment 056 Outlier Gate

**Date:** 2026-08-06

**Status:** promotion rejected; exact matched outliers preserve local
reconstruction wins but do not produce depth-stable or statistically reliable
functional gains

> **Superseded allocation result:** the later 704-free-row screen in
> `Docs/93-product-codebook-free-row-allocation-screen.md` reverses the late
> block failure and passes held-out gates across representative depth and two
> seeds. This document remains authoritative for the rejected 672-row policy.

## Question

Does the compact k16/no-flip product code remain better when both it and the
free-word control use the exact BF16 outlier columns retained by completed
Experiment 056?

This gate keeps the retained 1,200-iteration representation from the search
duration screen: rank 1,152, 672 free right-factor rows, 480 product-coded rows,
two learned 256 by 16 half-word tables, a 16-bit combined index, and no
correction stream. The matched free-word control remains rank 970. Both arms
receive the same seven exact columns and include their values and indices in
the measured bit rate.

## Authoritative outlier inventory

The indices were read from Experiment 056's completed logical-model shards,
not reselected by the probe:

| Block | `mlp.down_proj` exact input columns |
| ---: | --- |
| 0 | 383, 384, 1441, 1801, 4497, 4680, 6592 |
| 12 | 50, 1791, 3043, 3886, 4002, 5534, 5748 |
| 24 | 844, 1111, 1660, 2382, 4902, 5597, 5987 |

The free control costs 1.016717 BPW with the seven-column sidecar. The product
candidate costs 0.994098 BPW, saving 0.022620 BPW while retaining the same
columns exactly.

## Reconstruction results

| Block | Free NRMSE | Product NRMSE | Product change | Result |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 0.514894 | 0.513172 | **-0.334%** | pass |
| 12 | 0.529194 | 0.528460 | **-0.139%** | pass |
| 24 | 0.500256 | 0.501173 | **+0.183%** | **fail** |

The exact outliers do not rescue the late-block failure. The representation's
matrix advantage remains depth-dependent.

## Matched held-out splice quality

Block 12 was evaluated on two disjoint 48-sequence WikiText-2 windows of 512
tokens. Unlike the earlier asymmetric screen, the free and product arms both
contain Experiment 056's exact retained columns.

| Window | Free KL | Product KL | Relative change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| sequences 0-47 | 0.043510 | 0.043275 | -0.540% | [-0.002361, +0.001916] |
| sequences 48-95 | 0.037407 | 0.036713 | -1.855% | [-0.001954, +0.000563] |
| combined 0-95 | 0.040459 | 0.039994 | **-1.148%** | **[-0.001672, +0.000760]** |

Both windows point in the favorable direction, but neither interval nor the
combined 96-sequence interval excludes zero. The earlier significant 3.809%
block-12 result compared a product candidate with seven columns against a free
control without them; it is not evidence for the representation alone.

## Decision

Do not promote the two-codebook representation into the resident format or a
numbered full-model experiment yet. It fails the representative late-block
matrix gate, its functional advantage is not statistically reliable when
outliers are matched, and the separate seed screen found a block-12 seed that
worsened held-out KL.

The next useful work is algorithmic rather than simply longer search:

1. make codebook/free-row allocation layer-aware so late blocks can fall back
   to free words or receive a different coded fraction;
2. train the product tables through resident tuning instead of freezing the
   reconstruction-stage assignment; and
3. repeat the exact matched-column gate across multiple seeds before any
   full-model launch.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block0-down-r970-1200-product-right-k16-rank1152-free672-binary-search-fixed056-outliers7-seed0.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-1200-product-right-k16-rank1152-free672-binary-search-fixed056-outliers7.json`
- `evidence/m4/sign-word-codebook-probe/block24-down-r970-1200-product-right-k16-rank1152-free672-binary-search-fixed056-outliers7-seed0.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-1200-binary-search-fixed056-outliers7-splice-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-1200-binary-search-fixed056-outliers7-splice-offset48-48.json`
