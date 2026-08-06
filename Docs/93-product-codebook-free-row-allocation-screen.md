# Product-Codebook Free-Row Allocation Screen

**Date:** 2026-08-06

**Status:** rank-1,152/free-704 retained as a down-projection resident-integration
candidate; full-format promotion still requires other projection shapes and
constrained resident tuning

## Question

Can a small change in the free-versus-coded right-factor allocation remove the
depth and seed instability of the compact k16/no-flip product code while
remaining below the matched free-word rate?

The screen keeps the successful 1,200-iteration fit, 8+8 control/tabu search,
two learned 256 by 16 half-word tables, 16-bit combined index, rank-1,152
physical cap, no correction stream, and each block's exact seven Experiment 056
BF16 `mlp.down_proj` columns. Both candidate and rank-970 free-word control use
the same exact columns.

## Rate ceiling

The matched free-word control costs 1.016717 BPW. The earlier 672-free-row
product candidate costs 0.994098 BPW but fails at block 24. At rank 1,152,
704 is the largest 32-row-aligned free allocation below the control that does
not cross the next expensive step:

| Candidate | Effective BPW | Savings versus free |
| --- | ---: | ---: |
| rank 1,152 / free 672 | 0.994098 | 0.022620 |
| **rank 1,152 / free 704** | **1.007987** | **0.008731** |
| rank 1,152 / free 736 | 1.021876 | over budget |

The selected policy spends 0.013889 BPW of the earlier compression headroom to
make 32 additional right-factor rows free, while retaining a real rate saving.

## Block-24 allocation sweep

Three under-ceiling Pareto candidates were tested against the same seed-0
control and exact outliers.

| Rank | Free rows | Effective BPW | NRMSE | Change versus free |
| ---: | ---: | ---: | ---: | ---: |
| **1,152** | **704** | **1.007987** | **0.497180** | **-0.615%** |
| 1,088 | 800 | 1.012488 | 0.497597 | -0.532% |
| 1,024 | 864 | 1.003100 | 0.502222 | +0.393% |

Rank 1,152/free 704 dominates rank 1,088/free 800: it is both cheaper and more
accurate. Reducing rank to 1,024 loses too much factor capacity. The change from
672 to 704 free rows reverses block 24's earlier +0.183% exact-column failure.

## Representative-depth reconstruction

| Block | Seed | Free NRMSE | Product NRMSE | Change |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.514894 | 0.509206 | **-1.105%** |
| 12 | 0 | 0.529194 | 0.524246 | **-0.935%** |
| 12 | 1 | 0.529149 | 0.524343 | **-0.908%** |
| 24 | 0 | 0.500256 | 0.497180 | **-0.615%** |
| 24 | 1 | 0.500273 | 0.497129 | **-0.628%** |

The retained allocation passes early, middle, and late depth. The middle and
late checks are stable across both tested factorization seeds.

## Held-out splice quality

Each row combines two disjoint 48-sequence WikiText-2 windows of 512 tokens.
Every individual 48-sequence window also has a fully negative paired interval.

| Block | Seed | Free KL | Product KL | Relative change | Paired 95% interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.193877 | 0.173494 | **-10.513%** | **[-0.026315, -0.014681]** |
| 12 | 0 | 0.040459 | 0.038195 | **-5.594%** | **[-0.003385, -0.001144]** |
| 12 | 1 | 0.040052 | 0.037910 | **-5.349%** | **[-0.003255, -0.001041]** |
| 24 | 0 | 0.024739 | 0.022578 | **-8.735%** | **[-0.002952, -0.001378]** |
| 24 | 1 | 0.023857 | 0.022372 | **-6.223%** | **[-0.002075, -0.000895]** |

This removes the functional seed instability seen at 672 free rows. The
improvement is not a product-versus-unmatched-outlier artifact: both arms use
the exact same Experiment 056 columns and both rates include the sidecar.

## Decision

Retain rank 1,152/free 704 as the first compact product-code allocation that
passes the representative down-projection reconstruction and held-out gates.
It is now justified to prototype codebook-aware resident tuning and allocation.

Do not launch a numbered full-model experiment yet. The current evidence is
limited to `mlp.down_proj`; attention and MLP input projections have different
matrix shapes, physical ranks, and two-column Experiment 056 sidecars. Before a
complete run, the implementation must:

1. carry product-code constraints through resident tuning without silently
   materializing free signs;
2. make coded/free allocation shape- and layer-aware under the global bit
   budget;
3. screen the other projection families with their exact retained outliers;
4. add packed/runtime representation contracts and round-trip tests; and
5. repeat the complete exact-quality protocol only after those boundaries pass.

## Evidence

Core reconstruction receipts:

- `evidence/m4/sign-word-codebook-probe/block0-down-r970-1200-product-right-k16-rank1152-free704-binary-search-fixed056-outliers7-seed0.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-1200-product-right-k16-rank1152-free704-binary-search-fixed056-outliers7-seed0.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-1200-product-right-k16-rank1152-free704-binary-search-fixed056-outliers7-seed1.json`
- `evidence/m4/sign-word-codebook-probe/block24-down-r970-1200-product-right-k16-rank1152-free704-binary-search-fixed056-outliers7-seed0.json`
- `evidence/m4/sign-word-codebook-probe/block24-down-r970-1200-product-right-k16-rank1152-free704-binary-search-fixed056-outliers7-seed1.json`

Held-out receipts use the corresponding
`block{0,12,24}-down-product-right-k16-r1152-free704-1200-binary-search-fixed056-outliers7[-seed1]-splice[-offset48]-48.json`
names under `evidence/m4/sign-word-codebook-probe`.
