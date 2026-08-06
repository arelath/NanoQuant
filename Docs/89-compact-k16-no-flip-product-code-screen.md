# Compact k16 No-Flip Product-Code Screen

**Date:** 2026-08-05

**Status:** physical-cap reconstruction gate passed; two held-out functional
windows directionally favorable but combined confidence interval crosses zero

**Follow-up:** the 1,200-iteration duration screen in
`Docs/91-product-codebook-search-duration-screen.md` strengthens the combined
96-sequence KL improvement to 3.81% with a paired interval excluding zero.
The 1,200-step setting supersedes this document's 800-step optimizer choice.

## Question

Can a compact 16-bit encoding replace the retained 19-bit
codebook-plus-two-flip word while preserving Experiment 056's physical rank
cap and leaving room for its default outliers?

The earlier k16 result does not answer this question. It used an explicit
65,536-entry 32-sign table on both factors, charged more than four million
table bits, fell to rank 896, and regressed weighted NRMSE by 34.98%.

This screen uses the existing Cartesian-product projector as a bounded proxy
for a compact linear-style code. A 16-bit word is split into two 8-bit
indices, each selecting one learned 16-sign half-word. The representation has
65,536 decodable 32-sign words but stores only two 256 by 16 sign tables:
8,192 table bits. It is a separable learned product code, not yet a general
GF(2) `[32,16]` linear generator.

## Protocol

- Model: pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Matrix: block 12 `mlp.down_proj`, shape 1,152 by 6,912.
- Objective: retained corrected-CCE Fisher state with 0.6 shrinkage.
- Free control: rank 970 with ordinary 32-bit sign words.
- Candidate: rank 1,152, preserving Experiment 056's physical rank cap;
  free left factor; 672 free right-factor rows; 480 product-coded rows.
- Coded words: 16-bit index, no correction or flip stream.
- ADMM: 800 outer by five inner iterations, two-pass scale fit, seed zero.
- Both arms receive the retained control-then-tabu binary-factor search; only
  the candidate's free signs are mutable.
- The default-size sidecar arm adds seven exact BF16 Fisher-selected columns.
  This matches the 0.1% down-projection count, but not Experiment 056's
  residual-probe selection identity.
- Functional gate: two disjoint 48-sequence WikiText-2 windows, 512 tokens
  per sequence.

The compact descriptor also charges a 16-bit free-row count, for 8,208 total
metadata bits.

## Reconstruction

| Arm | Actual BPW | Weighted NRMSE | Change vs free words |
| --- | ---: | ---: | ---: |
| Free words, rank 970 | 1.000502 | 0.532465 | - |
| k16 product, rank 1,152, free 672 | 0.977883 | 0.531932 | **-0.100%** |
| Same plus seven exact columns | 0.994098 | 0.530843 | **-0.305%** |

This is the first materially lower-rate fixed code in the current sequence to
cross the block-12 reconstruction gate. Unlike the retained rank-1,344
codebook candidate, it stays at the algebraic rank ceiling. The seven-column
arm remains below the free control's bit cost.

## Held-out splice quality

The splice gate uses the no-outlier arm so both replacements isolate the
factor representation. The independently reconstructed candidate has
weighted NRMSE 0.531994, a 0.088% improvement over its matched free control.

| WikiText window | Free KL | Candidate KL | Change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 0-47 | 0.045156 | 0.043504 | **-3.659%** | [-0.003770, +0.000242] |
| 48-95 | 0.038377 | 0.038213 | **-0.426%** | [-0.001416, +0.001037] |
| Combined 96 | 0.041766 | 0.040858 | **-2.174%** | [-0.002141, +0.000235] |

Both windows favor the product code, but neither the first window nor the
combined inventory excludes zero. The point estimate is nevertheless better
than the equal-bit 15-bit/one-flip arm's earlier 2.16% improvement and is
close to the retained 19-bit arm's 3.88% first-window improvement.

## Decision

Retain compact k16/no-flip encoding as a promising format direction. Do not
promote it or start a numbered full-model run yet.

The next gates are:

1. implement and compare a true compact GF(2) `[32,16]` generator rather than
   assuming the separable product code represents all linear encoders;
2. repeat representative depths and factorization seeds;
3. use Experiment 056's exact residual-selected columns on both the free and
   coded controls;
4. if those pass, add codebook-aware marginal cost curves to the global
   allocator and perform resident constrained tuning before a complete run.

The previous direct frozen-overlay failure still applies: installing freshly
fit codebook weights after resident tuning discards too much recovered model
behavior. A full experiment must carry the code constraint through resident
tuning and global redistribution.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-product-right-k16-rank1152-free672-binary-search-outliers0-7.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-binary-search-splice-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-product-right-k16-r1152-free672-binary-search-splice-offset48-48.json`
