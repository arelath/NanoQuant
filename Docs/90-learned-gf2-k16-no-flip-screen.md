# Learned GF(2) k16 No-Flip Screen

**Date:** 2026-08-06

**Status:** rejected at the physical-cap reconstruction gate; no splice or
numbered full-model run justified

## Question

Can the compact k16/no-flip product-code result be improved or matched with a
true learned binary linear code?  A GF(2) `[32,16]` code stores a 16 by 32
generator (512 bits) and represents every 32-sign word with a 16-bit message.
It therefore has almost no table overhead and does not need a correction or
bit-flip stream.

## Representation and optimizer

The analysis-only implementation constrains each coded sign word to

`c = mG mod 2`,

where `m` is a 16-bit message and `G` is a learned, full-row-rank 16 by 32
binary generator.  Sign bit zero decodes to +1 and sign bit one decodes to -1.

The final learner uses:

- exactly balanced generator rows (16 negative signs per row);
- minimum non-zero codeword distance at least four;
- a full-rank check after every generator update;
- a GF(2) information-set solve for hard-decision message initialization;
- two alternating exact searches over the generator's two 8-dimensional
  subcodes for soft assignment refinement; and
- coordinate-optimal balanced generator-row updates.

The information-set initializer was necessary to rule out a decoder artifact.
An earlier all-zero-start decoder used only 6,106 to 9,152 messages and reached
8.84 to 9.29 bits of empirical entropy.  The final decoder uses more than
51,000 messages and reaches about 15.47 bits of entropy, so the reported
failure is not message-space collapse.

## Protocol

- Pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Matrix: block 12 `mlp.down_proj`, shape 1,152 by 6,912.
- Objective: retained corrected-CCE Fisher state with 0.6 shrinkage.
- Free control: rank 970 with ordinary 32-bit sign words.
- Candidate: rank 1,152, preserving Experiment 056's physical rank cap;
  free left factor; 672 free right-factor rows; 480 linear-coded rows.
- ADMM: 800 outer by five inner iterations, two-pass scale fit.
- Both arms receive the retained control-then-tabu binary-factor search; only
  the candidate's representation-free signs are mutable.
- Seeds: zero and one.
- Candidate sidecar: seven exact BF16 Fisher-selected columns for seed zero.

The linear descriptor charges 512 generator bits plus a 16-bit free-row count.
With no outliers the candidate occupies 0.976918 BPW, slightly below the
0.977883 BPW product-code candidate.

## Reconstruction result

| Arm | Actual BPW | Weighted NRMSE | Change vs matched free words |
| --- | ---: | ---: | ---: |
| Seed 0 free words, rank 970 | 1.000502 | 0.532465 | - |
| Seed 0 learned GF(2), rank 1,152 | 0.976918 | 0.582555 | **+9.407%** |
| Seed 0 GF(2) plus seven exact columns | 0.993133 | 0.581009 | **+9.117%** |
| Seed 1 free words, rank 970 | 1.000502 | 0.532373 | - |
| Seed 1 learned GF(2), rank 1,152 | 0.976918 | 0.582743 | **+9.461%** |

For seed zero, binary-factor search improves the GF(2) arm from 0.598136 to
0.582555 NRMSE, but cannot approach the free control.  Seed one repeats both
the magnitude and direction.  The seven-column sidecar closes only 0.29
percentage point of the 9.41% gap.

The final seed-zero code uses 52,130 of 65,536 messages at 15.474 bits of
empirical entropy.  Seed one uses 51,916 messages at 15.468 bits.  Both
generators retain rank 16, balanced rows, and minimum distance four.

## Interpretation and decision

Reject a strict learned GF(2) `[32,16]` word code for this factor format.  The
linear closure constraint is much more damaging than the separable product
code's learned nonlinear half-tables, despite comparable 16-bit payloads and
slightly lower metadata cost.

Do not run a splice/KL gate or a numbered full-model experiment.  The block
reconstruction gate fails by more than nine percent on two seeds, while the
product code passed the same gate by 0.10% without outliers and 0.31% with
seven columns.

The useful conclusion is representational: k16 itself is viable, but the
65,536 decoded words need nonlinear learned structure.  Further work should
focus on richer compact nonlinear generators (for example multi-stage or
residual product codes), not additional optimization of a strict linear
subspace.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-linear-right-k16-rank1152-free672-balanced-d4-information-init-binary-search-outliers0-7.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-linear-right-k16-rank1152-free672-balanced-d4-information-init-seed1-binary-search.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-linear-right-k16-rank1152-free672-balanced-d4-binary-search-outliers0-7.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-linear-right-k16-rank1152-free672-binary-search-outliers0-7.json`
