# Product-Codebook Objective Payload Search

**Date:** 2026-08-10

**Status:** exact reconstruction gain passed; combined held-out incremental
quality gate failed

## Question

The retained k16 product code assigns each 16-sign half-word to the learned
table entry with the largest projection against its latent value. For uniform
magnitudes this is exactly the entry with the most matching sign bits in the
same positions. Can a codebook-aware search do better by selecting the two
8-bit indices that minimize the final compressed reconstruction error instead?

## Exact product-payload search

The corrected-code payload coordinate solver now also supports a fixed
`ProductSignCodebook`. For one right-factor component and one 32-column word,
all other factor signs and scales are held fixed. The solver scores the exact
calibration-weighted reconstruction change for every entry in each 16-sign
half table.

The two halves are separable under this coordinate objective. Therefore the
independently best first and second entries are also the exact best result
among all 65,536 combined k16 words, while requiring only 512 half-table
evaluations per word. Proposed words are ranked, sequentially rescored against
the current residual, accepted only on exact improvement, and followed by the
same 64-pass scale refit and outer-pass rollback gate as the corrected-code
search. Final indices are decoded and checked for bit equality with the
searched right factor.

This is an analysis-only extension. It changes no resident algorithm, packed
format, artifact schema, GGUF, or runtime path.

## Protocol

- Model: pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Matrix: block 12 `mlp.down_proj`, shape 1,152 by 6,912.
- Objective: retained corrected-CCE Fisher state with 0.6 shrinkage.
- Free control: rank 970 with ordinary sign words.
- Product candidate: rank 1,152, 672 free right rows and 480 product-coded
  rows, using two learned 256 by 16 tables and no corrections.
- Factor fit: 1,200 ADMM outer iterations by five inner iterations, seed zero.
- Both arms receive the retained 8+8 control-then-tabu search over their free
  signs before payload search.
- Payload budget: eight outer passes, at most 8,192 proposals per pass, and a
  64-pass scale refit.
- Functional gate: WikiText-2 sequences 0-95 in two disjoint 48-sequence,
  512-token windows.

## Reconstruction result

The free-word control is saturated: payload search accepts nine one-sign word
updates and changes its weighted error by less than one part per million. The
product payload accepts 673 word substitutions containing 1,199 sign changes.

| Arm or stage | Weighted error | Weighted NRMSE |
| --- | ---: | ---: |
| Free words after binary search | 6.587547 | 0.530611 |
| Product code before payload search | 6.563577 | 0.529645 |
| Product code after payload search | **6.563304** | **0.529634** |

The product payload stage reduces its weighted error by 0.00416%. Its NRMSE
lead over free words grows from 0.1821% to 0.1842%. This is valid static
objective headroom, but it is very small.

The eight passes evaluate 424,673,280 half-table proposals across 1,990,656
word visits. Gains remain monotonic and representation-valid.

## Held-out quality

Separate executions can produce a small common numerical shift even when the
free reconstruction is identical. The incremental payload comparison therefore
uses the paired difference-in-differences:

```text
(payload product - payload free) -
(binary-only product - binary-only free)
```

| Window | Binary-only product vs free | Payload product vs free | Incremental delta | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 0-47 | -3.818% | -4.289% | -0.000220 | [-0.000602, +0.000157] |
| 48-95 | -3.797% | -3.218% | +0.000223 | [+0.000031, +0.000414] |
| Combined 96 | **-3.809%** | **-3.806%** | **+0.000001** | **[-0.000218, +0.000212]** |

The first window favors payload search and the second rejects it. Across the
full retained inventory the effects cancel: product KL is 0.040435 after
payload search versus 0.040435 before it, after controlling for the matched
free arm. There is no measurable held-out benefit.

## Decision

Do not enable product payload search in resident compression and do not launch
a numbered run from this result. The proposed matching-bit intuition is already
largely represented by the ordinary product assignment, and replacing that
assignment with the exact final matrix objective finds only tiny reconstruction
headroom that does not improve held-out behavior.

This result does not rule out a broader joint search that changes factor signs
and codebook tables together. Such a search should use an operator-level or
functional acceptance signal rather than spending more compute on the diagonal
matrix objective. A true tabu variant would also need an explicit reason to
accept temporary uphill moves; the exact coordinate solver shows that greedy
fixed-table assignment headroom alone is not the missing quality mechanism.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block12-down-r970-1200-product-k16-r1152-free672-objective-payload-splice48-20260810-053202.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-1200-product-k16-r1152-free672-objective-payload-splice-offset48-48-20260810-053611.json`
- `evidence/m4/sign-word-codebook-probe/objective-payload-reconstruction-cache/`
