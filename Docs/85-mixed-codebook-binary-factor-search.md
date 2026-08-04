# Mixed Codebook Binary-Factor Search

**Date:** 2026-08-03

**Status:** representation-preserving block-12 screen completed; mixed codebook
still wins at equal bits, but free-factor search narrows rather than enlarges
its advantage

## Question

The retained mixed dominant-factor format beats free sign words by combining
256 fully free basis rows with 1,088 rows encoded by a 10-bit sign-word
codebook and two corrections per word. Experiment 055 now also has a retained
control-then-tabu search that improves final binary factors.

Can those mechanisms be combined without silently turning the coded rows back
into ordinary free signs, and does the combination improve the equal-bit
result?

## Representation-preserving search

The generic binary-factor search cannot be applied unchanged: any accepted
bit flip in a coded row can invalidate its stored codebook index and correction
pair. The probe therefore uses component masks:

- the rank-970 free-word control may search every sign;
- the rank-1,344 mixed arm may search its entire free left factor;
- the mixed arm may search only the first 256 free rows of its right factor;
- the remaining 1,088 corrected-codebook rows are immutable and are checked
  for exact equality after search.

Only the mask-compatible one-bit, variable-depth, and tabu moves are enabled.
Continuous, pair, block, component-replacement, codebook-transfer, and joint
window moves are rejected when a mask contains immutable components. Both
arms receive the same Experiment 055 search settings: 64 scale passes, eight
control outer passes, 16 one-bit passes, two length-64 variable-depth passes,
and eight tabu outer passes with two 256-step sweeps.

This is an analysis-only change. It does not alter the resident algorithm,
artifact schema, packed overlay, GGUF, or runtime.

## Protocol

- Model: pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Matrix: block 12 `mlp.down_proj`, shape 1,152 x 6,912
- Objective: retained corrected-CCE Fisher state with 0.6 shrinkage
- Control: rank 970, ordinary free words
- Candidate: rank 1,344, free left factor, 256 free right rows, 1,088 k10
  two-correction right rows
- ADMM: 800 outer iterations, five inner iterations
- Initial scale fit: two passes
- Search common refit: 64 passes
- Seed: zero

Canonical command:

```powershell
.\.venv\Scripts\python.exe tools\probe_sign_word_codebook.py `
  --model <pinned-snapshot>\model.safetensors `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\sign-word-codebook-probe\block12-down-r970-800-mixed-right-flip2-k10-rank1344-free256-binary-search.json `
  --block 12 --projection down --baseline-rank 970 `
  --candidate-rank 1344 --right-free-rows 256 `
  --index-widths 10 --outer-iterations 800 `
  --codebook-mode full-right-flip2 `
  --assignment-batch-words 8192 --binary-search
```

## Result

| Arm | Before search NRMSE | After search NRMSE | Search change | Search time |
| --- | ---: | ---: | ---: | ---: |
| Free words, rank 970 | 0.533293 | 0.532465 | -0.155% | 28.30 s |
| Mixed codebook, rank 1,344 | 0.528163 | 0.527928 | -0.045% | 26.23 s |

The mixed arm remains better after both arms are searched:

- weighted NRMSE is 0.852% lower than the searched free-word control;
- weighted error energy is 1.697% lower;
- all 1,088 coded right-factor rows remain bit-identical;
- the mixed search changes 10,126 signs in the 256 free right rows and 70
  signs in the free left factor;
- tabu contributes only 572 accepted vector updates in the mixed arm, versus
  12,668 in the fully free control.

Before search, the mixed codebook lead was 0.962% NRMSE. The fully free control
has more mutable capacity and improves by 0.155%, while the constrained mixed
arm improves by 0.045%, so the equal-bit lead narrows by 0.110 percentage
point. The mechanisms are compatible and the absolute candidate is better,
but free-only factor search is not a source of additional relative codebook
advantage.

## Decision

Retain representation-masked search as a valid tool and as the fair control
for any future codebook comparison. Do not promote a new resident format or
complete model run from this one-matrix reconstruction screen.

The next codebook-specific optimizer should target the payload that this
screen intentionally froze: exact-objective substitutions of a coded word's
table index and correction pair, with rollback and a matched free-word
control. That search can test whether the ADMM nearest-latent assignment left
useful final-product headroom. Any surviving reconstruction candidate still
requires a held-out functional gate because prior binary-factor and codebook
work repeatedly showed that small matrix-objective gains do not necessarily
compose through the model.
