# Progressive Sign-Fixing Screen

**Date:** 2026-07-30
**Status:** completed; equal-bit reconstruction screen failed

## Question

The fitted codebook screen in
[Sign-Word Codebook Screen](51-sign-word-codebook-screen.md) showed that
arbitrary 8- and 12-bit word codebooks could not repay their loss of
independent signs with additional components. This follow-up tests a cheaper
implicit code:

1. warm up an over-complete free-sign factorization;
2. find the still-variable bit position with the strongest global majority;
3. fix that position to its majority value in every 32-sign word;
4. continue fitting and repeat until only `k` positions remain variable.

The resulting words need no lookup table. A 32-bit mask and 32-bit template
per factor identify the variable and fixed positions. Each word then stores
only its `k` variable signs.

## Implementation

`src/nanoquant/domain/progressive_sign_fixing.py` implements exact bit
accounting, aligned rank selection, majority selection, constraint
application, and an analysis-only constrained factorizer.

For U and V independently, the solver:

- runs unconstrained ADMM for the first 25% of iterations;
- progressively fixes `32-k` positions between 25% and 50% of the run;
- uses the remaining half for convergence with the completed mask;
- selects each position from the current latent factors by unweighted global
  sign majority, as proposed;
- exports signs that exactly satisfy the final mask and template.

Ordinary nonconvex ADMM multipliers became unstable after the feasible set
was reduced to a global coordinate subspace. Once fixing begins, the solver
therefore continues as projected alternating ridge solves without carrying
stale dual state. This produced finite, repeatable results and leaves the
fixed-sign constraint active throughout the remainder of fitting.

The implementation is analysis-only. It does not change the resident
algorithm, artifact schema, GGUF, or runtime.

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Matrix: block 12 `mlp.down_proj`, shape 1,152 x 6,912
- Importance: retained 256-sample corrected-CCE Fisher state, shrinkage 0.6
- Baseline: ordinary production ADMM, rank 970, 800 outer iterations,
  two-pass scale fit
- Candidate: progressive global fixing, `k in {8, 12}`, 800 outer
  iterations, two-pass scale fit
- Fixing schedule: free through iteration 200, progressive through iteration
  400, fixed thereafter
- Rank alignment: 32
- Budget: complete free-word signs and 16-bit scales versus `k` bits per
  sign word, 16-bit scales, and two 64-bit mask/template descriptors
- Seeds: 0 for both arms; seeds 1 and 2 repeat the less-bad k=12 arm

The canonical command is:

```powershell
.\.venv\Scripts\python.exe tools\probe_sign_word_codebook.py `
  --model <pinned-snapshot>\model.safetensors `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\sign-word-codebook-probe\block12-down-r970-800-progressive-k12-projected.json `
  --block 12 --projection down --baseline-rank 970 `
  --index-widths 12 --outer-iterations 800 `
  --codebook-mode progressive --progressive-warmup-fraction 0.25 `
  --codebook-freeze-fraction 0.5
```

## Equal-bit arithmetic

| Arm | Rank | Rank/baseline | Actual BPW | Metadata bits | Unused bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free 32-bit words | 970 | 1.000x | 1.000502 | 0 | 0 |
| Progressive k=12 | 2,560 | 2.639x | 0.993586 | 128 | 55,072 |
| Progressive k=8 | 3,840 | 3.959x | 0.996158 | 128 | 34,592 |

Removing the lookup tables funds 96 more aligned ranks at k=12 than the
arbitrary-table experiment. The k=8 aligned rank is unchanged.

## Result

| Arm | Weighted RMSE | RMSE change | Weighted error-energy change | Raw RMSE change |
| --- | ---: | ---: | ---: | ---: |
| Free words, rank 970 | 0.533293 | - | - | - |
| Progressive k=12, rank 2,560 | 0.867535 | **+62.68%** | +164.63% | +56.91% |
| Progressive k=8, rank 3,840 | 0.897032 | **+68.21%** | +182.93% | +62.76% |

Progressive fixing is much worse than the arbitrary codebook result, whose
k=12 arm regressed by 9.26%. The lookup-free representation gains 3.9% more
rank than that arm, but imposes a far stronger constraint: 20 of every 32
positions in k=12, or 24 in k=8, must have the same sign in every word.

### Majority strength

| Arm | U mean | U maximum | V mean | V maximum |
| --- | ---: | ---: | ---: | ---: |
| Progressive k=12 | 50.240% | 50.413% | 51.699% | 54.924% |
| Progressive k=8 | 50.257% | 50.356% | 51.727% | 54.591% |

The learned U signs have essentially maximum marginal entropy: even the best
remaining position is generally only a few tenths of a percentage point from
a coin flip. V contains a little more positional bias, but not enough to
justify fixing most of each word globally. Choosing the best position at
each step therefore minimizes the immediate damage without exposing useful
low-dimensional structure.

### Seed repeat

| Seed | Free-word RMSE | k=12 RMSE | k=12 change |
| ---: | ---: | ---: | ---: |
| 0 | 0.533293 | 0.867535 | +62.68% |
| 1 | 0.533245 | 0.869432 | +63.05% |
| 2 | 0.533205 | 0.862466 | +61.75% |

The mean regression is 62.49%, with all seeds losing by more than 61%.
Initialization noise does not explain the result.

## Runtime implication

The storage exchange also raises factorized work from about 0.98x dense at
rank 970 to 2.59x dense for k=12 and 3.89x for k=8. Although mask/template
decode is cheaper than a LUT, it cannot remove the extra factor
accumulations. Neither reconstruction nor compute passes the screen.

## Decision

Reject globally progressive bit-position fixing at k=8 and k=12 for this
equal-1-BPW operating point. Do not proceed to splice KL, a numbered complete
run, packed-format work, or a runtime kernel.

This is a useful negative result: there is no exploitable global marginal
sign bias in these learned factors. Future sign compression would need to
preserve correlations among whole words or use locally conditioned masks;
globally fixing positions destroys too much independent factor capacity,
even when the saved bits fund 2.64x to 3.96x as many components.
