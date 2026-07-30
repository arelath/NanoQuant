# Sign-Word Codebook Screen

**Date:** 2026-07-30
**Status:** completed; equal-bit reconstruction screen failed

**Follow-up:** [Sparse-Corrected Asymmetric Sign Codes](53-sparse-corrected-sign-code-screen.md)
retains this rejection for pure codebooks but finds a new passing
reconstruction candidate by leaving U free and adding two sparse V-word
corrections.

## Question

Section 3 of
[CapacityRelaxations](ImprovementSuggestions/CapacityRelaxations.md) proposes
replacing every stored 32-sign word with a fixed-width index into a fitted
sign-word codebook. Reducing a word from 32 bits to 8 or 12 index bits would
fund substantially more rank components at the same total storage.

This screen asks the proposal's first question on Gemma-3-1B
`model.layers.12.mlp.down_proj.weight`: can a codebook-constrained,
over-complete joint fit beat the ordinary free-sign rank-970 fit before any
packed-format or runtime work?

## Implementation

`src/nanoquant/domain/sign_word_codebook.py` implements an analysis-only
constrained factorizer. It does not change the resident algorithm, persisted
artifact schema, GGUF, or runtime.

The solver:

- supports rank above `min(out, in)` with a dual-form ridge solve when rank is
  the larger dimension;
- alternates continuous factor solves with weighted nearest-codeword
  assignment and sign-of-centroid table updates;
- resets the ADMM dual when the discrete table changes, freezes the fitted
  table halfway through the run, and uses the remaining iterations for
  fixed-set convergence;
- retains FP32 optimization state to avoid rejecting an over-complete arm
  because BF16 duals overflow;
- exports exact codebook-decodable signs and then runs the ordinary two-pass
  scale fit.

The primary arms use a fully arbitrary `2^k x 32` table independently fitted
for U and V. An efficient Cartesian-product approximation, two fitted
16-sign half-tables whose combined index is still `k` bits, was also measured
but is not used for the final format verdict.

`tools/probe_sign_word_codebook.py` provides exact bit accounting,
checkpointed scalar evidence, codebook utilization, convergence traces, and
raw/corrected-Fisher reconstruction metrics. Four CPU regressions cover
equal-budget rank selection, both decode forms, and exact constrained-factor
export.

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Matrix: block 12 `mlp.down_proj`, shape 1,152 x 6,912
- Importance: retained 256-sample corrected-CCE Fisher state, shrinkage 0.6
- Baseline: ordinary production ADMM, rank 970, 800 outer iterations,
  two-pass scale fit
- Candidate: arbitrary fitted 32-sign tables, `k in {8, 12}`, 800 outer
  iterations, two-pass scale fit
- Rank alignment: 32
- Budget: complete free-word signs and 16-bit scales versus fixed-width
  indices, 16-bit scales, and both full decode tables
- Seeds: 0 for both arms; seeds 1 and 2 repeat the closest k=12 arm

The canonical commands are:

```powershell
.\.venv\Scripts\python.exe tools\probe_sign_word_codebook.py `
  --model <pinned-snapshot>\model.safetensors `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\sign-word-codebook-probe\block12-down-r970-800-full-k12.json `
  --block 12 --projection down --baseline-rank 970 `
  --index-widths 12 --outer-iterations 800 `
  --codebook-update-interval 10 --codebook-freeze-fraction 0.5 `
  --codebook-mode full --assignment-batch-words 4096
```

The k=8 command differs only in index width, assignment batch 16,384, and
output name.

## Equal-bit arithmetic

| Arm | Rank | Rank/baseline | Actual BPW | Table bits | Unused bits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free 32-bit words | 970 | 1.000x | 1.000502 | 0 | 0 |
| Arbitrary k=8 | 3,840 | 3.959x | 0.998200 | 16,384 | 18,336 |
| Arbitrary k=12 | 2,464 | 2.540x | 0.989841 | 262,144 | 84,896 |

Both table costs are charged wholly to this one matrix. That is conservative
relative to amortizing a tensor-type table across 26 blocks. Conversely, a
dedicated per-matrix table is more flexible than the proposed shared table,
making the reconstruction comparison optimistic for the candidate.

The k=12 aligned budget fragment cannot fund the next 32 ranks. Fully
amortizing its table would raise the aligned rank only from 2,464 to 2,560,
so table amortization is not large enough to explain the measured gap.

## Result

### Primary arbitrary-table arms

| Arm | Weighted RMSE | RMSE change | Weighted error-energy change | Raw RMSE change |
| --- | ---: | ---: | ---: | ---: |
| Free words, rank 970 | 0.533293 | - | - | - |
| Arbitrary k=8, rank 3,840 | 0.672365 | **+26.08%** | +58.96% | +25.48% |
| Arbitrary k=12, rank 2,464 | 0.582675 | **+9.26%** | +19.38% | +9.22% |

More iterations did not rescue either arm. At 400 iterations, arbitrary k=8
was already 23.91% worse and arbitrary k=12 was 9.93% worse. The 800-iteration
run slightly improved k=12 and made k=8 worse.

The k=12 result is not caused by dead entries or a collapsed table. U and V
both use all 4,096 entries. Their empirical index entropies are 11.966 and
11.994 bits, respectively, very close to the 12-bit maximum. The table is
fully exercised; its word constraint is simply too expensive in
reconstruction quality for the extra 2.54x components to repay.

### Seed repeat

| Seed | Free-word RMSE | k=12 RMSE | k=12 change |
| ---: | ---: | ---: | ---: |
| 0 | 0.533293 | 0.582675 | +9.2597% |
| 1 | 0.533245 | 0.582603 | +9.2562% |
| 2 | 0.533205 | 0.582606 | +9.2649% |

The mean regression is 9.2603%, spanning only 0.0087 percentage points across
three deterministic seeds. Initialization noise does not explain the result.

### Product-table diagnostic

The cheaper product-table assignment was a useful implementation diagnostic,
not the primary test. At 400 iterations its least-bad point was k=14,
rank 1,888, at +10.91% weighted RMSE. Product k=12 was +27.75%. Continuing to
the arbitrary tables materially improved the result, but not enough to beat
free signs.

Early k=8 product runs also exposed unstable moving-set ADMM behavior. Freezing
the table and resetting stale duals corrected the numerical failure. The
arbitrary k=8 arm then converged to a finite result and still lost decisively,
so the final rejection is not based on the divergence.

## Runtime implication

The storage exchange also produces unattractive arithmetic intensity:

- rank-970 factorized work is about 0.98x a dense matrix-vector product;
- k=12 rank 2,464 is about 2.50x dense work before LUT decode;
- k=8 rank 3,840 is about 3.89x dense work before LUT decode.

A specialized LUT kernel could reduce sign bandwidth, but it cannot remove
the extra factor accumulations. Since neither arm passes reconstruction, no
decode-throughput kernel or packed schema is justified.

## Decision

Reject 8- and 12-bit sign-word codebooks for the proposed equal-1-BPW
down-projection operating point. Do not proceed to splice KL, a numbered
complete run, artifact changes, GGUF changes, or runtime implementation.

This is a bounded rejection, not a proof against every codebook:

- it covers the proposal's exact anchor matrix and bit widths;
- it uses diagonal corrected-Fisher fitting rather than a held-out functional
  objective;
- it does not test learned variable-width codes, multiword vector
  quantization, or a jointly trained model-wide codebook.

Those broader variants require a new funding/runtime argument. The tested
fixed-width mechanism already receives an optimistic dedicated table and
fails both weighted and unweighted reconstruction before the functional gate.
The useful lesson is that component breadth does not compensate for removing
independent sign-word freedom at this exchange rate.
