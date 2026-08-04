# Corrected-Codebook Payload Search

**Date:** 2026-08-03

**Status:** exact reconstruction search passed; held-out functional promotion
gate failed

## Question

[Mixed Codebook Binary-Factor Search](85-mixed-codebook-binary-factor-search.md)
showed that the retained control-then-tabu search can optimize the free signs
of the mixed codebook representation without invalidating its payload. That
screen deliberately froze the 1,088 coded rows.

This experiment searches the stored payload itself: can exact substitutions
of a coded word's 10-bit table index and two correction positions improve the
equal-bit result after the free signs have converged?

## Exact payload coordinate search

For one right-factor component and one 32-column word, every other factor sign
and all scales are fixed. The weighted reconstruction objective is then
separable across the 32 columns. The solver computes the exact objective cost
of flipping each current sign and uses it to score:

- every one of the 1,024 table entries; and
- the best two correction positions for each entry.

This avoids separately enumerating all `1,024 * C(32, 2) = 507,904` payloads.
The best candidates across the matrix are selected, rescored sequentially
against the current residual, accepted only on exact improvement, and followed
by a 64-pass common scale refit. Every outer pass has a full weighted-error
rollback gate. Final indices and correction positions are decoded and checked
for bit equality with the searched right factor.

The matched rank-970 free-word control receives its exact best arbitrary
32-sign coordinate proposal under the same objective and pass budget.

This is analysis-only. It changes no resident algorithm, artifact schema,
packed overlay, GGUF, or runtime path.

## Protocol

- Model: pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Matrix: block 12 `mlp.down_proj`, shape 1,152 x 6,912
- Objective: retained corrected-CCE Fisher state with 0.6 shrinkage
- Control: rank 970 free words
- Candidate: rank 1,344, free left factor, 256 free right rows, 1,088 k10
  two-correction right rows
- ADMM: 800 outer by five inner iterations
- Pre-payload search: the retained 64-pass control-then-tabu protocol on all
  free signs
- Payload scale refit: 64 passes after each accepted outer pass
- Seed: zero

The maximum-depth reconstruction command adds:

```powershell
--binary-search --payload-search `
--payload-search-passes 8 --payload-search-max-words 8192
```

to the canonical command in the previous experiment.

## Reconstruction result

The free-word payload stage is already saturated after binary search: it finds
only 12 single-sign changes and has no measurable final NRMSE effect.

| Payload budget | Candidate NRMSE | Payload NRMSE change | Error-energy change | Accepted words | Sign changes | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 passes / 2,048 words | 0.527839 | -0.0164% | -0.0329% | 2,844 | 5,708 | 8.07 s |
| 4 passes / 4,096 words | 0.527797 | -0.0244% | -0.0488% | 5,859 | 10,616 | 18.91 s |
| 8 passes / 8,192 words | **0.527780** | **-0.0276%** | **-0.0552%** | 7,111 | 12,618 | 31.91 s |

The maximum-depth arm evaluates 1.925 billion table-entry proposals across
2,322,432 word visits. Gains diminish with each budget doubling. Against the
equally searched free-word control at 0.532465 NRMSE, the final candidate is:

- 0.8798% lower in weighted NRMSE; and
- 1.7519% lower in weighted error energy.

Payload search therefore finds real representation-valid reconstruction
headroom, but it recovers only a small fraction of a percentage point.

## Held-out functional gate

The maximum-depth factors were evaluated on WikiText-2 windows 0 through 47,
48 sequences and 24,528 next-token targets. Separate 24-sequence runs were
also retained, but the decision uses matched 48-sequence evaluation geometry.

### Codebook versus free words

| Arm | KL nats/token | Change versus free words | Paired 95% interval |
| --- | ---: | ---: | ---: |
| Binary-search-only free words | 0.045156 | - | - |
| Binary-search-only codebook | 0.043403 | -3.88% | [-0.003826, +0.000523] |
| Payload-run free words | 0.045013 | - | - |
| Payload-searched codebook | 0.043580 | -3.18% | [-0.003404, +0.000753] |

Both codebook point estimates beat free words, but neither interval excludes
zero on this single-block 48-sequence gate.

### Incremental payload effect

Separate executions show a small common numerical shift even for the same
free reconstruction. The clean incremental comparison is therefore the paired
difference-in-differences:

```text
(payload codebook - payload free) -
(binary-only codebook - binary-only free)
```

The result is:

- point delta: **+0.000321 nats/token**;
- paired 95% interval: `[-0.000033, +0.000688]`;
- payload codebook KL versus binary-only codebook KL: **+0.409%**.

Positive is worse. The interval narrowly crosses zero, so this is not a
statistically conclusive regression. It is nevertheless a clear promotion
failure: the exact static-objective gain does not improve the held-out
functional point estimate and instead moves it in the wrong direction.

## Decision

Do not enable payload search in resident compression and do not launch a
complete codebook run from this result. Retain the exact solver, brute-force
oracle tests, payload round-trip checks, and splice integration as analysis
tools.

The mixed codebook representation itself remains supported by the earlier
multi-block and seed-stability evidence. This rejection is narrower: after
the free signs have converged, spending additional offline compute to improve
the diagonal matrix objective through codeword/correction substitutions does
not improve held-out model behavior.

Any further codebook optimization should move directly to a functional or
operator-level acceptance signal. Strengthening this static payload search is
not justified by the measured gain-to-compute curve or the functional result.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip2-k10-rank1344-free256-payload-search-v2.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip2-k10-rank1344-free256-payload-search-deep.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-r970-800-mixed-right-flip2-k10-rank1344-free256-payload-search-max.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-mixed-binary-search-splice-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-mixed-payload-search-splice-48.json`
