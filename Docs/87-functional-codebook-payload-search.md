# Functional Codebook Payload Search

**Date:** 2026-08-03

**Status:** exact fit/held-out operator gate passed; incremental model-quality
promotion gate failed

## Question

The static payload search in
[Corrected-Codebook Payload Search](86-codebook-payload-search.md) improved the
diagonal reconstruction objective but slightly worsened the 48-sequence
functional point estimate. This experiment asks whether actual MLP activations
can select safer codeword and correction substitutions.

## Functional acceptance

The static solver still generates exact 32-sign payload proposals. For each
outer pass, the functional variant:

1. keeps at most the 256 best static proposals;
2. evaluates exact `down_proj` output SSE on real captured MLP inputs;
3. retains only proposals that improve both a fit and a disjoint held-out
   window;
4. ranks survivors by held-out improvement and rescans them sequentially
   against the current residual; and
5. rolls back the full outer pass if the common 64-pass scale refit does not
   improve both functional windows as well as the static objective.

The exact update uses the rank-one output change induced by a right-factor
word, so it does not materialize a candidate dense weight for every proposal.
Final corrected-code payloads still undergo exact decode-equality validation.
The matched free-word arm uses the same gate.

## Protocol

- Pinned model, revision, corrected-CCE Fisher state, ranks, bit budget, ADMM,
  binary-factor search, seed, and 48-sequence KL window are identical to the
  static payload experiment.
- Matrix: block 12 `mlp.down_proj`, shape 1,152 x 6,912.
- Functional fit: WikiText-2 sequences 48 through 51.
- Functional validation: sequences 52 through 55.
- Final splice gate: sequences 0 through 47, disjoint from both functional
  search windows.
- Functional shortlist: 256 word proposals per outer pass.

## Search result

| Arm | Accepted words | Sign changes | Fit SSE change | Held-out SSE change |
| --- | ---: | ---: | ---: | ---: |
| Free words | 2 | 2 | -0.0028% | -0.0025% |
| Corrected codebook | 206 | 526 | -0.1897% | -0.1935% |

The codebook solver ranked 1,536 functional candidates and accepted five
outer passes. Its weighted NRMSE moved from 0.527926 to 0.527911. This is much
more conservative than static-only payload search, which accepted 7,111 words
and reached 0.527780 NRMSE.

The independently captured held-out operator signal therefore does what was
intended: it rejects most small diagonal-objective changes while preserving a
measured improvement on unseen MLP activations.

## 48-sequence splice gate

| Search arm | Free KL | Codebook KL | Codebook minus free | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Binary search only | 0.045156 | 0.043403 | -3.88% | [-0.003826, +0.000523] |
| Static payload | 0.045013 | 0.043580 | -3.18% | [-0.003404, +0.000753] |
| Functional payload | 0.045003 | 0.043529 | -3.27% | [-0.003541, +0.000813] |

The codebook point estimate remains better than free words, but its interval
still crosses zero. The clean incremental comparison against binary-search-only
factors is the paired difference-in-differences:

```text
(functional-payload codebook - functional-payload free) -
(binary-only codebook - binary-only free)
```

It is **+0.000280 nats/token**, with a paired 95% interval of
`[-0.000014, +0.000594]`. Positive is worse. Functional acceptance reduces
the adverse static-payload shift by 0.000041 nats/token, but it does not reverse
it and neither comparison is statistically conclusive.

## Decision

Do not promote functional payload search into resident compression and do not
expand this candidate to the five-block gate. The predeclared block-12 gate
requires an incremental held-out improvement over binary-search-only factors;
the point estimate still regresses.

Retain the functional acceptance implementation and its rejection tests as an
analysis tool. The result also narrows the diagnosis: diagonal payload updates
can be filtered into changes that generalize to disjoint local MLP outputs,
but that local improvement still does not improve downstream model KL.

## Evidence

- `evidence/m4/sign-word-codebook-probe/block12-down-mixed-functional-payload-search-splice-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-mixed-binary-search-splice-48.json`
- `evidence/m4/sign-word-codebook-probe/block12-down-mixed-payload-search-splice-48.json`
