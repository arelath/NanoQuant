# Coupled MLP Operator-Scale Refit

## Question

The dominant-factor format search found no better independent matrix
encoding than the k10, two-correction mixed representation. The remaining
question was whether `gate_proj` and `up_proj` could be optimized as the
coupled operator they actually form:

```text
silu(gate_proj(x)) * up_proj(x)
```

Independent weighted matrix RMSE does not reward errors that cancel after
the SiLU and multiplication. This probe keeps every sign, rank, codebook
entry, correction, and free row fixed, and refits only the existing
per-output-channel scales.

## Method

For each intermediate channel, the probe searches a bounded positive gate
multiplier. At each gate point it solves the paired up multiplier by least
squares against the teacher's post-SiLU product. The fit objective is:

```text
||silu(g_t) * u_t - silu(a * g_q) * (b * u_q)||²
```

where `a` and `b` are independent channel vectors. The unchanged scale
vectors are retained as an identity fallback, so the fit set cannot regress.

This has zero format-bit cost. Gate/up are factorized in transposed
orientation, where an original output-channel multiplier is an update to
the existing 16-bit `scale_pre` value. It does not add a tensor, index,
codebook entry, or runtime operation. The present splice probe materializes
the adjusted dense weights for evaluation; a packed implementation should
fold the multipliers into `scale_pre`.

## Protocol

- Model: pinned `google/gemma-3-1b-it`
- Projection group: block-12 `gate_proj` plus `up_proj`
- Free-word control: rank 970
- Mixed candidate: k10, two corrections, rank 1,344, 256 free rows
- Factorization: 800 outer iterations
- Operator fit: WikiText sequences 48 through 51
- Operator validation: sequences 52 through 55
- Initial KL screen: sequences 0 through 11
- Disjoint KL confirmation: sequences 24 through 47
- Bootstrap: paired sequence resampling, 10,000 resamples, 95% interval

The fit, operator validation, initial KL screen, and confirmation inventories
are mutually disjoint.

## Bound search

Three multiplier ranges were screened. All improve the mixed representation
with confidence on the initial 12-sequence KL window:

| Bounds, gate / up | Mixed operator-validation NRMSE | Mixed KL before | Mixed KL after | KL change |
| --- | ---: | ---: | ---: | ---: |
| 0.75-1.25 / 0.5-2 | -5.132% | 0.053456 | 0.049535 | -7.334% |
| 0.5-1.5 / 0.25-4 | -5.531% | 0.053456 | 0.048958 | -8.414% |
| **0.25-2 / 0.1-8** | **-5.825%** | 0.053456 | **0.048837** | **-8.641%** |

The wide range had the best held-out operator error and mixed KL, so it was
selected before the disjoint confirmation. Some fitted channels reach the
gate bounds and the largest up multiplier is about 7.66. Those values are
representable by the existing scale field, but the boundary contact means
the range is a tested policy rather than evidence that the unconstrained
optimum has been found.

## Disjoint confirmation

The wide refit transfers to the mixed representation on sequences 24
through 47:

| Representation | KL before | KL after | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| Free words | 0.072037 | 0.072874 | +1.161% | [-0.004578, +0.007557] |
| **Mixed k10** | 0.064676 | **0.059676** | **-7.731%** | **[-0.007262, -0.002936]** |

The free-word refit is inconclusive and slightly worse on this independent
window. The mixed refit is not merely exploiting a generic calibration
benefit: its improvement is specific to the mixed factor errors.

After refitting both arms, mixed KL is 18.110% below free-word KL on the
confirmation window, with paired delta interval
[-0.023112, -0.004937].

## Combined evidence

Combining the initial screen and disjoint confirmation gives 36 sequences
and 18,396 next-token targets:

| Comparison | KL before | KL after | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| Mixed, refit minus original | 0.060936 | **0.056063** | **-7.997%** | **[-0.006531, -0.003301]** |
| Free words, refit minus original | 0.066479 | 0.065458 | -1.536% | [-0.004721, +0.003779] |
| Mixed refit minus free refit | 0.065458 | **0.056063** | **-14.353%** | **[-0.016507, -0.003609]** |

The combined mixed refit also beats the unrefitted free-word control by
15.668%, with paired delta interval [-0.013826, -0.007187].

## Decision

Accept coupled output-scale refitting as a strong research candidate for
mixed gate/up pairs.

- It buys a confirmed 7.7-8.0% relative KL reduction for the block-12 mixed
  splice without spending more bits or runtime arithmetic.
- Do not apply it unconditionally to free-word factors; that arm did not
  pass disjoint confirmation.
- Do not yet claim a model-wide gain. Block selection remains unresolved,
  and the packed mixed overlay still needs transposed gate/up orientation
  support.
- Use held-out post-SiLU product error, not independent matrix RMSE, as the
  next 26-block screening metric. Any multi-block policy must be declared
  before its disjoint splice evaluation.

The next useful extension is to refit `down_proj` against the resulting
student gated activation. That adds another coupled degree of freedom while
still changing only existing scale values.

## Evidence

- `evidence/m4/sign-word-codebook-probe/operator-refit/block12-gate-up-refit-wide-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/operator-refit/block12-gate-up-refit-wide-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/operator-refit/block12-gate-up-refit-conservative-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/operator-refit/block12-gate-up-mixed-k10-r1344-free256-800-refit-fit48-val52-kl0-12.json`
