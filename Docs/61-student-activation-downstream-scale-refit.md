# Student-Activation Downstream Scale Refit

## Question

The coupled gate/up refit improves the post-SiLU product without adding
format bits. Once `down_proj` is also quantized, however, independent down
matrix fitting still assumes the teacher activation. The actual input at
runtime is the quantized student product:

```text
z_q = silu(gate_q(x)) * up_q(x)
```

This experiment asks whether existing down scale vectors can compensate for
that student activation before the down projection mixes its 6,912
channels.

## Method

All three block-12 MLP projections use the equal-bit mixed k10,
two-correction representation:

- gate and up are transposed for factorization;
- down remains in source orientation;
- rank is 1,344 with 256 free dominant-factor rows;
- the free-word control uses rank 970.

The confirmed wide gate/up operator refit is applied first. The downstream
input arm then minimizes:

```text
||y_t - (z_q * d) @ W_down_q.T||²
```

where `y_t` is the teacher gated-MLP output and `d` is a positive
6,912-element channel vector. A diagonal Hessian approximation
preconditions deterministic descent, every step uses backtracking, and the
identity vector is the rollback baseline. The accepted candidate uses 50
steps and bounds `0.25 <= d <= 4`.

This has zero additional storage and runtime arithmetic. In source-oriented
down factorization, `d` folds directly into the existing down
`scale_pre` vector.

Two cheaper or more elaborate alternatives were measured:

1. Fit only down's 1,152 output-channel scales analytically.
2. After fitting down input scales, also fit the output scales.

## Protocol

- Model: pinned `google/gemma-3-1b-it`
- Block: 12
- Factorization: 800 outer iterations
- Gate/up bounds: gate 0.25-2, up 0.1-8
- Down input bounds: 0.25-4
- Operator fit: WikiText sequences 48 through 51
- Operator validation: sequences 52 through 55
- Initial KL screen: sequences 0 through 11
- Disjoint KL confirmation: sequences 24 through 47
- Bootstrap: paired sequence resampling, 10,000 resamples, 95% interval

All fit, validation, and KL inventories are disjoint.

## Candidate screen

The output-scale-only arm improves held-out block-output NRMSE by 2.255%,
but mixed KL changes from 0.073937 to 0.074086, a 0.201% regression. It is
rejected without a confirmation run.

The higher-capacity input-scale solver behaves differently:

| Solver | Mixed KL before | Mixed KL after | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| 20 input steps | 0.073937 | 0.071820 | -2.864% | [-0.003790, +0.000338] |
| **50 input steps** | 0.073937 | **0.071882** | **-2.780%** | **[-0.003433, -0.000154]** |
| 50 input + output | 0.073937 | 0.072306 | -2.206% | [-0.003627, +0.001325] |

The 20-step point is numerically best on the small screen but inconclusive.
The 50-step input-only point crosses the statistical gate and is therefore
the candidate sent to disjoint confirmation. The extra output scales give
back part of the gain.

The 50-step solver accepts all iterations. Mixed held-out block-output
NRMSE changes from 0.369986 to 0.356894, a 3.539% reduction. Its fitted
input multipliers span 0.25 to 3.33, so some channels contact the lower
bound.

## Disjoint confirmation

The 24-sequence confirmation preserves the input-scale gain:

| Arm | KL | Change from gate/up operator refit | Paired 95% delta interval |
| --- | ---: | ---: | ---: |
| Mixed gate/up operator refit | 0.086150 | — | — |
| **Mixed + down input refit** | **0.084208** | **-2.254%** | **[-0.003185, -0.000617]** |
| Mixed + input/output refit | 0.083978 | -2.521% | [-0.003969, -0.000163] |

The extra output-scale stage also passes relative to the gate/up arm on this
window, but it was weaker on the initial screen. It needs to beat the
input-only arm directly to justify its extra fitting policy.

## Combined evidence

Combining the initial and confirmation inventories gives 36 sequences and
18,396 next-token targets:

| Comparison | KL before | KL after | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| **Mixed input refit minus gate/up refit** | 0.082079 | **0.080099** | **-2.412%** | **[-0.002971, -0.000897]** |
| Mixed joint refit minus gate/up refit | 0.082079 | 0.080087 | -2.427% | [-0.003448, -0.000369] |
| Joint minus input-only | 0.080099 | 0.080087 | -0.015% | [-0.000806, +0.000851] |
| Free-word input refit minus gate/up refit | 0.097323 | 0.094748 | -2.646% | [-0.004339, -0.000787] |

The joint arm's 0.015% advantage over input-only is noise. Input-only is
therefore the preferred representation. After input refitting, mixed KL is
15.461% below the corresponding free-word arm, with paired interval
[-0.022533, -0.008350].

## Decision

Accept the 50-step down input-channel refit as a block-12 research
candidate.

- It adds a confirmed 2.25-2.41% relative KL reduction after the coupled
  gate/up refit.
- It changes only existing down `scale_pre` values.
- Reject down output-scale-only refitting.
- Reject the additional output-scale stage because it does not beat
  input-only directly.
- Do not yet claim model-wide transfer. A broader screen needs a
  predeclared rule based on held-out gated-MLP output error, followed by
  multi-block disjoint KL confirmation.

The coupled sequence now has two independently confirmed zero-bit stages:
gate/up product-scale fitting followed by student-activation-aware down
input-scale fitting.

## Evidence

- `evidence/m4/sign-word-codebook-probe/downstream-refit/block12-mlp-mixed-k10-r1344-free256-800-gatewide-down025-4-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block12-mlp-mixed-k10-r1344-free256-800-gatewide-downinput20-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block12-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block12-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
