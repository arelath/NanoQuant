# Tuned Frozen MLP Scale Placement

## Question

The complete sign-word MLP overlay failed because it discarded resident
tuning. This experiment instead starts from Experiment 022's immutable tuned
pre-KD factors and changes only transformations representable by existing
factor scale vectors:

- gate/up output scales;
- down input scales;
- down output scales;
- their input/output joint form.

No rank, sign, outlier, patch, or physical payload is added.

## Protocol

- Base: retained Experiment 022, factorized backend, pre-global-KD.
- Dense teacher: pinned Gemma 3 1B revision.
- Scale fit: WikiText sequences 48-51.
- Scale validation: sequences 52-55.
- Functional screen: sequences 272-283, length 512.
- Functional confirmation: sequences 284-307, length 512.
- Direct gate: exact retained 64x128 protocol, token hash
  `sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`.

## Uniform placement failure

Applying each policy to all 26 MLP blocks regresses held-out KL:

| Policy | KL | Change from baseline |
| --- | ---: | ---: |
| Baseline | **1.647205** | — |
| Output | 1.769 | +7.4% |
| Input | 1.79 | +8.7% |
| Operator | 1.83 | +11.1% |
| Joint | 1.97 | +19.6% |

Every block improves its isolated held-out MLP-output objective. Their uniform
composition nevertheless fails, again proving that local reconstruction
improvement is not an additive placement rule.

## Block 0 transfer

Block 0 is the strongest independent validation candidate. Operator plus down
output refit passes both functional inventories:

| Inventory | Baseline KL | Candidate KL | Change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Screen | 1.647205 | 1.631261 | **-0.968%** | **[-0.021641, -0.010076]** |
| Confirmation | 1.504494 | 1.497168 | **-0.487%** | **[-0.010723, -0.003709]** |

On the exact retained perplexity protocol, block 0 improves Experiment 022
from 273.872886 to 273.727247, a 0.0532% reduction.

## Four-block policy

The next predeclared tier selects the four strongest independent validation
placements:

```text
0:output, 17:joint, 18:joint, 24:joint
```

It passes both functional inventories:

| Inventory | Baseline KL | Candidate KL | Change | NLL change |
| --- | ---: | ---: | ---: | ---: |
| Screen | 1.647205 | 1.606150 | **-2.492%** | -0.032917 |
| Confirmation | 1.504494 | 1.463723 | **-2.710%** | -0.035406 |
| Combined 36 | 1.552065 | **1.511199** | **-2.633%** | -0.034576 |

The combined paired interval is [-0.050813, -0.030038].

The exact retained perplexity improves from 273.872886 to 272.899489:

- absolute change: -0.973397;
- relative change: **-0.3554%**;
- no additional representation bits.

## Eight-block rejection

Adding the next four validation-ranked placements produces:

```text
11:operator, 20:joint, 21:joint, 22:operator
```

The eight-block composition regresses screen KL by 2.043% and NLL by
0.018439. It is rejected without confirmation. The failure establishes a
sharp composition boundary rather than a monotonic depth trend.

## Decision

Accept the four-block mapping as the current zero-bit tuned-factor candidate.
It is the first sign-word follow-up that improves the actual retained model,
but its 0.36% perplexity gain is not yet a substantially higher-quality model.

Next search additions one at a time against this fixed base. Candidate
selection must use language-functional behavior; isolated MLP-output gains
alone are insufficient.

## Evidence

- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-all26-fit48-val52-kl272-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-block0-fit48-val52-kl272-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-block0-output-fit48-val52-kl284-24.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-17-18-24-policy-fit48-val52-kl272-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-17-18-24-policy-fit48-val52-kl284-24.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-17-18-24-policy-direct64x128.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-11-17-18-20-21-22-24-policy-fit48-val52-kl272-12.json`
