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

## Marginal decomposition and six-block rejection

Each of the four additions was exported separately and layered over the
confirmed four-block overlay on the exact 64x128 gate:

| Added block | Policy | Candidate perplexity | Change versus four-block candidate |
| ---: | --- | ---: | ---: |
| 11 | Operator | 272.115518 | -0.783972 |
| 20 | Joint | 279.906692 | +7.007203 |
| 21 | Joint | 272.222663 | -0.676826 |
| 22 | Operator | 276.503678 | +3.604189 |

Blocks 20 and 22 explain the eight-block reversal and are rejected. Blocks 11
and 21 appear beneficial on the exact gate, and their six-block composition
reaches 270.901199 perplexity, 1.085% below the Experiment 022 baseline.

That apparent gain does not generalize to the untouched sequences 308-319:

| Arm | KL | NLL |
| --- | ---: | ---: |
| Experiment 022 baseline | **1.548193** | **4.842978** |
| Six-block policy | 1.571254 | 4.849836 |

KL regresses 1.490%, interval [+0.004481, +0.039919]. Blocks 11 and 21 are
therefore exact-benchmark selection overfit and do not advance.

## Post-KD five-block restart

The pre-KD placement study was restarted from Experiment 022's final global-
tuned state. Block 23 was the only new marginal addition that passed an
untouched functional confirmation, producing the final mapping:

```text
0:output, 17:joint, 18:joint, 23:joint, 24:joint
```

Fresh post-KD functional windows confirmed the composition independently:

| Inventory | Baseline NLL | Candidate NLL | Change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Sequences 344-355, 12x512 | 4.655256 | 4.571150 | **-1.807%** | **[-0.099091, -0.070154]** |
| Sequences 356-379, 24x512 | 5.185110 | 5.097131 | **-1.697%** | **[-0.098016, -0.078506]** |

On the exact 64x128 retained protocol, the dense reference overlay reduces
perplexity from 228.590472 to 216.468670, a 5.303% reduction. Its paired NLL
change is -0.054486 with interval [-0.064715, -0.044709].

## Equal-size factor encoding

The dense overlay is not the representation claim. Each selected matrix is a
separable row/column rescale. The same transformation was encoded in the
existing NanoQuant terms:

- output multipliers replace `scale_post` and rescale existing outlier rows;
- input multipliers replace `scale_pre` and rescale existing outlier columns;
- a correction patch, when present, rescales its left/output and right/input
  factors correspondingly.

The selected Experiment 022 layers have BF16 outlier values and no patches.
The durable component overlay replaces 45 tensors occupying 599,040 bytes
with 599,040 bytes: zero changed shapes, zero changed dtypes, and zero payload-
byte delta. It is hash-bound to component SHA-256
`808ec038c948eadfd17b8a927a04d092fc5a199c439cec6b726efbedde822c13`,
the frozen commit identity, and global-tuning artifact
`sha256-86427bf8fcec089f56d925612642ff658ca925f54d75315703f10879dc5955cb`.

Loading those saved component tensors—not the fit's in-memory values—gives
216.533466 perplexity. The factor form differs from the dense reference by
only +0.000299 NLL, with interval [-0.000754, +0.001297], so the remaining
rounding difference is statistically indistinguishable on this protocol.

## Logical and packed proof

The component overlay was streamed through the complete deployment export:

| Gate | Result |
| --- | ---: |
| Logical blocks / layers / tensors | 26 / 130 / 910 |
| Logical weight bytes | 2,739,492,456 |
| Fresh logical validation | **Exact** |
| Packed weight bytes | **89,480,656** |
| Packed storage ratio vs logical | 3.2663% |
| Logical-to-packed conversion | **Exact** |
| Packed reference maximum absolute error | **0.0** |
| Packed reference output elements | 459,264 |

The packed byte count is identical to the original Experiment 022 packed
artifact, so its measured effective BPW remains **1.0244947118**. A model-
level evaluation loaded directly from the new packed artifact and reproduced
216.533466 perplexity exactly on the same token hash.

## Complete retained quality benchmark

The single-sequence packed probe is not the completion gate. The final packed
artifact was subsequently evaluated through the complete Experiment 022
quality protocol:

- 64 WikiText sequences of length 128, batch size 8;
- 200 retained examples each for PIQA, ARC-Easy, ARC-Challenge, HellaSwag,
  WinoGrande, and BoolQ;
- task batch size 4;
- resident BF16 reference and factorized packed candidate in the same run.

The BF16 scores reproduced Experiment 022 exactly. The WikiText token hash,
all task semantic keys, prompt hashes, and full task input identities also
match the retained benchmark. The newer result schema additionally records
the BOS, padding, and WikiText context policies explicitly.

| Benchmark | BF16 | Experiment 022 packed | New packed | Change vs 022 |
| --- | ---: | ---: | ---: | ---: |
| WikiText perplexity | 96.459609 | 228.550618 | **216.241794** | **-5.386%** |
| PIQA `acc_norm` | 0.720 | 0.605 | **0.630** | +0.025 |
| ARC-Easy `acc_norm` | 0.605 | **0.380** | 0.370 | -0.010 |
| ARC-Challenge `acc_norm` | 0.395 | 0.215 | **0.230** | +0.015 |
| HellaSwag `acc_norm` | 0.585 | 0.460 | **0.485** | +0.025 |
| WinoGrande `acc` | 0.625 | 0.520 | **0.525** | +0.005 |
| BoolQ `acc` | 0.810 | 0.635 | **0.640** | +0.005 |
| Mean primary task score | — | 0.469167 | **0.480000** | **+0.010833** |

Five of six task scores improve. ARC-Easy loses two correct answers out of
200; the other tasks gain a net fifteen, for a net gain of thirteen primary-
metric correct answers across the 1,200 retained task examples. These small
task movements should not be interpreted individually as statistically
decisive, but the benchmark rules out a broad task-quality tradeoff behind the
large perplexity improvement.

The official batch-8 perplexity (216.241794) differs slightly from the earlier
single-sequence probe (216.533466) because factorized BF16 execution is batch-
shape sensitive. Comparisons above use the protocol-matched batch-8 values.
The complete result reports `passed: true` and is bound to packed descriptor
SHA-256 `a39d6085b2e1e55a820d6071b964cc3b6a9fecd51f15cdca6bc6fea66634f4de`.

## Decision

Accept the post-KD five-block mapping as the new compressed-model candidate.
It is a materially higher-quality model than Experiment 022: perplexity is
5.275% lower at identical packed bytes and effective BPW, and the complete
factorized, logical, and packed representations agree through their required
parity gates. The complete retained quality benchmark independently confirms
a 5.386% protocol-matched perplexity reduction while improving five of six
task scores and the mean task score.

Further additions must be selected one at a time against this fixed post-KD
base on fresh language-functional windows. Isolated MLP-output gains and the
retained 64x128 benchmark are not selection sets.

## Evidence

- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-all26-fit48-val52-kl272-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-block0-fit48-val52-kl272-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-block0-output-fit48-val52-kl284-24.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-17-18-24-policy-fit48-val52-kl272-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-17-18-24-policy-fit48-val52-kl284-24.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-17-18-24-policy-direct64x128.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-11-17-18-20-21-22-24-policy-fit48-val52-kl272-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-11-17-18-21-24-policy-direct64x128.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-blocks0-11-17-18-21-24-policy-fit48-val52-kl308-12.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-blocks0-17-18-23-24-policy-nll344-12x512.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-blocks0-17-18-23-24-policy-nll356-24x512.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-component-replay-direct64x128.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-blocks0-17-18-23-24-factor-components-v2/manifest.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-factor-components-v2-logical-validation.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-factor-components-v2-packed-validation.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-factor-components-v2-packed-direct64x128.json`
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-factor-components-v2-packed-quality.json`
