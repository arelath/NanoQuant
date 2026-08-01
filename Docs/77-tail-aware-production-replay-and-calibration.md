# Tail-Aware Production Replay and Final-Norm Calibration

## Decision

The retained Experiment 035 production replay passes after combining the
256-step, mass-weight-0.5 top-k-plus-tail checkpoint with a 1.06 scale folded
into Gemma's final RMSNorm effective weight. The final candidate is a normal,
hash-addressed, factorized run at unchanged effective BPW. It is ready to be
the candidate policy in a fresh matched campaign; it is not evidence for a
universal coefficient or scale.

Experiment 036 remains paused. No block-25 refit or transplanted MLP-scale seed
is part of this candidate.

## Production replay correctness

The first production replay was rejected. Its teacher top-64 samples, token
positions, indices, and selected logits matched the retained analysis cache,
but it computed all 512 selected-token normalizers in one BF16 matrix
multiplication. The retained analysis used 128-token chunks. The normalizers
differed by mean absolute 0.01106 and maximum 0.25, enough to change training.

Production now preserves the legacy whole-selection top-k pass while computing
full-vocabulary log-normalizers in bounded token/vocabulary chunks. The
normalizer algorithm is bound into the tail-aware protocol identity. In the
corrected replay all 256 teacher batches match the retained targets byte for
byte, including normalizers.

The corrected interrupted arm stopped after epoch 1 and resumed in a separate
process. Its eight losses exactly match the retained analysis:

```text
2.552030600607395
2.389506570994854
2.250994972884655
2.237709905952215
2.2526894956827164
2.1965729855000973
2.2912366464734077
2.2556060776114464
```

The uninterrupted control produced the same teacher-epoch artifact identities
and the same final checkpoint
`sha256-c809526f608e4b42b8a52ef06b88295e30b503576de1c39f3621a08004d805ac`.
That content identity covers parameters, optimizer/Kahan state, scheduler
state, losses, and targets. The resumed arm committed global tuning artifact
`sha256-de61f2cc7d8507b664de818a2a3930a7713c03cf5dceb1336dc7e9fff3bffb75`.

## Why the original candidate still failed Gate C

The 16x128 monitor passed, but the predeclared 48x512 WikiText-validation slice
showed a stronger selected-mass collapse:

| Candidate | Epoch | NLL | Full KL | Tail KL | Teacher-top-64 mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pre-KD | - | 4.73300 | 1.53330 | 1.45174 | 0.81455 |
| Mass weight 0.5 | 8 | **4.37840** | **1.38151** | **1.31128** | 0.70838 |
| Mass weight 1.0 | 8 | 4.42936 | 1.39812 | 1.32520 | 0.73507 |
| Mass weight 2.0 | 8 | 4.51465 | 1.45280 | 1.37571 | **0.75072** |

Weight 2.0 was the first coefficient-only survivor, but its complete quality
benchmark was inferior: WikiText perplexity 206.75 and six-task mean 0.4500.
Increasing the training coefficient makes all 677 selected tensors spend
capacity on a largely global confidence error.

## Foldable confidence calibration

The mass deficit is substantially a logit-temperature error. Gemma computes
its final RMSNorm effective weight as `1 + weight`; replacing it with
`scale * (1 + weight)` scales the hidden state entering the tied output head.
The fold is therefore represented by
`new_weight = scale * (1 + old_weight) - 1`. It adds no inference operation and
reuses the existing 1,152-element BF16 auxiliary parameter.

The tight folded-scale sweep on the production checkpoint was:

| Scale | NLL | Full KL | Tail KL | Teacher-top-64 mass | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1.055 | 4.41386 | 1.36297 | 1.28904 | 0.74980 | fail |
| **1.060** | **4.41962** | 1.36264 | 1.28823 | **0.75396** | pass |
| 1.065 | 4.42190 | **1.36208** | **1.28741** | 0.75609 | pass |
| 1.075 | 4.43297 | 1.36191 | 1.28632 | 0.76335 | pass |

Scale 1.06 is the lowest tested survivor and preserves more NLL than the
larger scales. It was committed as a zero-copy fork at
`evidence/035/experiment035-topk-tail-production-v2-folded-scale1p06-run`.
The derived global-tuning artifact is
`sha256-9671c47a6eeb3c3ec37fe6ec1fb13e32cc48d1fb5ec5944f4711c9e4e4bfa48a`;
its protocol hash binds the base KD protocol, scale, parameter name, and
calibration algorithm.

Normal serialized reload gives NLL 4.41949, full KL 1.36204, tail KL 1.28766,
and mass 0.75418 on the 48x512 gate. The resident audit validates 708 artifacts
and all 26 blocks. Effective BPW remains 1.024494712.

## Complete quality benchmark

The authoritative result below comes through the ordinary hash-verifying
factorized loader, using 64x128 WikiText and 200 examples from each retained
task:

| Arm | WikiText PPL | PIQA | ARC-E | ARC-C | HellaSwag | WinoGrande | BoolQ | Task mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 96.46 | 0.720 | 0.605 | 0.395 | 0.585 | 0.625 | 0.810 | 0.6233 |
| Pre-KD compressed | 257.49 | 0.600 | 0.360 | 0.240 | 0.375 | 0.515 | 0.650 | 0.4567 |
| Uncalibrated 0.5 analysis | **178.36** | 0.605 | 0.365 | 0.220 | 0.400 | 0.540 | 0.645 | 0.4625 |
| Weight 2.0 | 206.75 | 0.595 | 0.365 | 0.220 | 0.380 | 0.510 | 0.630 | 0.4500 |
| **Production 0.5 + scale 1.06** | 191.79 | 0.595 | 0.360 | 0.235 | 0.395 | **0.555** | 0.645 | **0.4642** |

Calibration gives back some of the uncalibrated PPL gain to satisfy the mass
constraint, but it remains substantially better than pre-KD and weight 2.0.
It also has the best task mean among the serialized gate-compliant arms.

## Independent C4 confirmation

On the pinned C4 48x512 slice, the calibrated production model improves NLL
from 5.25731 to 5.03745 (perplexity 191.96 to 154.08) and teacher KL from
1.22227 to 1.10118. The paired candidate-minus-pre-KD intervals are:

- NLL: -0.21985, 95% interval [-0.23586, -0.20510];
- KL: -0.12109, 95% interval [-0.13221, -0.11018].

Both are entirely beneficial. The earlier fresh block-25 refit from the same
tail-aware checkpoint was entirely harmful on its untouched confirmation set
(NLL +0.05971 and KL +0.02022), so it remains rejected.

## Next experiment

The next numbered campaign should create one fresh Gemma frozen state and
branch it into matched conditional and tail-aware arms. The tail-aware arm must
use the explicit 0.5 coefficient, 256-step cap, and identity-bound final-norm
calibration policy. Checkpoint selection must use a held-out broad mass/NLL/KL
gate; the pinned C4 slice remains a final confirmation, not a tuning set.

The fresh campaign is not accepted until it completes artifact validation,
logical/packed reload, GGUF export, BPW/byte accounting, the complete quality
benchmark, and same-frozen-state comparison. A second model is still required
before treating either 0.5 or 1.06 as a general default.
