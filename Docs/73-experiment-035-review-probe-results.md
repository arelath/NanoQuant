# Experiment 035 Review Probes and Revised Experiment 036 Direction

## Status

Completed on 2026-07-31. These probes implement the highest-priority checks
from [72-experiment-035-036-review.md](72-experiment-035-036-review.md) on the
retained Experiment 035 artifacts. Experiment 036 was paused before these
checks. Its uniform-control run has 11 durable blocks through block 10; the
candidate campaign has not started and must not resume under the original
six-block-seed policy.

Subsequent audit in
[74-block25-anomaly-and-topk-tail-mass-audit.md](74-block25-anomaly-and-topk-tail-mass-audit.md)
changes the interpretation and priority. The block-25 result is a valid
full-model recovery result, but it is not a block-local reconstruction repair.
It compensates a probability-mass error created by conditional top-64 KD. The
next experiment should repair that objective on the retained pre-KD state
before this refit is promoted as a production stage.

## Conclusions

1. The block-25 defect is systematic across Experiments 022 and 035.
2. Experiment 022's block-25 multiplier values transfer surprisingly well,
   but a fresh refit on Experiment 035's own factors is significantly better.
3. The full six-block transplant is inferior to block 25 alone. Winsorizing
   extreme seed values does not repair its composition interactions.
4. Several other blocks improve independently, but every tested addition after
   block 25 regresses composed NLL. Independent screening cannot choose a
   deployed prefix.
5. Global top-k KD creates or amplifies the block-25-class defect and actively
   penalizes its correction. The objective is not merely insensitive.
6. A fresh, factor-compatible block-25 refit reduces retained perplexity from
   `221.035336` to `148.747976` at zero represented bytes and improves the mean
   task score without regressing any retained task relative to the same-run
   post-KD baseline.

The transferable production asset is therefore the post-KD refit and
composition-gating procedure. The original six-block seed remains useful as
evidence and a transfer control, not as the next default pipeline policy.

## Seed transfer on Experiment 035

The raw Experiment 022 initializer was applied covariantly to Experiment 035's
own post-KD factor components. A 24x512 screen on WikiText validation sequences
152-175 and a 48x512 confirmation on sequences 176-223 agreed:

| Independent arm | Confirmation NLL delta | Confirmation KL delta | Decision |
| --- | ---: | ---: | --- |
| block 25 | **-0.395554** | **-0.575140** | transfers strongly |
| block 24 | -0.068824 | -0.079699 | transfers independently |
| block 23 | -0.039808 | -0.041725 | transfers independently |
| block 17 | -0.007581 | -0.015782 | small confirmed transfer |
| block 18 | +0.001647 | +0.000731 | reject |
| block 0 | +0.007513 | +0.011567 | reject |
| full raw seed | -0.329977 | -0.542747 | worse than block 25 |
| full winsorized seed | -0.314709 | -0.536496 | worse than raw/full and block 25 |

Winsorization restored the original fit bounds and clipped 3,670 of 124,416
values, including the 100x and 0.015625x extrema. It did not improve the full
composition. The failure is primarily block interaction, not only extreme
channels.

## Fresh post-KD defect screen

Teacher-context refits were fitted on test sequences 380-383, locally validated
on 384-387, and functionally confirmed on 412-459. This isolates the simple
procedure recommended by the review:

| Fresh teacher-context arm | NLL delta | 95% interval | KL delta | 95% interval |
| --- | ---: | --- | ---: | --- |
| block 25 joint | **-0.435423** | [-0.454625, -0.416287] | **-0.648702** | [-0.667895, -0.629424] |
| block 24 joint | -0.072489 | [-0.074971, -0.069974] | -0.082838 | [-0.085579, -0.080143] |
| block 23 joint | -0.034475 | [-0.036689, -0.032250] | -0.041104 | [-0.043320, -0.038716] |
| block 21 joint | -0.027188 | [-0.029460, -0.024935] | -0.032334 | [-0.034578, -0.030159] |
| block 17 joint | -0.020583 | [-0.023944, -0.017322] | -0.035447 | [-0.039048, -0.031961] |
| block 4 joint | -0.019755 | [-0.022219, -0.017168] | -0.024899 | [-0.027724, -0.022107] |
| block 18 joint | -0.001133 | [-0.003635, +0.001470] | -0.002632 | [-0.005265, -0.000034] |
| block 0 output | +0.007081 | [+0.004788, +0.009258] | +0.015127 | [+0.012757, +0.017462] |
| block 19 joint | +0.012326 | [+0.009797, +0.014849] | +0.014083 | [+0.011947, +0.016209] |

Fresh block 25 confirmed again on untouched validation sequences 248-295:
NLL `-0.447993`, interval `[-0.474445,-0.424476]`; KL `-0.676585`, interval
`[-0.701717,-0.652327]`.

The fresh factor-compatible block-25 state is byte-neutral. On the same
validation 104-151 inventory, fresh minus transplanted block 25 is:

- NLL `-0.004190`, interval `[-0.007324,-0.001129]`;
- KL `-0.038538`, interval `[-0.041311,-0.035716]`.

Thus the values contain a systematic transferable component, but refitting the
fresh campaign remains measurably superior.

## Composition gate

Independent gains were composed in descending confirmed-effect order on fresh
validation sequences 224-247:

```text
25 -> 24 -> 23 -> 21 -> 17 -> 4
```

Block 25 alone improved NLL by `-0.413972`. Every addition regressed marginal
NLL. The first addition, block 24, changed NLL by `+0.020822`, interval
`[+0.017650,+0.024308]`. The longest accepted prefix is therefore block 25
alone. This reproduces the interaction warning from the earlier six-block work
and rules out selecting a production policy from independent arms.

## When the defect appears

The same fresh block-25 fitting protocol was run before and after Experiment
035 global KD:

| State | Baseline NLL | Baseline KL | Correctable NLL delta | Correctable KL delta |
| --- | ---: | ---: | ---: | ---: |
| pre-KD | 4.846180 | 1.667282 | -0.012619 | -0.006930 |
| post-KD | 4.761640 | 2.015799 | **-0.435423** | **-0.648702** |

Global KD improves causal NLL by about 0.085 while increasing full-vocabulary
teacher KL by about 0.349. Across that boundary the separable-correctable
block-25 NLL defect grows about 35x and its KL defect grows about 94x.

The exact retained first-epoch top-k cache prefers the defective state:

| Arm minus post-KD | Retained top-k KL delta | Full-vocabulary behavior |
| --- | ---: | --- |
| transplanted block 25 | +0.083679 | large NLL/KL improvement |
| fresh block 25 | **+0.105220** | larger NLL/KL improvement |

The global objective therefore actively opposes the desired correction. A
second continuation using the same top-k objective should not follow the fresh
refit until the objective is changed and held-out NLL selection is implemented.

## Complete retained benchmark

The factor-compatible fresh block-25 overlay completed WikiText 64x128 and 200
examples for each retained task:

| Benchmark | Same-run post-KD | Fresh block 25 | Change |
| --- | ---: | ---: | ---: |
| WikiText perplexity | 221.035336 | **148.747976** | -32.706% |
| PIQA `acc_norm` | 0.615 | **0.625** | +0.010 |
| ARC-Easy `acc_norm` | 0.370 | 0.370 | 0.000 |
| ARC-Challenge `acc_norm` | 0.235 | **0.245** | +0.010 |
| HellaSwag `acc_norm` | 0.430 | **0.445** | +0.015 |
| WinoGrande `acc` | 0.540 | **0.545** | +0.005 |
| BoolQ `acc` | 0.645 | 0.645 | 0.000 |
| Mean primary task score | 0.472500 | **0.479167** | +0.006667 |

The task inventory provides a favorable out-of-distribution signal, but it is
not a replacement for the review's proposed pinned non-WikiText language-model
NLL/KL gate. That remains a production-stage follow-up.

## Revised next step

Do not resume Experiment 036 unchanged. Preserve its control commits for later
reuse, but replace the candidate policy with a per-campaign post-KD refit stage:

1. fit teacher-context block-25 scales on the active run's own post-KD factors;
2. encode them through existing factor terms with zero byte delta;
3. require a private 24-sequence marginal screen and untouched 48-sequence
   confirmation;
4. permit additional blocks only through sequential composition gates;
5. skip the current top-k continuation after the refit;
6. add held-out full-vocabulary NLL/KL and a pinned non-WikiText secondary gate;
7. complete exact logical/packed export and the retained quality benchmark.

## Evidence

- `evidence/035/experiment035-seed-transfer-validation152-24x512-kl.json`
- `evidence/035/experiment035-seed-transfer-validation176-48x512-kl.json`
- `evidence/035/experiment035-postkd-block25-teacher-context-fit380-val384-confirm412-48x512.json`
- `evidence/035/experiment035-postkd-blocks0-17-18-teacher-context-fit380-val384-confirm412-48x512.json`
- `evidence/035/experiment035-postkd-blocks4-19-21-teacher-context-fit380-val384-confirm412-48x512.json`
- `evidence/035/experiment035-postkd-blocks23-24-teacher-context-fit380-val384-confirm412-48x512.json`
- `evidence/035/experiment035-fresh-teacher-prefix-screen-validation224-24x512-kl.json`
- `evidence/035/experiment035-fresh-teacher-block25-confirm-validation248-48x512-kl.json`
- `evidence/035/experiment035-prekd-block25-teacher-fit380-val384-confirm412-48x512.json`
- `evidence/035/experiment035-block25-topk-objective-sensitivity.json`
- `evidence/035/experiment035-postkd-block25-fresh-factor-confirm104-48x512-kl.json`
- `evidence/035/experiment035-postkd-fresh-block25-factor-quality.json`
