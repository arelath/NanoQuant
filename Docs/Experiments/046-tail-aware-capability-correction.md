# Experiment 046: Tail-Aware Capability Correction

## Status

Completed as exploratory evidence only. The correction trajectory was measured,
but Experiment 046 cannot produce a formal survivor or promote a model because
its execution did not satisfy the repository's precommit and slice-ordering
requirements. No checkpoint is materialized and no production policy changes.

## Question

Experiment 045 rejected earlier checkpoint selection: tail-aware NLL and KL
continued improving through epoch 8. Experiment 044 nevertheless trails the
Experiment 042 correction-plus-fold endpoint by 4.94% packed perplexity.
Experiment 046 asks whether the already implemented one-sided correction has a
real NLL/full-KL benefit when initialized from the coherent tail-aware epoch-8
state, rather than using selected mass as an acceptance target.

This is a retained-state analysis. It does not reopen factorization, change the
tail coefficient, apply a norm fold, or introduce a block refit.

## Frozen treatment

- initializer: Experiment 044 epoch-8 global tuning, exactly 256 primary KD
  steps;
- correction objective: conditional top-64 plus one-sided batch mass deficit;
- minimum teacher-mass ratio: 0.8;
- mass loss weight: 2.0;
- learning rate: 1e-5;
- four epochs of at most 32 batches, yielding checkpoints at 32, 64, 96, and
  128 correction steps;
- fresh optimizer and 128-step cosine horizon;
- the Experiment 044 calibration tokens and immutable teacher cache.

These values reproduce the native correction geometry already validated by
Experiments 038-042. They are diagnostic in this new initializer regime, not
promoted point values.

## Development monitor and selection

WikiText validation offset 468, 24x512, token hash
`sha256:4328c22c4fba37751ea26660b60505cf10203ad1fc803d386d60beb521efd919`
is reserved as `experiment046-wikitext-validation468-24x512`. It is retired
after first model evaluation and can never become a final gate.

The epoch-0 tail-aware initializer is the baseline. A correction checkpoint is
eligible only if paired 95% upper bounds are below zero for NLL, full KL, and
top-64-plus-tail KL versus epoch 0. Among eligible checkpoints, keep those
within 0.02 nats of the eligible minima for all three metrics and select the
earliest. Selected mass is reported with uncertainty but cannot select or
reject an arm.

The first probe opened the reserved WikiText monitor and then exposed that the
probe's monitor receipt retains aggregate means but not per-sequence values.
It therefore cannot satisfy the frozen uncertainty rule and cannot select a
checkpoint. No arm was removed or changed in response to those means. Before
further model evaluation, all four frozen checkpoints were carried to a new
paired development slice: C4 validation offset 248, 48x512, token hash
`sha256:279b1e708c6fb52401e514bf8aff4371b762b21a36c5f6185bf15cc64a749861`,
reserved as `experiment046-c4-validation248-48x512`. The same selection rule
applies there. If a checkpoint survives, its final C4 gate must use another
new interval beginning at offset 296 or later.

The C4 evaluator retains paired NLL and full KL but not the top-64-plus-tail
decomposition required by the frozen rule. Before selecting an arm, the final
11 non-overlapping 512-token windows available in the WikiText validation
stream are therefore reserved at offset 492, token hash
`sha256:e12c4f3345e1fc6f3c66f48edbd0f90794d095ff96929a903fb7f57fc788ef58`,
under `experiment046-wikitext-validation492-11x512`. All four correction
checkpoints advance unchanged. Eligibility uses paired C4 NLL/full-KL bounds
and the paired WikiText top-64-plus-tail-KL bound; the 0.02 plateau tolerance
is unchanged. Neither slice alone can select the checkpoint.

## Follow-up gates

Only a selected capability-improving checkpoint advances. Without retuning it
must then:

1. improve the ordinary 64x128 WikiText result over Experiment 044 and reach
   Experiment 042's predeclared 2% relative bound;
2. pass a newly reserved C4 48x512 NLL/full-KL non-regression gate that does
   not overlap the offset-248 development slice;
3. show no established regression on the 1,000-example six-task guardrail;
4. materialize as an immutable global-tuning artifact at unchanged BPW and
   represented factor bytes.

Failure on the development monitor rejects this exact correction. Passing it
does not authorize a fixed production recipe; it only justifies the untouched
follow-up gates and development of a genuinely adaptive stopping rule.

## Exploratory result

The four correction checkpoints reached exactly 32, 64, 96, and 128 optimizer
steps. Against the immutable 256-step Experiment 044 initializer, the aggregate
WikiText development monitor was:

| Arm | NLL | Full KL | Top-k + tail KL | Selected mass |
| --- | ---: | ---: | ---: | ---: |
| initializer | 4.683998 | 1.461649 | 1.376146 | 0.676949 |
| correction epoch 1 | 4.656220 | **1.391468** | **1.304204** | **0.728775** |
| correction epoch 2 | 4.611465 | 1.438141 | 1.354148 | 0.653534 |
| correction epoch 3 | **4.597921** | 1.401384 | 1.317034 | 0.680343 |
| correction epoch 4 | 4.597986 | 1.397513 | 1.313050 | 0.682372 |

The trajectory is not monotonic. Epoch 1 gives the best distributional means,
epoch 3 gives the best NLL, and epochs 3-4 are the only aggregate arms within
0.02 nats of all three checkpoint minima. Selected mass is reported only as a
calibration diagnostic and does not decide this result.

A 48x512 C4 diagnostic was then opened prematurely, before a valid WikiText
selection existed. Relative to the initializer, paired C4 NLL and full-KL
deltas were:

| Arm | NLL delta, paired 95% interval | KL delta, paired 95% interval |
| --- | --- | --- |
| correction epoch 1 | +0.010777 `[+0.002018, +0.020972]` | +0.007941 `[+0.001025, +0.015594]` |
| correction epoch 2 | -0.044587 `[-0.054730, -0.035791]` | -0.012142 `[-0.020786, -0.005207]` |
| correction epoch 3 | **-0.049511** `[-0.063550, -0.038455]` | -0.026436 `[-0.039786, -0.016702]` |
| correction epoch 4 | -0.047367 `[-0.060966, -0.036629]` | **-0.027664** `[-0.040431, -0.018175]` |

This rejects epoch 1 on C4 and makes epoch 3 a plausible single-candidate
hypothesis, but it is not a gate pass for any checkpoint.

## Procedural invalidation

Overlapping continuations began work before the experiment source was
committed. The treatment document and first WikiText reservation existed on
disk before model evaluation, but neither they nor the exact execution tooling
had a committed source identity. The C4 curve likewise used uncommitted
checkpoint-arm support. These results cannot authorize promotion.

The first WikiText monitor stored only aggregate values. Its frozen selection
rule required aligned sequence-level paired intervals for all three metrics.
That slice was correctly retired once model-dependent aggregates existed and
was not reopened. A continuation then opened a new, 11-sequence WikiText slice
after both the aggregate and C4 curves were known. Applying the original
three-metric rule to that later slice selects epoch 3, but the slice and its
candidate inventory were not precommitted, so this remains retrospective
hypothesis-generation evidence.

A second, post-hoc selector mixed C4 NLL/full KL with WikiText tail KL. That was
not the frozen rule above and is explicitly rejected even though it also chose
epoch 3. Changing metric roles after observing results is precisely the class
of mistake these experiments must prevent.

All three opened slices are permanently retired. Evidence is retained at:

- `evidence/046/experiment046-weight2-ratio0p8-correction/report.json`;
- `evidence/046/experiment046-c4-validation248-48x512-correction-curve.json`;
- `evidence/046/experiment046-wikitext-validation492-11x512-correction-curve.json`;
- `evidence/046/experiment046-wikitext-predeclared-rule-analysis.json`;
- `evidence/046/experiment046-capability-correction-selection.json` (rejected
  post-hoc rule).

The next experiment may test correction epoch 3 as one frozen candidate on
new, precommitted WikiText and C4 confirmation slices. It must not reselect an
epoch, reuse any Experiment 046 slice, or claim that this trajectory-specific
checkpoint is a transferable correction policy for a fresh factorization.
