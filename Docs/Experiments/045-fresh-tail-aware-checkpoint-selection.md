# Experiment 045: Fresh Tail-Aware Checkpoint Selection

## Status

Completed with no surviving earlier checkpoint. The frozen rule selected epoch
5 as the earliest three-metric plateau member, but its NLL is significantly
worse than epoch 8, so epoch 8 is retained and no derived model is materialized.

This result is analysis-only for an additional procedural reason. An
overlapping continuation opened the reserved slice after this document, the
selection rule, and the reservation had been written to the shared workspace,
but before their source commit completed. That misses the intended
commit-before-evaluation condition. Because the frozen outcome rejects a
replacement and changes no model, it may conservatively rule out the earlier
checkpoint hypothesis; it cannot support promotion of a candidate.

## Question

Experiment 044 proved that 256-step tail-aware KD improves its own fresh pre-KD
factorization on untouched WikiText and C4, yet its final packed model is 4.94%
worse than Experiment 042. Experiment 045 asks whether transferring a fixed
epoch-8 endpoint across factorizations left a materially better earlier
checkpoint unused.

This is an analysis-only retained-state experiment. It opens no new
factorization and changes no trained values. The immutable Experiment 044
pre-KD state, teacher cache, and eight primary KD checkpoints are frozen.

## Arms

- pre-KD control: 0 optimizer steps;
- tail-aware epochs 1 through 8: exactly 32, 64, 96, 128, 160, 192, 224, and
  256 optimizer steps;
- all arms use the same Experiment 044 frozen identity, top-64 plus aggregated
  tail objective, tail mass weight 0.5, cache, tokens, and optimizer protocol.

Any observed step mismatch invalidates the arm.

## Development slice

WikiText validation offset 444, 24x512, token hash
`sha256:6b35ae6c5ef767ddec6cb400f82f975d90809e4320f9de92b2b3d8d8f902eadd`
is reserved under `experiment045-wikitext-validation444-24x512`. It is a
development slice and is permanently retired after first model evaluation.
It cannot serve as a final gate.

## Frozen selection rule

1. A checkpoint is eligible only when paired 95% upper bounds versus pre-KD
   are below zero for both NLL and full-vocabulary KL.
2. Compute the minima across all eight checkpoints for NLL, full KL, and
   top-64-plus-tail KL. Keep eligible checkpoints whose point estimates are
   within 0.02 nats of all three minima.
3. Select the earliest remaining checkpoint. If none remains, there is no
   survivor.
4. Replacing epoch 8 additionally requires selected-minus-epoch-8 NLL to have
   a paired 95% upper bound below zero, while the paired upper bounds for full
   KL and top-64-plus-tail KL must be at most +0.02 nats. Otherwise retain
   epoch 8 and reject checkpoint selection as an explanation of the gap.

The rule is frozen before opening the slice. Training loss, selected mass, the
ordinary test protocol, and task results cannot choose the checkpoint.

## Follow-up gates

Only a surviving earlier checkpoint is materialized. It must then:

1. improve the ordinary 64x128 WikiText result over Experiment 044 and either
   beat Experiment 042 or fall within its 2% relative bound;
2. pass a newly reserved 48x512 C4 slice with paired NLL and full-KL
   non-regression;
3. show no established regression on the 1,000-example six-task guardrail;
4. preserve the exact factor budget and remain representable by the existing
   global-tuning artifact contract.

No correction, final-norm fold, block-25 refit, or coefficient change may be
introduced during this experiment. If checkpoint selection cannot close the
gap, those become separate, freshly calibrated composition experiments.

## Result

All arms matched their declared step counts and immutable Experiment 044
identity. The development means were:

| Arm | Steps | NLL | Full KL | Top-k + tail KL | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre-KD | 0 | 5.134158 | 1.708103 | 1.609366 | 169.72 |
| epoch 1 | 32 | 4.824066 | 1.609028 | 1.518957 | 124.47 |
| epoch 2 | 64 | 4.758649 | 1.547678 | 1.462323 | 116.59 |
| epoch 3 | 96 | 4.718302 | 1.521553 | 1.437353 | 111.98 |
| epoch 4 | 128 | 4.686435 | 1.494005 | 1.410709 | 108.47 |
| epoch 5 | 160 | 4.670901 | 1.485623 | 1.402212 | 106.79 |
| epoch 6 | 192 | 4.665452 | 1.481484 | 1.398062 | 106.21 |
| epoch 7 | 224 | 4.665298 | 1.481242 | 1.397864 | 106.20 |
| epoch 8 | 256 | 4.664809 | 1.480959 | 1.397553 | 106.15 |

Every trained checkpoint improved NLL and full KL over pre-KD with paired 95%
upper bounds below zero. Epochs 5-8 were within 0.02 nats of all three minima,
so the predeclared plateau rule selected the earliest, epoch 5.

Epoch 5 does not satisfy the replacement rule. Relative to epoch 8 it is worse
on all three metrics:

- NLL `+0.006092`, paired 95% interval `[+0.004185, +0.007857]`;
- full KL `+0.004664`, interval `[+0.003373, +0.005907]`;
- top-k-plus-tail KL `+0.004659`, interval `[+0.003388, +0.005879]`.

The durable selection receipt therefore says `retain epoch8`. Earlier
checkpoint selection cannot explain or close Experiment 044's deployment
quality gap, so the follow-up materialization, C4, and task gates are not
opened.

Evidence:

- `evidence/045/experiment045-wikitext-validation444-24x512-checkpoint-curve.json`;
- `evidence/045/experiment045-wikitext-validation444-24x512-checkpoint-curve.checkpoint.json`;
- `evidence/045/experiment045-checkpoint-selection.json`.

## Frozen execution

`tools/probe_wikitext_kd_quality.py` evaluates the pre-KD control and all eight
immutable checkpoints in one invocation. Its report must include paired
intervals for every arm against both pre-KD and epoch 8; rerunning different
pairings after opening the slice is forbidden. The output is
`evidence/045/experiment045-wikitext-validation444-24x512-checkpoint-curve.json`.

`tools/select_wikitext_kd_checkpoint.py` applies the rule above without loading
a model or changing an artifact. Its output is
`evidence/045/experiment045-checkpoint-selection.json` and binds the sweep by
SHA-256. The intended protocol required both tools and their tests to be
committed before the reserved development slice was opened. As recorded in the
status above, the overlapping execution violated that ordering requirement, so
the result is restricted to conservative hypothesis rejection.
