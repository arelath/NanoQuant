# Experiment 045: Fresh Tail-Aware Checkpoint Selection

## Status

Predeclared. The development slice is reserved but no checkpoint has been
evaluated on it.

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

## Frozen execution

`tools/probe_wikitext_kd_quality.py` evaluates the pre-KD control and all eight
immutable checkpoints in one invocation. Its report must include paired
intervals for every arm against both pre-KD and epoch 8; rerunning different
pairings after opening the slice is forbidden. The output is
`evidence/045/experiment045-wikitext-validation444-24x512-checkpoint-sweep.json`.

`tools/select_wikitext_kd_checkpoint.py` applies the rule above without loading
a model or changing an artifact. Its output is
`evidence/045/experiment045-wikitext-validation444-24x512-selection.json` and
binds the sweep by SHA-256. Both tools and their tests are committed before the
reserved development slice is opened.
