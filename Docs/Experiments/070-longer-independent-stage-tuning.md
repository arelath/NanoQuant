# Experiment 070: longer joint tuning of independent stages

Test whether longer tuning or a different two-stage initialization beats the
short [warm-start probe](070-rank-split-warm-start.md). Stage input and output
scales are independently trainable and stored as BF16. The resident run,
recipe, and packed deployment format are unchanged.

## Protocol

- Block 25 up (832+288) and stacked QKV (640+224), both retained seeds.
  All split arms have identical ranks and bit cost within the original
  single-stage cap. Full and reduced single-stage controls use the same
  tuning schedule; the reduced control isolates equal component count.
- **Warm:** split one reduced-rank ADMM fit, retaining its output at
  initialization. **Simultaneous:** independently seed signs for both
  groups, initialize outer scales from source row/column magnitudes, and
  solve both groups' middle coefficients together. Neither group fits first;
  both start nonzero and all parameters then optimize the summed output.
  **Alternate:** sequential ADMM initialization followed by one backfitting
  sweep, revisiting each stage against the residual of the other. Accept
  stage replacements only if their persisted-BF16 weighted error improves.
  This arm uses more ADMM work; it is not a compute-matched initial fit.
- Keep one AdamW optimizer alive through 128 epochs, eight calibration
  sequences per epoch. Save and reload checkpoints at 0, 64, 256, and
  1,024 updates. Binary LR 3e-5, scale LR 1e-5, zero weight decay, gradient
  clipping at 1; the fixed rate is intentional for this duration comparison.
  Record training and validation activation NMSE and sign changes.
- Select one split method/checkpoint per owner by mean validation NMSE
  across both seeds. Select the full single-stage checkpoint by the same
  criterion. Compare the selected pairs by full-vocabulary teacher KL and
  NLL with single-owner dense splices into the otherwise unchanged tuned
  070 context, on a disjoint WikiText validation window (32 sequences,
  length 128, offset 192). This is a diagnostic, not the final test set or
  packed-runtime acceptance.

## Local results

All 20 training arms completed, with 80 persisted checkpoints. Mean
validation activation NMSE across the two seeds (lower is better):

| Owner / initialization | 0 steps | 64 | 256 | 1,024 |
| --- | ---: | ---: | ---: | ---: |
| Up / full single | 0.106531 | 0.099487 | 0.121483 | 0.144031 |
| Up / reduced single | 0.110835 | 0.103180 | 0.124654 | 0.147287 |
| Up / warm | 0.110835 | 0.103308 | 0.125767 | 0.154551 |
| Up / simultaneous | 0.999530 | 0.643173 | 0.416612 | 0.312011 |
| Up / alternate | 0.118091 | 0.109193 | 0.130920 | 0.164614 |
| QKV / full single | 0.078507 | 0.068341 | 0.075982 | 0.091907 |
| QKV / reduced single | 0.082917 | 0.072917 | 0.079635 | 0.093590 |
| QKV / warm | 0.082917 | 0.072888 | 0.080971 | 0.091843 |
| QKV / simultaneous | 0.992262 | 0.477942 | 0.319428 | 0.208182 |
| QKV / alternate | 0.088792 | 0.076245 | 0.080146 | 0.093388 |

The warm split won among splits for both owners at 64 updates. Longer
fixed-rate tuning raised warm validation error by 49.6% (up) and 26.0%
(QKV) at 1,024 updates. QKV training error nevertheless fell from
0.051992 to 0.046560; up training error also worsened slightly. The
warm-start sign counts continued changing substantially, rather than the
optimizer simply doing nothing. The simultaneous random-sign start kept
improving but did not catch the warm start under this schedule. The
alternating sweep accepted just one of eight proposals (QKV seed 0,
second stage, a 0.061% weighted weight-error improvement); it never
replaced the primary. This is not evidence about a warm-started
coordinate optimizer that can make incremental changes within a stage.

Model confirmation selected 64-step warm versus 64-step full on both
owners. Warm costs 9,307,648 versus 9,437,184 represented bits on up,
and 2,422,272 versus 2,465,792 on QKV: same budget cap, slightly fewer
actual bits due to the aligned rank constraint.

## Held-out model checks

Both model splice checks completed on the 32-sequence window. Deltas
are warm split minus the matched full single-stage control, in nats
per token; brackets are paired sequence bootstrap 95% intervals.

| Owner / seed | KL delta [95% interval] | NLL delta [95% interval] |
| --- | --- | --- |
| Up / 0 | +0.004699 [+0.001978, +0.007404] | +0.001017 [-0.005575, +0.007121] |
| Up / 1 | +0.002067 [-0.002786, +0.006059] | -0.012588 [-0.018675, -0.006364] |
| QKV / 0 | -0.003374 [-0.004913, -0.001930] | +0.000311 [-0.001987, +0.002445] |
| QKV / 1 | -0.000040 [-0.001303, +0.001197] | -0.005561 [-0.008194, -0.002854] |

Up KL rose 0.31–0.70% and NLL was mixed. QKV KL improved on one
seed and was effectively tied on the other; NLL was tied on the first
and improved on the second. The QKV result is a useful candidate for
replication at other blocks, not a consistent two-metric win or evidence
to replace the 070 allocator. Neither extended fixed-rate tuning nor
simultaneous random-sign initialization was selected for either owner.

Confirmation evidence: `evidence/070/joint-independent-up-confirmation-20261111`
and `evidence/070/joint-independent-qkv-confirmation-20261111`. The fresh
source audit passed for 26 blocks and 874 artifacts and is retained at
`evidence/070/joint-independent-stages-source-validation-20261111.json`.

## Reproduction

With a free CUDA device, from the repository root, using new output paths:

```powershell
$jointSnapshot = 'C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'
$jointRun = 'evidence/070/070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it'
.\.venv\Scripts\python.exe tools/validate_resident_run.py --run-output $jointRun --expected-blocks 26 --require-complete
.\.venv\Scripts\python.exe tools/probe_joint_independent_stages.py --warm evidence/070/rank-split-warm-start-20261111 --screen evidence/070/rank-split-screen-20261111 --snapshot $jointSnapshot --output evidence/070/joint-stages-replay
.\.venv\Scripts\python.exe tools/score_residual_rank_split.py --probe evidence/070/joint-stages-replay/confirmation-input --snapshot $jointSnapshot --output evidence/070/joint-stages-up-replay --owner mlp.up_proj --ranks 832 288 --offset 192 --samples 32
.\.venv\Scripts\python.exe tools/score_residual_rank_split.py --probe evidence/070/joint-stages-replay/confirmation-input --snapshot $jointSnapshot --output evidence/070/joint-stages-qkv-replay --owner self_attn.attn_qkv --ranks 640 224 --offset 192 --samples 32
```

Evidence: `evidence/070/joint-independent-stages-20261111`.
Eight focused tests, repository Ruff, and mypy passed. The bounded
full-suite check stopped at 277 passes, 53 deselections, and the known
Windows hardlink access denial in legacy GGUF export adoption. See
`regressions.xml` in the evidence directory.
