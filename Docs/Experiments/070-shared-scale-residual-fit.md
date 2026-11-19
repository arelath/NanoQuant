# Experiment 070: separate residual fit followed by joint tuning

This follows the [shared-scale gain probe](070-shared-scale-stage-gain.md).
The second group now starts from a separate residual fit, not from part of
an already-fitted full-rank factor. The two groups still share input/output
scales, use one fused forward, and store no extra scalar or scale vectors.

## Protocol

- Block 25 up, attention output, and stacked QKV, seeds 0 and 1. Use the
  previous first-stage rank as the cut, and retain the full single-stage
  component count: 832+320, 672+288, and 640+256 respectively.
- Fit the primary first with the retained 800-step ADMM settings and scale
  refit. Round its scales to stored BF16 for the frozen residual target.
  Seed new signs with ADMM on that residual in the orientation of the
  shared scales. This is a sign initializer, not an exact constrained
  sign search. Discard the residual ADMM scales and solve its middle
  coefficients against the original residual with the primary's fixed
  outer scales. No primary parameters are changed in this phase.
- Start joint tuning from either the fitted residual at gain 1, or the
  same residual scaled to 3% of the primary output RMS on calibration
  inputs. Verify the latter ratio after folding and rounding. Both arms
  tune all signs, middle and shared outer scales, and one gain for 64
  AdamW steps. Learning rates: 3e-5 binary, 1e-5 scales, 1e-3 gain.
- Fold the gain into the second group's middle coefficients and score
  the reloaded BF16-scale payload on the same eight validation windows.
  Compare with the verified full-rank single-stage controls from the
  [warm-start probe](070-rank-split-warm-start.md): same bits, seeds, inputs,
  ADMM steps per fit, and joint tuning step count. The split initialization
  uses two ADMM fits instead of one and adds a gain during tuning.

## Results

All 12 candidate arms completed. Mean validation activation NMSE across
both seeds (lower is better):

| Owner | Primary alone | Equal-bit control | Fitted before joint | Fitted after joint | 3% start after joint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Up | 0.15669566 | 0.09954091 | 0.12069984 | 0.10977345 (+10.28%) | 0.13333016 (+33.95%) |
| Attention output | 0.02496077 | 0.01172460 | 0.01484286 | 0.01220440 (+4.09%) | 0.01615991 (+37.83%) |
| Stacked QKV | 0.11954621 | 0.06843249 | 0.08995828 | 0.07670611 (+12.09%) | 0.09270331 (+35.47%) |

Percentages are relative to the equal-bit control. Separate fitting
corrected part of the primary's error, and joint tuning improved every
fitted-residual arm. Neither initialization beat the full-rank single-stage
fit on any seed. The fitted residual initially contributed 14.45–23.01%
of primary output RMS (owner means), so the 3% start removed much of an
already-useful correction. The learned gain grew from 0.130 to 0.192 on up,
0.176 to 0.225 on output, and 0.208 to 0.270 on QKV (means), but did not
regain the fitted magnitude within the 64-step schedule.

Do not change 070 or advance a full-model check from these results. This
tests one residual initializer and a short joint-tuning schedule; it does
not establish a limitation of the shared-scale representation itself.

## Reproduction

Use a free CUDA device and a new output path:

```powershell
$residualSnapshot = 'C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'
.\.venv\Scripts\python.exe tools/probe_shared_scale_residual_fit.py --warm evidence/070/rank-split-warm-start-20261111 --screen evidence/070/rank-split-screen-20261111 --snapshot $residualSnapshot --output evidence/070/shared-scale-residual-fit-replay
```

Evidence: `evidence/070/shared-scale-residual-fit-20261111`.
Eight focused tests, repository Ruff, and mypy passed.
The bounded repository pytest check stopped at 275 passes, 53
deselections, and the known Windows hardlink access denial in legacy
GGUF export adoption; see `regressions.xml`.
