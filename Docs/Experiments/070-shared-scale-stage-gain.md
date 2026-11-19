# Experiment 070: shared scales with a second-stage gain

Test one learned scalar for the second component group while sharing the
input and output scale vectors. If a scalar multiplies both outer scale
vectors in the second stage, its net effect is the product of those
multipliers. Train that net gain directly. Multiply the second group's
middle scales by the gain in FP32, then round once to BF16. The result fits
the original single-stage representation, with no additional bits.

Use one fused forward operation in every arm, so the comparison does not
introduce an extra stage-wise BF16 rounding step. This is a local diagnostic,
not a change to the resident algorithm or a native export/runtime test.

## Protocol

- Start from the persisted full-rank single-stage weights of the
  [warm-start probe](070-rank-split-warm-start.md). Use block 25 up, output,
  and stacked QKV, both seeds. The group boundary is the previous first-stage
  rank: 832, 672, and 640 respectively. Retain all components and hold signs fixed.
- At a unit gain the initialization reproduces the parent tensor payload
  exactly. Compare continued scale tuning, gain-only tuning, and scale
  tuning plus a gain. All arms see the same 64 steps and captured
  calibration inputs. AdamW learning rates are 1e-5 for scales and 1e-3
  for the gain, with zero weight decay and gradient clipping at 1.
- Fold the gain before writing the ordinary sign/BF16-scale payload,
  reload it for scoring, and verify unchanged represented-bit cost.
  Score the retained eight WikiText validation windows; these windows
  are exploratory, not independent end-to-end confirmation.

## Results

18 arms completed. Mean validation activation NMSE across two seeds:

| Owner | Starting error | Scales only | Gain only | Scales + gain | Scales + gain vs. scales only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Up | 0.09954091 | 0.09915092 | 0.09955820 | 0.09914836 | -0.003% |
| Attention output | 0.01172460 | 0.01164625 | 0.01172809 | 0.01168102 | +0.299% |
| Stacked QKV | 0.06843249 | 0.06809787 | 0.06841088 | 0.06808221 | -0.023% |

Jointly tuned gains ranged from 0.984 to 1.022. They did not produce a
meaningful improvement over directly tuning the original scales. This is
consistent with the gain being a different parameterization of middle scales
that were already trainable, rather than a more expressive representation.
Keep 070 unchanged. No full-model KL/NLL run was advanced.

## Reproduction

From the repository root with the CUDA device free, using a new output path:

```powershell
$gainSnapshot = 'C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'
.\.venv\Scripts\python.exe tools/probe_shared_scale_stage_gain.py --warm evidence/070/rank-split-warm-start-20261111 --screen evidence/070/rank-split-screen-20261111 --snapshot $gainSnapshot --output evidence/070/shared-scale-stage-gain-replay
```

Evidence: `evidence/070/shared-scale-stage-gain-20261111`. Six focused tests,
repository Ruff, and mypy passed. The fold/reload regression verifies identical
BF16 forward outputs with and without explicitly retaining the learned scalar
under the same fused operation order.
The bounded full-suite check stopped after 275 passes, 53 deselections, and
the known Windows hardlink access denial in legacy GGUF export adoption.
See `regressions.xml` in the evidence directory.
