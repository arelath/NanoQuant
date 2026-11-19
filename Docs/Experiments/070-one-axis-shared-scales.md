# Experiment 070: share one outer scale vector

Compare two jointly tuned residual-stage configurations: shared input
scales with independent output scales, and shared output scales with
independent input scales. Both save the shared vector once at 16-bit BF16
precision. The diagnostic payload restores the omitted vector by reference
to the saved copy; this does not introduce a production file format.

## Protocol

- Use the verified block-25 [warm-start controls](070-rank-split-warm-start.md):
  up, attention output, and stacked QKV; seeds 0 and 1.
- Refit each reduced single-stage parent with the same 800-step ADMM
  settings, seed, and scale refit. Split its components with the same cut.
  Each new arm starts with identical factors and scales. Tie the selected
  outer scale by parameter identity so gradients from both stages accumulate
  into one optimizer update. All other parameters remain jointly trainable.
- Use the same 64 joint-tuning steps and activation captures. Keep both stage
  forwards separate, as in the fully independent control, to avoid adding a
  fused-versus-split arithmetic confound.
- Score after reloading the saved BF16 scales and hard signs on the same
  eight validation windows. Audit that payload entries match the represented
  cost: one bit per sign and 16 bits per scale. File headers and diagnostic
  INT8 sign containers are not counted as deployment bytes.
- Each owner is capped by its original full-rank single-stage cost. Verify
  that the tested 32-aligned rank fits, but the next 32-component step does
  not. Both variants are compared at the same total rank, not necessarily
  identical represented bits. The retained fully independent arm uses the
  same rank and initialization but stores both outer vectors twice.

## Results

All 12 candidate arms completed. Mean validation activation NMSE across
the two seeds (lower is better):

| Owner | Full single | Reduced single | Independent split | Share input | Share output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Up | 0.09954091 | 0.10316529 | 0.10332282 | 0.10325863 | 0.10334576 |
| Attention output | 0.01172460 | 0.01244081 | 0.01269834 | 0.01249652 | 0.01289702 |
| Stacked QKV | 0.06843249 | 0.07298757 | 0.07280186 | 0.07291428 | 0.07289262 |

Relative to fully independent stages at the same rank, sharing either
axis changed up and QKV errors by no more than 0.16%. Attention output
preferred sharing input scales on both seeds: its mean was 1.59% better
than independent stages, whereas sharing output scales was 1.56% worse.
Sharing input still did not beat the reduced single-stage attention
output control.

| Owner | Split ranks | Bits saved sharing input | Bits saved sharing output |
| --- | --- | ---: | ---: |
| Up | 832+288 | 18,432 | 110,592 |
| Attention output | 672+256 | 16,384 | 18,432 |
| Stacked QKV | 640+224 | 18,432 | 24,576 |

Savings are against the independent two-stage arm. No candidate could
recover a 32-component step under its full single-stage cap. All variants
still trailed the full single-stage control, by 3.7–10.0% relative NMSE.
This supports one-axis sharing as a cheaper version of an already needed
two-stage representation, not as a reason to replace these single-stage
owners. No allocator change or full-model benchmark was advanced.

## Reproduction

From the repository root with a free CUDA device and a new output path:

```powershell
$axisSnapshot = 'C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'
.\.venv\Scripts\python.exe tools/probe_one_axis_shared_scales.py --warm evidence/070/rank-split-warm-start-20261111 --screen evidence/070/rank-split-screen-20261111 --snapshot $axisSnapshot --output evidence/070/one-axis-shared-scales-replay
```

Evidence: `evidence/070/one-axis-shared-scales-20261111`.
Seven focused tests, repository Ruff, and mypy passed.
The bounded full-suite check stopped at 275 passes and 53 deselections
after the known Windows hardlink access denial in legacy GGUF export
adoption. See `regressions.xml` in the evidence directory.
