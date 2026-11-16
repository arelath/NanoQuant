# Notes: where quantization error hurts, and why it adds up

Working notes for the Medium article in [article.md](article.md). Every number here comes from
the Gemma 3 270M FP32 sweep in
`Datasets/gemma-3-270m-it-fp32-sensitivity-16x256/measurements/1f87110f94d213e6173fe716312c168c16a784b91e3cb03a6db9b9ce2d4aa43c/`
(`results.json`, `kl-response-fits.json`, `joint-kl-targets-100-runs.json`, `gguf-kl.json`).
Figures are rendered by `tools/plot_kl_article_figures.py` into [figures/](figures/).

## The one-sentence pitch

If you damage one weight matrix of a small LLM at a time and measure how far the output
distribution moves, you get a map of where error hurts, and the map is useful because the
damage from all 126 matrices adds up to within 1–2% of the sum of the parts.

## Reader

People who use or build quantized models and know what a GGUF is, but who have not looked
inside one. They know "Q4_K_M is fine, Q2 is rough". They may not know what KL divergence is,
so it needs two plain sentences. No formulas beyond one power law.

## Setup facts

- Model: `google/gemma-3-270m-it`, revision `ac82b4e`. 18 decoder blocks, 7 projection
  matrices each (attention q, k, v, o; MLP gate, up, down), so 126 matrices. Hidden size 640.
- Text: 16 windows of 256 tokens from `eaddario/imatrix-calibration`, 4,080 scored next-token
  positions.
- Metric: KL(teacher ‖ damaged model) over the full 262,144-token vocabulary, averaged per
  position, in nats/token. The teacher is the undamaged model's own output.
- Damage: isotropic Gaussian noise added to one matrix, scaled so the noise norm is a fixed
  fraction of the matrix's norm (relative Frobenius error).
- Sweep: 6 error levels (1%, 2.5%, 5%, 10%, 20%, 35%) × 126 matrices × 3 noise seeds =
  2,268 trials.
- Precision: FP32, TF32 disabled. Clean-model KL against the stored teacher is 3e-7 nats/token.

## Finding 1: KL grows with the square of weight error

- Fit per matrix: `KL = c × error^p`. Median exponent 1.975 (range 1.70–2.10). Median R² 0.9987.
- So halving a matrix's error cuts its KL cost by about 4×. Each matrix is described by one
  number, its coefficient `c`.
- Figure: `kl-error-response.png`. Lines are parallel on log axes, which is the same statement.

## Finding 2: sensitivity varies about 87× between matrices

At a uniform 5% error (from the fits):

- Most sensitive: block 3 MLP up (0.0222 nats/token), block 0 MLP down (0.0148), block 17 MLP
  down (0.0144), block 13 attention v (0.0115), block 0 MLP gate (0.0113).
- Least sensitive: block 16 attention q (0.00026), block 16 attention k (0.00030), block 15
  attention q (0.00036).
- Ratio of the extremes: 87×.
- Share of total sensitivity by projection type: MLP down 24.9%, MLP up 19.7%, attention v
  16.4%, MLP gate 14.1%, attention o 8.8%, attention k 8.3%, attention q 7.8%. The three MLP
  matrices carry 59%; q and k together carry 16%.
- Share by block: block 0 14.6%, block 1 10.9%, block 3 10.7%, block 17 7.7%. Those four of 18
  blocks hold 44%. Blocks 6–9 and 14–16 hold 2.3–3.2% each.
- Oddities worth a sentence: block 3's MLP up matrix is the single most sensitive matrix in the
  model; block 13's k and v and block 5's q are sensitive islands in otherwise tolerant rows.
- Figure: `kl-sensitivity-heatmaps.png`.

## Finding 3: the equal-KL map

Invert each fit: how much error can this matrix take before it costs a fixed KL?

- At 0.001 nats/token per matrix: tolerances run from 1.0% (block 3 MLP up) to 9.9% (block 16
  attention q), median 3.3%. A 10× range.
- At 0.0025: 1.6%–15.6%, median 5.2%. At 0.005: 2.3%–22.1%, median 7.4%.
- All 126 targets at all three levels sit inside the measured 1%–35% range. Nothing is
  extrapolated.
- Median tolerance by type at 0.001: attention k 5.3%, attention q 5.0%, MLP gate 3.6%,
  attention o 3.4%, MLP up 2.9%, attention v 2.8%, MLP down 2.3%.
- Figure: `equal-kl-target-error-heatmap-0.001-nats.png` (the 0.0025 and 0.005 versions are in
  `figures/` too and show the same pattern scaled up).

## Finding 4: errors add (the headline)

Perturb all 126 matrices at once, each to the error its own fit says costs the target KL.
Prediction = sum of the 126 fitted KLs. 100 independent noise draws per target.

| Target per matrix (nats/token) | Additive prediction | Mean measured | Measured / predicted | 95% CI of ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.1260 | 0.1229 | 0.976 | 0.965–0.986 |
| 0.0025 | 0.3150 | 0.3090 | 0.981 | 0.970–0.992 |
| 0.005 | 0.6300 | 0.6239 | 0.990 | 0.978–1.002 |

- Slightly sub-additive (1–2.4% below the sum). The gap shrinks as error grows and the top
  interval includes 1.
- Any single noise draw scatters about ±5–6% around the mean (SD 0.0065, 0.0176, 0.0388);
  individual runs range from 0.87× to 1.21×. Three runs were not enough to see the 2% gap; it
  took 100.
- Control: the same joint code path at zero error gives exactly 0.
- Why this is plausible: for small perturbations KL is a quadratic form in the weight change
  (the Fisher metric). Independent noise in different matrices has zero expected cross term, so
  the quadratic forms add. The measurement says this second-order picture still holds at a
  joint KL of 0.6 nats/token, which is not a small perturbation.
- Why it matters: 126 separate measurements predict the 127th. A bit allocator can treat KL as
  a budget and spend it matrix by matrix.
- Figure: `joint-additivity.png`.

## Finding 5: real quantizers, and where they put their error

Seven GGUFs dequantized and loaded into the same FP32 model, so only weight error is measured
(not llama.cpp runtime arithmetic). Unsloth files from `unsloth/gemma-3-270m-it-GGUF` at
revision `c90975d`; all carry imatrix metadata (`imatrix_unsloth.gguf`).

| File | Projection bits/weight | Median weight error | Projections-only KL | Additive prediction | Measured / predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 8.50 | 0.57% | 0.0047 | 0.0063 | 0.74 |
| Q4_K_M | 5.43 | 4.5% | 0.181 | 0.396 | 0.46 |
| Q3_K_M | 4.54 | 7.9% | 0.358 | 0.912 | 0.39 |
| Q2_K | 4.15 | 8.1% | 0.943 | 1.975 | 0.48 |
| UD-IQ2_XXS | 3.79 | 8.1% | 1.737 | 5.217 | 0.33 |
| Binary factors, before distillation | 2.82 | 23.5% | 1.815 | 8.535 | 0.21 |
| Binary factors, after distillation | 2.82 | 23.6% | 1.504 | 8.572 | 0.18 |

Points to make:

- **The prediction overshoots real quantizers by 2–3×.** Gaussian noise of the same norm is
  more damaging than K-quant or I-quant error. The prediction still ranks the five llama.cpp
  files in the right order.
- **It does not rank across quantizer families.** The binary-factor files are predicted worse
  than UD-IQ2_XXS (8.5 vs 5.2) but the distilled one measures better (1.50 vs 1.74).
- **Why: see Finding 8.** The one-matrix-at-a-time test (run 2026-11-15) separates the two
  candidate explanations: each matrix's error being less harmful than Gaussian noise, and
  errors in different matrices costing less together than apart.
- **Q8_0 caveat:** all 126 of its errors (0.53%–0.78%) are below the 1% floor of the sweep, so
  its prediction is extrapolated.
- **Names mislead on this model.** K-quants need rows that divide into 256-wide blocks. Five of
  the seven projection types have 640 inputs, so they fall back to IQ4_NL or Q5_0. In the
  "Q2_K" file only attention o and MLP down are actually Q3_K; nothing is Q2_K, and the
  projections average 4.15 bits. Compare files by measured bits/weight, not by name.
- **Where the low-bit files put error:** UD-IQ2_XXS holds q, k, gate and up at about 8% error
  and v at 4.5%, and puts 27% on MLP down and 35% on attention o. Those two types account for
  89% of its predicted KL (55% and 34%).
- **Unsloth's dynamic allocation agrees with the map.** For MLP down it uses the larger IQ3_S
  format in blocks 0, 1, 3, 5 and 17. Our four most sensitive MLP down blocks are 0, 17, 3, 1.
- **Stock llama.cpp allocation is positional.** Q4_K_M gives extra bits (Q6_K for MLP down,
  Q8_0 for attention v) to blocks 0, 1, 4, 7, 10, 13, 15, 16, 17. That catches 0, 1 and 17 but
  misses block 3 (third most sensitive for MLP down) and spends on blocks 4, 10 and 7, which
  rank 10th, 11th and 14th of 18. Unsloth's fifth protected block, 5, ranks 16th on our map.
- **The embedding is its own line item.** Gemma ties the token embedding to the output
  projection. At Q8_0 it costs 0.0034 nats/token, which is 41% of the Q8_0 file's total KL
  (0.0083). UD-IQ2_XXS stores it at Q5_1 and pays 0.103.
- **File size hides this.** At Q8_0 the embedding is about 178 MB, most of a 220–290 MB file.
- Figures: `quantizer-weight-error-maps.png`, `predicted-vs-measured-kl.png`.

## Finding 8: why real error is cheaper (one matrix at a time)

Source: `gguf-isolated-kl.md/json` and `block-scaled-noise-kl*.md/json` beside the sweep, from
`tools/measure_gguf_isolated_kl.py` and `tools/measure_block_scaled_noise_kl.py`.

Method: for each file, load one dequantized projection at a time into the clean model (126
scores), then every block's seven projections together (18), every projection type across all
blocks (7), and everything (1).

- Per-matrix ("shape") factor = sum of isolated KLs / sum of Gaussian predictions.
- Interaction factor = joint KL / sum of isolated KLs.
- Their product is measured / predicted.

| File | Gaussian prediction | Sum of isolated KLs | Joint KL | Per-matrix | Interaction | Product |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary factors, before distillation | 8.535 | 2.956 | 1.815 | 0.346 | 0.614 | 0.213 |
| Binary factors, after distillation | 8.572 | 2.970 | 1.504 | 0.346 | 0.506 | 0.175 |
| UD-IQ2_XXS | 5.217 | 2.115 | 1.737 | 0.405 | 0.821 | 0.333 |
| Q2_K | 1.975 | 1.082 | 0.943 | 0.548 | 0.871 | 0.477 |
| Q3_K_M | 0.912 | 0.417 | 0.358 | 0.458 | 0.858 | 0.393 |
| Q4_K_M | 0.396 | 0.206 | 0.181 | 0.520 | 0.879 | 0.458 |
| Q8_0 | 0.0063 | 0.0051 | 0.0047 | 0.809 | 0.915 | 0.740 |

Points to make:

- **Most of the saving is per-matrix.** For the four low-bit llama.cpp files the per-matrix
  factor is 0.41–0.55 and interaction 0.82–0.88. Of the total gap to the prediction, 86–89% is
  per-matrix. For the binary-factor files it is 79–83%.
- **By matrix type** (median isolated KL / Gaussian prediction, llama.cpp low-bit files): q
  0.23–0.26, k 0.26–0.27, v 0.80–0.97, o 0.53–0.62, gate 0.52–0.58, up 0.50–0.67, down
  0.35–0.57. Range over single matrices 0.05–1.24.
- **Block 5's q matrix is a Gaussian-only island.** Isolated real KL is 0.05–0.18 of its
  prediction in all seven files, and 0.16 under block-scaled random noise. Block 13 q: 0.07–0.18
  in the low-bit files, 0.50 at Q8_0.
- **Block-scaled random noise control.** Uniform noise scaled by the largest magnitude in each
  run of 32 weights along a row, 5% relative error, three seeds: per-matrix factor 0.88–0.90
  (q 0.53, k 0.57, v 1.02, o 0.84, gate 0.83, up 0.92, down 0.95). So block scaling alone
  explains about a tenth.
- **Q8_0 behaves like block-scaled noise.** Factor 0.81; per-matrix median real / block-noise
  0.95; rank correlation 0.81 across the 126 matrices. `quantize_q8_0` in llama.cpp ignores
  the imatrix (`(void)quant_weights`). Caveat: all Q8_0 errors are below 1%, so its Gaussian
  predictions are extrapolated.
- **Every format below 8 bits is cheaper.** Per-matrix factor by stored type, pooled over the
  five llama.cpp files with duplicate tensors counted once: Q6_K 0.45, Q5_K 0.43, Q5_1 0.41,
  Q5_0 0.62, Q4_K 0.40, IQ4_NL 0.47, IQ3_S 0.41, Q3_K 0.57, IQ2_S 0.32, IQ2_XXS 0.40.
- **Calibration is the cause: see Finding 9.** All Unsloth files carry imatrix metadata and
  every format except Q8_0 uses it, so bit width is confounded with calibration in these
  files. The controlled test below removes the confound.
- **Interaction, llama.cpp files:** whole file 0.82–0.92. Block groups (seven matrices of one
  block): median 0.95–1.00, range 0.83–1.26. Type groups (one type across 18 blocks): median
  0.96–0.99. So it accumulates across the model, not inside a block or a type. Present at
  Q8_0 (0.915 at 0.005 nats/token), so not saturation.
- **Interaction, random noise reference:** block-scaled noise with the same draws used alone
  and together: 0.985, 0.930, 0.945 (three seeds). So independent noise already gives a few
  percent, and the llama.cpp excess over that is about 5–10 points. Each real file is a single
  deterministic realisation, and the five llama.cpp files share many identical tensors, so
  they are not five independent samples.
- **Interaction, binary factors:** 0.61 before distillation, 0.51 after. Block groups median
  0.89 and 0.87. Likely reason: block-by-block fitting against the teacher with compressed
  predecessors, so later matrices compensate for earlier ones. That reason is our reading of
  the method, not something this test isolates.
- **Distillation changed only the interaction.** Sum of isolated KLs 2.956 → 2.970; per-matrix
  isolated KL after / before has median 1.006 (range 0.79–1.20). Joint KL 1.815 → 1.504.
- **A better forecast.** For llama.cpp files the sum of isolated real KLs overestimates the
  joint by 9–22%, against 2–3× for the Gaussian map.
- Figure: `shape-and-interaction.png`.

## Finding 9: calibration is the per-matrix saving (imatrix ablation)

Source: `gguf-kl-imatrix-ablation.md/json` and `gguf-isolated-kl-imatrix-ablation.md/json`
beside the sweep. Files and provenance: `Datasets/gemma-3-270m-it-gguf-local-quantize/`
(`source.json`). Run 2026-11-15.

Method: convert the pinned Google snapshot to an F32 GGUF, then run the local
`llama-quantize` (llama.cpp commit `2927569f3`) to Q4_K_M, Q3_K_M and Q2_K, each with and
without `--imatrix imatrix_unsloth.gguf_file` (Unsloth's published imatrix for this model,
built from `unsloth_calibration_gemma-3-270m-it.txt`, not our test text). Same binary, same
tensor types within a pair.

| File | Median weight error | Joint KL | Per-matrix | Interaction | Product |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M, calibrated | 4.51% | 0.1808 | 0.520 | 0.878 | 0.457 |
| Q4_K_M, no calibration | 4.54% | 0.2978 | 0.864 | 0.898 | 0.776 |
| Q3_K_M, calibrated | 8.66% | 0.4596 | 0.511 | 0.849 | 0.434 |
| Q3_K_M, no calibration | 8.80% | 0.7651 | 0.800 | 0.914 | 0.731 |
| Q2_K, calibrated | 9.12% | 1.0710 | 0.568 | 0.888 | 0.505 |
| Q2_K, no calibration | 9.20% | 1.6558 | 0.828 | 0.946 | 0.784 |

Points to make:

- **Same error, different damage.** Per-matrix weight error with / without the imatrix has
  median 0.99–1.00 (range 0.91–1.15). KL without it is 1.65×, 1.66× and 1.55× the calibrated
  KL; calibration cuts KL by 39%, 40% and 35% at the same file size.
- **Without calibration, quantization error behaves like block-scaled random noise.**
  Per-matrix factor 0.80–0.86 against 0.88–0.90 for the noise and 0.81 for Q8_0. By type
  (no imatrix / block-scaled noise): q 0.46–0.54 / 0.53, k 0.51 / 0.57, v 1.03–1.11 / 1.02,
  o 0.77–0.81 / 0.84, gate 0.83–0.85 / 0.83, up 0.85–0.89 / 0.92, down 0.94–1.01 / 0.95.
- **Where calibration helps** (Q4_K_M, calibrated / uncalibrated per-matrix ratio by type):
  down 0.42 / 1.01, q 0.26 / 0.46, k 0.27 / 0.51, o 0.53 / 0.81, gate 0.58 / 0.83, up
  0.67 / 0.89, v 0.97 / 1.03. Largest on MLP down, q and k; almost none on v.
- **Full chain for calibrated Q4_K_M:** 0.864 (block scaling) × 0.602 (calibration) × 0.878
  (interaction) = 0.457.
- **Calibration adds a little interaction too:** 0.85–0.89 calibrated against 0.90–0.95
  uncalibrated. Independent block-scaled noise gave 0.93–0.99.
- **Replica check.** The local calibrated Q4_K_M has the same tensor types as Unsloth's Q4_K_M
  and matches its per-matrix weight error to 1e-5 (joint KL 0.1808 against 0.1812). The local
  Q3_K_M and Q2_K are not replicas of Unsloth's: this llama.cpp falls back to Q4_0 for the
  640-wide matrices where Unsloth's files have IQ4_NL.
- **Not tested:** IQ formats (llama-quantize refuses very low-bit types without an imatrix),
  other imatrix files, other quantizers.
- Figure: `calibration-ablation.png`.

## Finding 10: imatrix-weighted error predicts per-matrix KL

Source: `outputs/074/074-imatrix-weighted-kl.json` and
`outputs/074/measure_imatrix_weighted_kl.py` (the original 074 analysis), generalised to every
file as `tools/predict_imatrix_weighted_kl.py` with reports
`gguf-isolated-kl-imatrix-weighted.md/json` and
`gguf-isolated-kl-imatrix-ablation-imatrix-weighted.md/json` beside the sweep. Both give the
same 074 numbers. CPU only, no model run.

Method: for each projection, sum the squared residual per input column, weight each column by
its importance divided by the mean importance (Unsloth's imatrix, the one 074 and the llama.cpp
files were calibrated on), divide by the squared norm of the original weights and take the
square root. That noise-equivalent error goes through the same fitted Gaussian curve. Nothing
is fitted to the measured GGUF KL.

| File | Unweighted | Weighted | Sum of isolated | Joint | Unweighted / isolated | Weighted / isolated | Weighted / joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary factors, before distillation | 8.536 | 3.179 | 2.956 | 1.815 | 2.89 | 1.08 | 1.75 |
| Binary factors, after distillation | 8.573 | 3.220 | 2.970 | 1.504 | 2.89 | 1.08 | 2.14 |
| UD-IQ2_XXS | 5.217 | 2.635 | 2.115 | 1.737 | 2.47 | 1.25 | 1.52 |
| Q2_K (Unsloth) | 1.976 | 1.187 | 1.082 | 0.943 | 1.83 | 1.10 | 1.26 |
| Q3_K_M (Unsloth) | 0.912 | 0.491 | 0.417 | 0.358 | 2.19 | 1.18 | 1.37 |
| Q4_K_M (Unsloth) | 0.396 | 0.244 | 0.206 | 0.181 | 1.92 | 1.18 | 1.35 |
| Q8_0 | 0.0063 | 0.0064 | 0.0051 | 0.0047 | 1.24 | 1.24 | 1.36 |
| Q4_K_M, calibrated (local) | 0.396 | 0.244 | 0.206 | 0.181 | 1.92 | 1.18 | 1.35 |
| Q4_K_M, no calibration | 0.384 | 0.389 | 0.331 | 0.298 | 1.16 | 1.17 | 1.31 |
| Q3_K_M, calibrated (local) | 1.059 | 0.648 | 0.541 | 0.460 | 1.96 | 1.20 | 1.41 |
| Q3_K_M, no calibration | 1.046 | 1.044 | 0.837 | 0.765 | 1.25 | 1.25 | 1.37 |
| Q2_K, calibrated (local) | 2.123 | 1.344 | 1.206 | 1.071 | 1.76 | 1.11 | 1.25 |
| Q2_K, no calibration | 2.113 | 2.054 | 1.750 | 1.656 | 1.21 | 1.17 | 1.24 |

Points to make:

- **Calibrated files:** the unweighted prediction is 1.76–2.89× the isolated sum; the weighted
  one is 1.08–1.25×. For 074 after distillation the overestimate falls from 189% to 8.4%.
- **Uncalibrated files and Q8_0:** weighting changes nothing (1.16–1.25× either way). Their
  error is not steered by the imatrix, so the weighted and unweighted errors agree. The
  leftover 16–25% matches the block-scaling factor (0.80–0.90) from Findings 8 and 9.
- **Ranking.** Spearman correlation between predicted and measured isolated KL over 126
  projections: 074 0.84 → 0.96; UD-IQ2_XXS 0.93 → 0.97; Q2_K 0.91 → 0.96; Q3_K_M 0.82 → 0.89;
  Q4_K_M 0.92 → 0.95. For 074 the weighted estimate names the same five largest blocks as the
  measured block KLs, with blocks 1, 0 and 3 in the right order; the unweighted one puts
  block 11 in its top five.
- **It does not see interaction.** Weighted / joint is 1.24–1.52 for the llama.cpp files and
  1.75–2.14 for 074. Through distillation the weighted sum rises slightly (3.179 → 3.220)
  while the joint KL falls 17.1%.
- **Use:** prioritising blocks and projections without running the model. Whole-model error
  still needs the interaction.
- **Caveat: same imatrix.** The weighting uses the imatrix the quantizers optimised against.
  An imatrix from different text would be the independent test. Not run.

## Finding 6: weight error is not the whole story

- Distillation left the binary-factor model's weight error unchanged (23.5% median before and
  after) and cut KL by 17% (1.815 → 1.504) and NLL change from +1.17 to +0.44 nats/token.
  Same distance from the original weights, different direction.
- NLL (perplexity) can hide damage that KL sees. Q3_K_M shows an NLL change of only +0.018
  nats/token (+1.8% perplexity) with a KL of 0.358; Q4_K_M has half the KL (0.184) but a
  larger NLL change (+0.042). Token-level gains and losses cancel in NLL; they cannot cancel
  in KL. Use KL to measure fidelity.

## Finding 7 (sidebar): measure in FP32

- The first 270M sweep ran in BF16. At 1% error its median KL was 0.0028 nats/token; the FP32
  median is 0.000092. That is a 30× inflation from arithmetic noise alone.
- The BF16 floor sat near 0.003–0.004 nats/token even at 0.01% error, so low-error points were
  flat and the fitted exponent came out at 1.05 (median) instead of 2.0.
- FP32 with TF32 disabled removed it: the local slope between 1% and 2.5% error has median 2.00.
- Lesson: when the quantity you are measuring is 1e-4 nats/token, the evaluation precision is
  part of the experiment.

## Caveats to state plainly

- One model, 270M parameters. A 1B sweep (FP16, 182 matrices, 5%–35% error) gives a median
  exponent of 1.94, so the square law is not unique to 270M, but the joint and GGUF tests have
  only been run on 270M.
- 4,080 token positions from one calibration set. Enough for stable means, small for anything
  per-token.
- Gaussian noise is a probe, not a quantizer. Exact additivity (0.98–0.99) is proven for
  Gaussian noise only; real files come in at 0.82–0.92 (llama.cpp) and 0.51–0.61 (ours).
- Weight-only measurement. llama.cpp's runtime adds its own arithmetic differences.
- The binary-factor format is our own experimental work, not a released quantizer. Easy to
  cut from the article if it distracts.

## Things not to claim

- That the map tells you the best bit allocation. It tells you the cost side for one error
  shape; the rate side (how error falls with bits per matrix) is a separate measurement.
- That real quantization error is exactly additive. Measured: 8–18% sub-additive for the
  llama.cpp files, 39–49% for the binary-factor files.
- That calibration always gives 35–40%. That is three K-quant formats and one imatrix on one
  small model.
- That attention q and k "don't matter". They are 16% of total sensitivity.
- That results transfer to larger models unchanged.

## Structure of the article

1. Hook: a quantized model is 126 small approximations. Which ones matter?
2. The experiment in four sentences. What KL is.
3. Square law (figure: response curves).
4. The map (figure: sensitivity heatmaps, then the equal-KL heatmap).
5. Headline: it adds up (figure: additivity).
6. Real quantizers: where they put error (figure: error maps) and how they compare to the
   prediction (figure: predicted vs measured).
6b. Why real error is cheaper: one matrix at a time (figure: shape and interaction), then
    the calibration ablation (figure: calibration ablation), then the imatrix-weighted
    prediction (no figure).
7. Two things weight error does not tell you (distillation; NLL vs KL).
8. Sidebar: the BF16 trap.
9. Limits and what comes next.

## Medium formatting notes

- Medium has no tables and no math rendering. The article uses lists and figures instead, and
  writes the one formula as inline code.
- Images must be uploaded by hand; the article's image paths are relative to this folder.
- Six figures. If that is too many, drop the response-curve figure first.

## Title options

- Quantization Error Adds Up. Almost Exactly. (current)
- A Map of Where Quantization Hurts a Small LLM
- 126 Matrices, One Budget: Measuring Where Quantization Error Goes

## General-audience version

`article-general-audience.md` is a shorter second article for people who download and run
quantized models but do not work on them.

- Vocabulary: "table of numbers" for a weight matrix, "drift" for KL (named once), "rounding
  error" for weight error, "calibration" for the imatrix. No units in the text.
- Kept: the square law, the 87× spread and where sensitivity sits, additivity (98–99%), real
  files at a third to a half of the noise prediction, the calibration ablation (55–66% more
  drift without it), the Q2_K naming point, and practical advice for choosing a file.
- Dropped: the binary-factor format, distillation, the per-matrix and interaction split,
  block-scaled noise, per-format numbers, NLL against KL, and the BF16 trap.
- Figures: the 0.001 equal-KL heatmap, `joint-additivity.png`, and
  `calibration-drift-bars.png` (made for this version).
- The last paragraph refers to the technical article. Add the link once that one is published.

