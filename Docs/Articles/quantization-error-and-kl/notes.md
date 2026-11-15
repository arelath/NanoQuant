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
- **We do not know why from this data.** Two candidate explanations that this experiment
  cannot separate: (a) each matrix's quantization error is less harmful than Gaussian noise of
  the same norm, because imatrix-weighted rounding pushes error into directions the inputs
  rarely excite; (b) structured errors in different matrices partly cancel. Swapping one
  matrix at a time from a GGUF would separate them. Not done yet. Do not claim either.
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
- Gaussian noise is a probe, not a quantizer. The additivity result is proven for Gaussian
  noise only.
- Weight-only measurement. llama.cpp's runtime adds its own arithmetic differences.
- The binary-factor format is our own experimental work, not a released quantizer. Easy to
  cut from the article if it distracts.

## Things not to claim

- That the map tells you the best bit allocation. It tells you the cost side for one error
  shape; the rate side (how error falls with bits per matrix) is a separate measurement.
- That real quantization error is additive. Untested.
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
