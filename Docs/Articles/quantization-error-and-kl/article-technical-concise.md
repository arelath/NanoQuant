# Where Quantization Error Hurts: A Measured Map of a Small LLM

*We put a controlled amount of error into each weight matrix of Gemma 3 270M, measured how far
the output moved, and then checked the result against real quantized files.*

Quantizing a model rounds its weights so the file gets smaller. Every weight matrix ends up
slightly wrong. We measured what each of those errors costs, whether the costs add up, and how
real quantizers compare with the prediction.

## The short version

- The damage from error in one matrix grows with the square of the error.
- Matrices differ a lot. At the same error, the most sensitive one does 87 times the damage of
  the least sensitive.
- The damage adds. Random error in all 126 matrices at once costs 98–99% of the sum of the 126
  separate costs.
- Real quantized files do two to three times less damage than random error of the same size.
- Most of that advantage is calibration. With the same format and the same weight error, a
  calibrated file does 35–40% less damage.
- Weight each matrix's error by the calibration statistics and you can predict its cost to
  within 8–25%, without running the model.

## Setup and two definitions

Gemma 3 270M has 18 decoder blocks. Each block has seven weight matrices: attention query,
key, value and output, and MLP gate, up and down. That is 126 matrices.

**Weight error** is the size of the change to a matrix divided by the size of the matrix
(relative Frobenius norm). A 5% error means the change has 5% of the matrix's norm.

**KL divergence** is how far the changed model's next-token probabilities move from the
original model's. We average it over token positions and report it in nats per token. Zero
means identical predictions. It measures how faithful the copy is, not how good its answers
are.

Every KL figure below comes from the same 4,080 token positions of ordinary text, over the
full vocabulary, with the model running in 32-bit floating point. The details are at the end.

## 1. Damage grows with the square of the error

We added Gaussian noise to one matrix at a time, at six error levels from 1% to 35%, with
three random seeds each. The other 125 matrices stayed untouched. That is 2,268 measurements.

Every matrix follows the same rule:

`KL ≈ c × error²`

The fitted exponent has a median of 1.975 across the 126 matrices and stays between 1.70 and
2.10. Double a matrix's error and its damage roughly quadruples.

![KL response curves for each projection type](figures/kl-error-response.png)

*Each line is one matrix, coloured by block. On a log scale the lines are nearly parallel. The
matrices share the square law and differ only in how high the line sits.*

Because the exponent is nearly the same everywhere, one number describes each matrix: the
coefficient `c`. That number is its sensitivity.

## 2. Sensitivity varies 87× across the model

At 5% error, the most sensitive matrix costs 87 times more KL than the least sensitive one.

- **By type.** The three MLP matrices hold 59% of the total sensitivity. MLP down alone holds
  25%. Attention query and key together hold 16%.
- **By depth.** Blocks 0, 1, 3 and 17 hold 44% between them. The middle blocks hold 2–3% each.
- **Exceptions.** The most sensitive matrix in the model is MLP up in block 3, not anything in
  block 0. Block 13's value matrix and block 5's query matrix are sensitive spots in rows that
  are otherwise tolerant.

The map is easier to use turned around: how much error can each matrix take before it costs a
fixed amount of KL?

![Weight-error tolerance of each matrix at 0.001 nats/token](figures/equal-kl-target-error-heatmap-0.001-nats.png)

*The error each matrix can take before it costs 0.001 nats/token on its own. Dark cells are
fragile, bright cells are tolerant.*

At 0.001 nats/token per matrix, the tolerance runs from 1.0% (block 3, MLP up) to 9.9% (block
16, attention query), with a median of 3.3%. A quantizer that gives every matrix the same
precision is off by up to a factor of ten.

## 3. The damage adds up

A per-matrix map is only useful if it predicts what happens when every matrix is wrong at
once. Errors could amplify each other through 18 blocks, or partly cancel.

We gave all 126 matrices, at the same time, the error that costs a fixed KL budget on its own.
If damage adds, the total is 126 times the budget. We ran 100 random draws at each of three
budgets.

- Budget 0.001 nats/token: predicted 0.126, measured 0.123 (97.6%).
- Budget 0.0025: predicted 0.315, measured 0.309 (98.1%).
- Budget 0.005: predicted 0.630, measured 0.624 (99.0%).

![Joint KL as a fraction of the additive prediction, 100 runs per target](figures/joint-additivity.png)

*Each dot is one run with all 126 matrices perturbed. The black marker is the mean with its
95% confidence interval.*

The total is 1–2% below the sum. Single runs scatter about 5% either way, so it takes many
runs to see that.

For very small errors this is expected. KL then behaves like a squared distance, and
independent changes add in a squared distance. The surprise is that it still holds at 0.62
nats/token, more than three times the damage of the 4-bit file below.

So KL works like a budget. You can measure each matrix's cost separately and add the costs.

## 4. Real quantizers beat the prediction by 2–3×

Gaussian noise is a probe, not a quantizer. So we took five published GGUF files and two
versions of an experimental binary-factor format of our own, converted their weights back to
full precision, and ran them through the same test. For each file we compared the measured KL
with the additive prediction for its 126 measured weight errors.

Bits are per weight, averaged over the 126 matrices. Error is the median over the matrices.

- **Q8_0:** 8.50 bits, 0.57% error. KL 0.0047, which is 74% of the prediction.
- **Q4_K_M:** 5.43 bits, 4.5% error. KL 0.181, 46% of the prediction.
- **Q3_K_M:** 4.54 bits, 7.9% error. KL 0.358, 39%.
- **Q2_K:** 4.15 bits, 8.1% error. KL 0.943, 48%.
- **UD-IQ2_XXS:** 3.79 bits, 8.1% error. KL 1.737, 33%.
- **Binary factors:** 2.82 bits, 23.5% error. KL 1.815, 21%.
- **Binary factors after distillation:** 2.82 bits, 23.6% error. KL 1.504, 18%.

![Measured KL against the additive prediction for Gaussian noise and real quantizers](figures/predicted-vs-measured-kl.png)

*On the dashed line, measured damage equals the additive prediction. Gaussian noise sits on
it. Real quantizers sit below it.*

Three things follow.

- The prediction has been a ceiling. No file did more damage than predicted.
- It ranks the five llama.cpp files in the right order.
- It does not rank across quantizer families. It predicts the binary-factor file to be worse
  than UD-IQ2_XXS, and after distillation that file measures better, at fewer bits.

The files also differ in where they spend bits. Q4_K_M gives extra bits to a fixed pattern of
blocks (0, 1, 4, 7, 10, 13, 15, 16, 17), which misses the sensitive block 3. UD-IQ2_XXS gives
its MLP down matrices a larger format in blocks 0, 1, 3, 5 and 17. The four most sensitive MLP
down blocks on our map are 0, 17, 3 and 1.

## 5. The gap has two parts

A file can beat the prediction in two ways. Each matrix's error can be less harmful than
random noise of the same size. Or the errors can cost less together than apart.

To separate them, we loaded one quantized matrix at a time into the otherwise original model
and measured its KL alone. That is 126 more measurements per file, and it gives two factors.

- **Per-matrix factor:** the sum of the 126 isolated costs, divided by the Gaussian prediction.
- **Interaction factor:** the whole-file KL, divided by the sum of the isolated costs.

Their product is the measured-to-predicted ratio from the last section.

- **Q8_0:** 0.81 × 0.92 = 0.74
- **Q4_K_M:** 0.52 × 0.88 = 0.46
- **Q3_K_M:** 0.46 × 0.86 = 0.39
- **Q2_K:** 0.55 × 0.87 = 0.48
- **UD-IQ2_XXS:** 0.41 × 0.82 = 0.33
- **Binary factors:** 0.35 × 0.61 = 0.21
- **Binary factors after distillation:** 0.35 × 0.51 = 0.18

![How each file's gap to the prediction splits into a per-matrix part and an interaction part](figures/shape-and-interaction.png)

*Each bar is the Gaussian prediction for one file. Blue is the KL the file actually costs.
Orange is what was saved inside individual matrices, green what was saved when they were
combined.*

For the four low-bit llama.cpp files, about nine tenths of the gap is the per-matrix factor.
It depends on the matrix type. Quantized query and key matrices cost about a quarter of the
Gaussian figure, the MLP matrices about half, and the value matrices nearly all of it.

## 6. The per-matrix saving is calibration

There were two candidate explanations for the per-matrix factor.

**Block scaling explains little.** A block quantizer gives each short run of weights its own
scale, so its error is small where the weights are small. We tested that alone, using random
noise scaled per block of 32 weights with no quantizer involved. It costs 89% of the Gaussian
figure. Q8_0, the one format here that ignores calibration data, sits at 81%. Every format
below 8 bits sits between 32% and 62%.

**Calibration explains the rest.** llama.cpp can quantize with an importance matrix: for each
matrix, statistics on how strongly each input is driven, collected by running the model on
sample text. We quantized the model ourselves to three formats, each one with and without it.
The calibration text is not our test text.

- **Q4_K_M:** weight error 4.51% with calibration and 4.54% without. KL 0.181 and 0.298.
  Per-matrix factor 0.52 and 0.86.
- **Q3_K_M:** weight error 8.66% and 8.80%. KL 0.460 and 0.765. Per-matrix factor 0.51 and
  0.80.
- **Q2_K:** weight error 9.12% and 9.20%. KL 1.071 and 1.656. Per-matrix factor 0.57 and 0.83.

![The same split for three formats quantized with and without calibration data](figures/calibration-ablation.png)

*Each pair is one format quantized twice. The weight error is the same within a pair. Only the
calibrated file gets the large per-matrix saving.*

Calibration leaves the size of the error alone and cuts KL by 35–40%. It moves the error onto
inputs the model rarely drives hard. It helps most on MLP down and on query and key, and does
almost nothing for the value matrices.

That accounts for the whole gap in a calibrated Q4_K_M file: block scaling leaves 0.86 of the
Gaussian prediction, calibration leaves 0.60 of that, and interaction leaves 0.88 of that. The
product is 0.46.

## 7. Importance-weighted error predicts the per-matrix cost

If calibration works by moving error off the important inputs, the same importance matrix
should tell us what a given error costs. The recipe:

1. For each input column of a matrix, sum the squared error in that column.
2. Multiply each column's sum by its importance, divided by the mean importance.
3. Add the columns up, divide by the squared norm of the original matrix, and take the square
   root.

The result is a "noise-equivalent error": the size of evenly spread noise that should cost the
same. We put it through the same fitted curves from section 1. Nothing is fitted to the real
files, and no model is run.

Compared with the measured sum of isolated costs:

- **Calibrated files:** the plain prediction is 1.8–2.9× too high. The weighted one is 8–25%
  too high.
- **Uncalibrated files and Q8_0:** both predictions are 16–25% too high. The weighting changes
  nothing because their error is not steered anywhere. The leftover matches the block-scaling
  effect.
- **Ranking:** for the binary-factor format, the rank correlation between predicted and
  measured cost over the 126 matrices rises from 0.84 to 0.96.

All the calibrated files used the same importance matrix that we weighted with. Repeating the
prediction with a second importance matrix that no quantizer saw gives 12–24% too high, nearly
the same.

The limit is interaction, which no per-matrix estimate can see. Against whole-file KL, the
weighted prediction is still 24–52% too high for the llama.cpp files and about double for the
binary-factor format. It is a tool for deciding which matrices to protect. It is not yet a
whole-model forecast.

## 8. Interaction is where trained formats gain

**llama.cpp files:** the whole file costs 82–92% of the sum of its isolated costs. Independent
random noise gives 93–99% in the same test, so part of that is ordinary. Within one block the
seven matrices combine to 95–100% of their sum. The saving builds up across depth.

**Binary factors:** 61% before distillation and 51% after. This format is fitted block by
block against the original model's activations, and each block sees the output of the
compressed blocks before it. Later matrices learn to compensate for earlier ones.

**Distillation changes only the interaction.** It left the median weight error at 23.5%. The
sum of isolated costs went from 2.96 to 2.97 nats/token. The whole-file KL fell from 1.82 to
1.50. No single matrix became a better copy. The matrices became better at covering for each
other.

## Three cautions

**File names can mislead on small models.** K-quant formats need matrix rows that divide into
blocks of 256 values. Five of this model's seven matrix types have 640 inputs, so llama.cpp
falls back to other formats. In the file named Q2_K, nothing is stored as Q2_K, and the
matrices average 4.15 bits per weight.

**Perplexity can hide damage.** Q3_K_M raises perplexity by 1.8% and Q4_K_M by 4.3%, yet
Q3_K_M has twice the KL. Perplexity averages over tokens, so positions where the quantized
model guesses better cancel positions where it guesses worse. KL counts every departure.

**Measure in full precision.** Our first sweep ran in BF16. Its arithmetic noise put a floor
of about 0.003 nats/token under every measurement. The real signal at 1% error is about 0.0001,
so the low-error points were flat and the fitted exponent came out near 1.

## Limits

- One model, and a small one. A partial sweep of Gemma 3 1B gives the same square law (median
  exponent 1.94). The joint and quantizer tests have only been run on 270M.
- 4,080 token positions from one text collection.
- Additivity within 2% is shown for random noise only. Real files combine to 8–18% less than
  the sum (llama.cpp) or 39–49% less (binary factors).
- The calibration test covers three llama.cpp formats and one importance matrix.
- The second importance matrix comes from the same text collection as our test windows.
- All of Q8_0's weight errors are below 1%, the lowest level in the sweep, so its prediction
  is extrapolated.
- We measured weight error only. A real runtime adds its own arithmetic differences.
- The map gives the cost of error. Choosing bit widths also needs to know how fast error falls
  as a matrix gets more bits, which is a separate measurement.

## How we measured

- **Model:** `google/gemma-3-270m-it`, hidden size 640, evaluated in FP32 with TF32 disabled.
  The clean model reproduces its own stored outputs to 3e-7 nats/token.
- **Text:** 16 windows of 256 tokens from `eaddario/imatrix-calibration`, giving 4,080 scored
  next-token positions.
- **Metric:** KL(original ‖ changed) over the full 262,144-token vocabulary, averaged per
  position.
- **Noise:** isotropic Gaussian, scaled so its Frobenius norm is the stated fraction of the
  matrix's norm.
- **Fits:** one power law per matrix, `KL = c × error^p`, fitted by least squares in log space
  to the seed-averaged KL at the six error levels. The additive prediction for a file is the
  sum of the 126 fitted curves at that file's measured errors.
- **Published files:** `unsloth/gemma-3-270m-it-GGUF`, all built with Unsloth's importance
  matrix. Weights were dequantized and loaded into the PyTorch model, so only the 126 matrices
  differ from the original.
- **Our own quantizations:** stock llama.cpp, the same importance matrix, and the same
  full-precision source file for both members of each pair. Our Q4_K_M matches the published
  one. Our Q3_K_M and Q2_K fall back to a different format for the 640-input matrices, so their
  errors differ slightly from the published files.
