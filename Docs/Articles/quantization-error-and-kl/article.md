# Quantization Error Adds Up. Almost Exactly.

*We damaged a small language model one weight matrix at a time, mapped where the damage
shows up in its output, and found that the whole is within 2% of the sum of its parts.*

A quantized language model is not one approximation. It is a hundred or more small ones, one
per weight matrix, and they are not equally important. A 4-bit file and a 2-bit file differ
mostly in how much error each matrix is allowed to carry.

So two questions seemed worth measuring directly:

1. If you put the same amount of error into each matrix, which ones change the model's output
   the most?
2. If you know the answer for each matrix on its own, do you know the answer for all of them
   together?

The first question gives a map. The second decides whether the map is useful. If errors in
different matrices amplify each other, measuring them one at a time tells you little about a
real quantized model, where every matrix is wrong at once.

We ran both on Gemma 3 270M. The answer to the second question surprised us: the damage from
all 126 matrices together came out at 98–99% of the sum of the 126 separate measurements.

## The experiment

Gemma 3 270M has 18 decoder blocks. Each block has seven weight matrices: four in attention
(query, key, value and output) and three in the MLP (gate, up and down). That makes 126
matrices.

For each matrix we added random Gaussian noise, scaled so the noise was a fixed fraction of the
matrix's own size: 1%, 2.5%, 5%, 10%, 20% and 35%. A "5% error" means the noise has 5% of the
matrix's norm. Everything else in the model stayed untouched.

Then we ran the damaged model over 4,080 token positions of ordinary text and compared its
next-token predictions with the original model's. The comparison is KL divergence, measured in
nats per token. KL is zero when the two models assign identical probabilities to every possible
next token, and it grows as the damaged model's probabilities drift away from the original's.
It measures how faithfully the copy reproduces the original, not whether the copy is "good".

Six error levels, 126 matrices and three random seeds each come to 2,268 measurements.

## Damage grows with the square of the error

The first result is a simple law. For every matrix, the KL cost follows

`KL ≈ c × error²`

Across the 126 matrices the fitted exponent has a median of 1.975 and stays between 1.70 and
2.10. Double the error in a matrix and its damage roughly quadruples. Halve it and you get
three quarters of the damage back.

![KL response curves for each projection type](figures/kl-error-response.png)

*Each line is one matrix, coloured by block. On a log scale the lines are nearly parallel:
every matrix follows the same square law and differs only in how high its line sits.*

This means each matrix can be summarised by one number, the coefficient `c`. That number is
its sensitivity.

## The map

Sensitivity varies a lot. At the same 5% error, the most sensitive matrix in the model costs
87 times more KL than the least sensitive one.

![Measured KL sensitivity by block and projection at six error levels](figures/kl-sensitivity-heatmaps.png)

*KL cost of damaging one matrix, for each block (columns) and matrix type (rows), at six error
levels. Brighter is worse. The pattern is the same in every panel; only the scale changes.*

A few things stand out.

**The MLP matrices carry most of the risk.** At a uniform 5% error, the three MLP matrices
account for 59% of the total sensitivity. The MLP down matrix alone is 25%. The attention query
and key matrices together are 16%.

**The first blocks and the last block matter most.** Blocks 0, 1, 3 and 17 hold 44% of the
total sensitivity between them, out of 18 blocks. The middle and late-middle blocks are far
more forgiving, at 2–3% each.

**There are islands.** The single most sensitive matrix in the model is not in the first block.
It is the MLP up matrix of block 3. Block 13's value matrix and block 5's query matrix are
sensitive spots in rows that are otherwise tolerant. A rule like "protect the first and last
layers" would miss all three.

The map is easier to use when turned around. Instead of asking how much damage a fixed error
causes, ask how much error each matrix can absorb before it costs a fixed amount of KL.

![Weight-error tolerance of each matrix at 0.001 nats/token](figures/equal-kl-target-error-heatmap-0.001-nats.png)

*The error each matrix can take before it costs 0.001 nats/token on its own. Dark cells are
fragile, bright cells are tolerant.*

At a budget of 0.001 nats/token per matrix, block 3's MLP up matrix can take only 1.0% error.
Block 16's attention query matrix can take 9.9%. The median is 3.3%. A quantizer that gives
every matrix the same precision is overspending on some matrices and underspending on others
by a factor of ten.

## The headline: it adds up

Now the second question. We took the tolerance map above and applied it to every matrix at the
same time. Each of the 126 matrices got exactly the error that, on its own, costs 0.001
nats/token.

If the damage adds, the model should lose 126 × 0.001 = 0.126 nats/token. If errors amplify
each other as they pass through 18 blocks, it should lose more. If they partly cancel, less.

We ran it 100 times with different random noise, then repeated the whole thing at budgets of
0.0025 and 0.005 nats/token per matrix.

- Budget 0.001: predicted 0.126, measured 0.123. That is 97.6% of the sum.
- Budget 0.0025: predicted 0.315, measured 0.309. That is 98.1%.
- Budget 0.005: predicted 0.630, measured 0.624. That is 99.0%.

![Joint KL as a fraction of the additive prediction, 100 runs per target](figures/joint-additivity.png)

*Each dot is one run with all 126 matrices perturbed. The black marker is the mean with its 95%
confidence interval.*

The damage is slightly less than additive, by 1–2%, and the gap closes as the errors get
larger. Individual runs scatter about 5% either way, so we needed all 100 to see a 2% effect.
Our first attempt with three runs could not tell 0.98 from 1.0.

There is a reason to expect this for tiny errors. For a small enough change, KL behaves like a
squared distance, and independent random changes in different matrices contribute separately
to a squared distance. What we did not expect is that the rule would still hold at 0.62
nats/token, which is a badly damaged model: more than three times the damage of the 4-bit
file measured below.

The practical consequence is that 126 cheap, separate measurements predict the expensive
joint one. KL behaves like a budget. You can decide how much each matrix is allowed to spend
and know what the total will be.

## What real quantizers do with that budget

Random noise is a probe, not a quantizer. So we took five published GGUF quantizations of the
same model, converted their weights back to full precision, loaded them into the same test and
measured them the same way. This isolates the weight error from anything the inference runtime
does.

First, where each file puts its error:

![Relative weight error by block and projection for six quantized files](figures/quantizer-weight-error-maps.png)

*Weight error of each matrix in six quantized files, on one shared colour scale. Darker means
more error. The top-left panel is an experimental one-bit-factor format of our own.*

**File names are a poor guide on a small model.** The K-quant formats need matrix rows that
split evenly into blocks of 256 values. Five of Gemma 270M's seven matrix types have 640
inputs, which do not, so the quantizer silently falls back to other formats. In the file named
Q2_K, nothing is actually stored as Q2_K. Only the attention output and MLP down matrices get
a K-quant at all, and the matrices average 4.15 bits per weight. Compare files by measured
bits, not by name.

**Low-bit files concentrate their error.** The most aggressive file, UD-IQ2_XXS, keeps five
matrix types at about 4–8% error and puts 27% error on MLP down and 35% on attention output.
MLP down is the most sensitive matrix type on our map. By the additive prediction, those two
rows account for 89% of that file's damage.

**The "dynamic" file agrees with the map.** Within the MLP down row, UD-IQ2_XXS gives a larger
format to blocks 0, 1, 3, 5 and 17. The four most sensitive MLP down blocks on our map are 0,
17, 3 and 1. Four of its five protected blocks are our top four, found by a different route.

**The standard recipe is positional.** Q4_K_M gives extra bits to a fixed pattern of blocks: 0,
1, 4, 7, 10, 13, 15, 16 and 17. That catches the first two and the last, misses block 3, and
spends extra bits on blocks 4, 7 and 10, which sit in the lower half of the MLP down ranking.

Then the test of the prediction. For each file we summed the predicted KL for its 126 measured
errors and compared that with the KL the file actually produces.

![Measured KL against the additive prediction for Gaussian noise and real quantizers](figures/predicted-vs-measured-kl.png)

*On the dashed line, measured damage equals the additive prediction. Gaussian noise sits on it.
Real quantizers sit below it.*

Real quantization error is two to three times less damaging than random noise of the same
size. The four low-bit llama.cpp files land at 33–48% of the prediction. Our own
binary-factor format lands at about 20%.

So the map does two things well for real quantizers and one thing badly:

- It has acted as a ceiling so far. None of the seven files did more damage than predicted.
- It ranks files within one quantizer family correctly. All five llama.cpp files fall in the
  predicted order.
- It does not rank across families. It predicts our binary-factor file to be worse than
  UD-IQ2_XXS, and after distillation it measures better, at fewer bits per weight.

We do not yet know why real error is cheaper. One explanation is that these quantizers are
calibrated: they round each weight with knowledge of which inputs the matrix typically sees,
so the error lands in directions that rarely get used, while random noise is spread evenly
over all of them. Another is that structured errors in different matrices partly cancel in a way that
independent noise cannot. Our data does not separate the two. Swapping in one quantized matrix
at a time would, and that is the next experiment.

## Two things weight error does not tell you

**Direction matters as much as distance.** Our binary-factor model was measured before and
after a distillation pass, which trains the compressed model to imitate the original's
outputs. The weight error was the same before and after: 23.5% for the median matrix. The KL
dropped by 17%. The weights moved no closer to the originals, and the model's behaviour did.

**Perplexity can hide damage that KL sees.** The Q3_K_M file raises perplexity by only 1.8%,
which looks better than Q4_K_M's 4.3%. Its KL is twice as high. Perplexity averages over
tokens, so places where the quantized model happens to guess better cancel places where it
guesses worse. KL counts every departure from the original. If the question is "how faithful
is this copy", perplexity is the wrong ruler.

## A measurement trap: half precision

Our first sweep of this model ran in 16-bit floating point (BF16), because that is how the
model is usually loaded. The low-error results looked strange. KL barely moved between 1% and
2.5% error, and the fitted exponent came out near 1 instead of 2.

The cause was arithmetic noise. Running the model in BF16 adds its own rounding error at every
step, and that noise put a floor of about 0.003 nats/token under every measurement. At 1%
error the true signal is about 0.0001 nats/token, thirty times smaller than the floor.

Re-running in 32-bit precision removed the floor and the square law appeared cleanly, all the
way down. When the effect you are measuring is one ten-thousandth of a nat, the precision of
the evaluation is part of the experiment.

## Limits

- This is one model, and a small one. A partial sweep of Gemma 3 1B shows the same square law
  (median exponent 1.94), but we have only run the joint test and the quantizer comparison on
  270M.
- The test text is 4,080 token positions from one calibration set.
- Additivity is shown for random noise only. Whether real quantization error adds the same way
  is open.
- The map gives the cost of error in each matrix. Choosing bit widths also needs to know how
  fast error falls as each matrix gets more bits, which is a separate measurement.

## What to take away

- KL damage grows with the square of weight error, matrix by matrix.
- Sensitivity varies about 87× across the matrices of one small model, with the MLP matrices
  and the first few blocks carrying most of it.
- Damage from random error in all matrices adds up to within 2% of the sum of the parts, even
  for a heavily damaged model.
- Real quantizers do two to three times better than random error of the same size, so the
  additive sum is a ceiling rather than a forecast.
- Measure fidelity with KL, in full precision.
