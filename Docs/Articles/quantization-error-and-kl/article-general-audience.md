# What Breaks When You Shrink an AI Model?

*We damaged a small language model one piece at a time to see which pieces matter. The answer
turned out to be simpler than we expected.*

If you run AI models on your own computer, you have probably downloaded a "quantized" one. The
file names end in codes like Q4_K_M or Q8_0. A quantized model is a compressed copy. The
original stores every one of its numbers at high precision. The compressed copy rounds those
numbers off, so the file is smaller and runs on ordinary hardware.

Rounding has a cost. The compressed model no longer gives quite the same answers as the
original. We wanted to know where that cost comes from:

1. Does it matter which part of the model gets rounded?
2. If you know what each part costs, can you predict the cost of the whole thing?

We tested both on Gemma 3 270M, a small open model from Google.

## A model is a stack of number tables

Gemma 3 270M is built from 18 layers stacked on top of each other. Each layer holds seven
large tables of numbers, so there are 126 tables in all.

Four of the seven belong to the layer's attention step, which decides which earlier words
matter for predicting the next one. The other three belong to the step that follows it, called
the MLP, which processes what the model has gathered about each word.

Compressing a model means rounding every one of those 126 tables. Each table ends up slightly
wrong, and the question is how much each of those small errors matters.

## The experiment

We disturbed one table at a time. We added random noise to it, sized as a fraction of the
table itself: 1%, 5%, and so on up to 35%. The other 125 tables stayed exactly as they were.

Then we gave the original model and the damaged model the same text and compared their
predictions for the next word. The further the damaged model's predictions moved from the
original's, the more that table matters. We will call that movement **drift**. (Its technical
name is KL divergence.) Zero drift means the two models behave identically.

We repeated this for every table, at six noise levels, three times each: 2,268 measurements.

## Small errors are cheap and big errors are expensive

The first result is a simple rule. When the error in a table doubles, the drift it causes
roughly quadruples. Halve the error and the drift falls to about a quarter.

This held for every one of the 126 tables. It is why each step down in file size hurts more
than the one before. A lightly compressed model is almost indistinguishable from the original,
and the smallest files fall off quickly.

## Some parts matter far more than others

The tables are not equally important. With the same 5% error, the most sensitive table in the
model caused 87 times more drift than the least sensitive one.

![How much rounding error each of the 126 tables can tolerate](figures/equal-kl-target-error-heatmap-0.001-nats.png)

*Each cell is one table. Columns are the 18 layers and rows are the seven kinds of table. The
number is how much error that table can take before it causes the same small amount of drift.
Dark cells are fragile and bright cells are tolerant.*

Three patterns stand out.

- **The MLP tables carry most of the risk.** They account for 59% of the model's total
  sensitivity. The attention tables are more forgiving on the whole.
- **The first few layers and the last layer matter most.** Four of the 18 layers hold 44% of
  the sensitivity between them. The middle layers tolerate much more.
- **There are exceptions.** The single most fragile table is in the fourth layer, not the
  first. A simple rule like "protect the first and last layers" would miss it.

The most fragile table can take about 1% error. The most tolerant can take almost 10%. A
compressor that rounds every table the same way is being too careful with some and too rough
with others.

## The damage adds up

That map only helps if it predicts what happens when every table is wrong at once, which is
the situation in a real compressed model. Errors could build on each other as they pass
through 18 layers and make things worse than the sum of the parts. Or they could partly cancel.

So we disturbed all 126 tables together. Each one got exactly the amount of error that, on
its own, causes one small unit of drift. If the damage simply adds, the total should be 126
units.

We ran this 100 times, and then twice more at higher damage levels. The total came out at
98–99% of the sum.

![Total drift as a fraction of the sum of the parts, 100 trials at each of three damage levels](figures/joint-additivity.png)

*Each dot is one trial with all 126 tables disturbed. A dot at 1.0 means the total drift was
exactly the sum of the individual amounts. The black marker is the average.*

This was the result that surprised us. It means drift behaves like a budget. You can measure
what each part costs on its own, add the costs up, and know the total. 126 small measurements
tell you what the whole compressed model will do.

## Real compression tools do better, and calibration is why

Random noise is only a stand-in for rounding. So we also measured real compressed files that
people download and use.

They did better than the noise. For the same amount of error, the more heavily compressed
files caused between a third and a half of the drift that random noise would.

The reason is calibration. Before rounding anything, a good compression tool runs the model
on sample text and records which numbers the model leans on most. It then rounds carefully
where that matters and loosely where it does not.

To check that this was the cause, we compressed the model ourselves, twice for each of three
common recipes: once with calibration and once without. The two files in each pair were the
same size, and the amount of rounding error in them was practically identical. The drift was
not.

![Drift for three compression recipes, each done with and without calibration](figures/calibration-drift-bars.png)

*Each pair is the same recipe at the same file size. Without calibration, the drift is 55–66%
higher.*

Calibration does not make the rounding error smaller. It moves the same amount of error to
places where the model does not notice it.

## What this means if you download models

- **Prefer calibrated files.** Many uploaders say so in the description, often with the word
  "imatrix" (importance matrix). In our test, calibration gave 35–40% less drift at the same
  file size.
- **Do not judge a file by its error alone.** Two files with the same amount of rounding error
  differed in drift by more than half. Where the error sits matters as much as how big it is.
- **Each step smaller costs more than the last.** Because drift grows with the square of the
  error, the smallest files give up much more than the mid-sized ones.
- **Be careful with file names on small models.** On this model, the file labelled Q2_K
  contains no 2-bit data at all. The model's tables are the wrong shape for that format, so
  the tool quietly uses larger formats instead, and the tables average just over 4 bits per
  number.

## What we did not test

This is one small model and about 4,000 positions of test text. A partial run on a model four
times larger showed the same doubling-and-quadrupling rule, but we have not repeated the full
experiment there.

Drift measures how closely a compressed model copies the original. It does not measure whether
the model's answers are good. A copy can drift a little and still be perfectly useful, and
that judgement depends on what you use it for.

We have written a longer companion article with the full numbers, the charts for each
compressed file, and the details of how the measurements were made.
