# Experiment 023: Gemma 3 1B interaction-corrected D2

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/023-d2-kl-interaction-tuned-compress-and-benchmark-gemma-3-1b-it.py`
- Retained report:
  [`023-d2-kl-interaction-tuned-compress-and-benchmark-gemma-3-1b-it-quality.md`](../../Results/023/023-d2-kl-interaction-tuned-compress-and-benchmark-gemma-3-1b-it-quality.md)

## Question

Could explicit interaction correction and operating-point tuning improve the D2 allocation from Experiment 022?

## What we did

We added the interaction-normalized D2 variant, tuned its trust/operating point, and kept the effective bit budget
matched to the simple D2 run.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.460 | 237.234 |
| Mean task accuracy | 0.6233 | 0.4733 |

Effective BPW was **1.024423**. Compared with Experiment 022, perplexity worsened from **228.551**, while mean task
accuracy rose by about **0.0041**.

## What we learned

The extra correction shifted the tradeoff rather than clearly improving it. A small aggregate task gain did not
justify worse language-model perplexity and additional allocator complexity without stronger evidence.

## Disposition

Not adopted over the simpler exact-unit D2 recipe.
