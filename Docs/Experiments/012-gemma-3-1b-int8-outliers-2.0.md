# Experiment 012: Gemma 3 1B with 2.0% INT8 outliers

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/012-compress-and-benchmark-gemma-3-1b-it.py`
- Retained report:
  [`012-compress-and-benchmark-gemma-3-1b-it-quality.md`](../../Results/012/012-compress-and-benchmark-gemma-3-1b-it-quality.md)

## Question

How much quality could a tenfold larger INT8 outlier set recover, and what would it cost in effective bits?

## What we did

We raised the outlier fraction from Experiment 011's 0.2% to 2.0% while retaining the broader compression recipe.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.460 | 231.663 |
| Mean task accuracy | 0.6233 | 0.4658 |

Effective BPW rose to **1.156306**, compared with **1.010575** in Experiment 011.

## What we learned

The larger outlier set substantially improved perplexity but did not improve mean task accuracy. Its roughly
0.146-BPW premium was too expensive for comparisons intended to remain near one bit per weight. Quality levers can
have different effects on perplexity and downstream tasks, so neither metric nor nominal BPW can be considered alone.

## Disposition

Useful upper-bound evidence, but not adopted as the fixed-budget recipe.
