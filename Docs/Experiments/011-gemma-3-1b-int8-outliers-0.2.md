# Experiment 011: Gemma 3 1B with 0.2% INT8 outliers

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/011-compress-and-benchmark-gemma-3-1b-it.py`
- Retained report:
  [`011-compress-and-benchmark-gemma-3-1b-it-quality.md`](../../Results/011/011-compress-and-benchmark-gemma-3-1b-it-quality.md)

## Question

Would doubling the INT8 outlier allowance from 0.1% to 0.2% provide an economical improvement over Experiment 006?

## What we did

We kept the projection-rank recipe and increased the small high-precision outlier set, then reran compression,
tuning, export, and the same short quality protocol.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.460 | 283.548 |
| Mean task accuracy | 0.6233 | 0.4675 |

Effective BPW was **1.010575**. Relative to Experiment 006, perplexity improved from **297.337** and mean task
accuracy improved from **0.4500**.

## What we learned

A very small increase in INT8 outliers was a comparatively efficient quality lever. Both language modeling and the
aggregate task score moved in the desired direction without a large departure from the one-bit target.

## Disposition

Accepted as a better 1B operating point and the reference for Experiment 012.
