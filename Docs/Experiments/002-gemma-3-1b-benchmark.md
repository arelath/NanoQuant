# Experiment 002: Gemma 3 1B benchmark

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/002-benchmark-gemma-3-1b-it.py`
- Retained report:
  [`002-gemma-3-1b-it-quality-benchmark.md`](../../Results/002/002-gemma-3-1b-it-quality-benchmark.md)

## Question

How did the accepted Experiment 001 candidate compare with the BF16 model under a common quality protocol?

## What we did

We evaluated the compressed candidate and BF16 model on the retained WikiText-2 slice and downstream task suite,
without changing the compressed artifact.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.901 | 453.571 |
| Mean task accuracy | 0.6258 | 0.4467 |

The candidate reproduced the legacy-quality target from Experiment 001, but was much worse than BF16.

## What we learned

A technically completed and legacy-matched candidate is not necessarily an acceptable model. Quality reports need a
named reference, protocol, and explicit acceptance gates; “completed” must describe workflow state rather than imply
quality approval.

## Disposition

Retained as the original 1B benchmark baseline for later rank, outlier, and allocation experiments.
