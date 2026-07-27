# Experiment 010: Gemma 3 270M cubic update

## Status

**Completed.**

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/010-compress-and-benchmark-gemma-3-270m-it.py`
- Retained report:
  [`010-compress-and-benchmark-gemma-3-270m-it-quality.md`](../../Results/010/010-compress-and-benchmark-gemma-3-270m-it-quality.md)

## Question

Would the revised cubic ADMM/update behavior improve the initial 270M compression result?

## What we did

We reran the complete 270M recipe locally with the updated numerical path and the same general one-bit quality
protocol used by Experiment 007.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 194.054 | 1,133.506 |
| Mean task accuracy | 0.5208 | 0.4242 |

## What we learned

The update improved perplexity materially from Experiment 007's **1,546.252** to **1,133.506**, confirming that the
numerical recipe mattered. The candidate was still about 5.84 times BF16, so the update became a better baseline
rather than a sufficient fix.

## Disposition

Adopted as the main 270M comparison baseline for Experiments 013 through 016.
