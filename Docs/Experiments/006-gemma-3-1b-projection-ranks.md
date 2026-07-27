# Experiment 006: Gemma 3 1B projection-specific ranks

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/006-compress-and-benchmark-gemma-3-1b-it.py`
- Retained report: [`006-gemma-3-1b-it-quality.md`](../../Results/006/006-gemma-3-1b-it-quality.md)

## Question

Would a full compression and tuning run with maximum value/key ranks and a 1.25-times query rank outperform the
original 1B recipe?

## What we did

Unlike Experiments 004 and 005, we applied the projection policy before factorization and ran the complete resident
compression, tuning, export, and quality workflow.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.901 | 297.337 |
| Mean task accuracy | 0.6258 | 0.4500 |

Effective BPW was **1.015863**. Perplexity improved substantially over Experiment 002's **453.571**, but remained
about 3.07 times BF16.

## What we learned

Projection-specific ranks can improve language modeling when incorporated before tuning. The gain did not close the
source-model gap, and task accuracy remained close to the old candidate. A useful rank policy still needed better
allocation and outlier handling.

## Disposition

Adopted as the next 1B baseline and transferred to the 270M model in Experiment 007.
