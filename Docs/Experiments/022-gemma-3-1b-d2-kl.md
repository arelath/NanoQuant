# Experiment 022: Gemma 3 1B exact-unit D2/KL allocation

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/022-d2-kl-compress-and-benchmark-gemma-3-1b-it.py`
- Retained report:
  [`022-d2-kl-compress-and-benchmark-gemma-3-1b-it-quality.md`](../../Results/022/022-d2-kl-compress-and-benchmark-gemma-3-1b-it-quality.md)

## Question

Would corrected exact-unit D2 improve the low-BPW 1B baseline from Experiment 017?

## What we did

We applied the corrected D2 allocator to the same model family and budget regime, then ran the complete resident,
tuning, export, and quality workflow.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.460 | 228.551 |
| Mean task accuracy | 0.6233 | 0.4692 |

Effective BPW was **1.024495**. Experiment 017 at **1.024487 BPW** had **274.912** perplexity and **0.4617** mean
task accuracy.

## What we learned

This was the clearest D2 result: a large perplexity improvement and a small task improvement at essentially identical
BPW. The corrected exact-unit signal generalized from diagnostic probes and 270M to a stronger 1B baseline.

## Disposition

Adopted as the best simple D2 1B recipe and the reference for Experiments 023 and 024.
