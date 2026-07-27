# Experiment 017: Gemma 3 1B tempered architecture allocation

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/017-compress-and-benchmark-gemma-3-1b-it.py`
- Retained report:
  [`017-compress-and-benchmark-gemma-3-1b-it-quality.md`](../../Results/017/017-compress-and-benchmark-gemma-3-1b-it-quality.md)

## Question

Would a tempered architecture-aware allocator transfer usefully from the 270M studies to Gemma 3 1B?

## What we did

We transferred the protected allocation concept with a 0.5 sensitivity temper and ran the full 1B pipeline near the
one-bit target.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.460 | 274.912 |
| Mean task accuracy | 0.6233 | 0.4617 |

Effective BPW was **1.024487**. It improved on Experiment 011's **283.548** perplexity with a modest bit increase,
but did not match Experiment 012's **231.663** perplexity at that experiment's much higher 1.156-BPW cost.

## What we learned

Tempered architecture-aware allocation was a useful low-BPW 1B operating point. It demonstrated the intended
quality/budget tradeoff more clearly than the 270M runs and became the proper baseline for KL-informed allocation.

## Disposition

Accepted as the pre-D2 low-BPW 1B baseline and transferred to larger and different architectures in 018 and 019.
