# Experiment 024: Gemma 3 1B combined best-methods recipe

## Status

**Completed.**

- Model: `google/gemma-3-1b-it`
- Launcher: `experiments/024-best-methods-compress-and-benchmark-gemma-3-1b-it.py`
- Retained report:
  [`024-best-methods-compress-and-benchmark-gemma-3-1b-it-quality.md`](../../Results/024/024-best-methods-compress-and-benchmark-gemma-3-1b-it-quality.md)

## Question

Would the best individually plausible ideas combine into a stronger final Gemma 3 1B recipe?

## What we did

We combined raw exact-unit KL/D2 measured on the tuned 48-by-512-token protocol, stacked QKV with a two-times value
objective, and global distillation. We excluded the rejected bias and residual-patch ideas. Task evaluation was
expanded to 1,000 rows, so its aggregate is not directly comparable to the earlier short task protocol.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 96.460 | 235.283 |
| Mean task accuracy, 1,000-row protocol | 0.6057 | 0.4635 |

Effective BPW was **1.024463**. Perplexity was slightly worse than Experiment 022's simpler **228.551** result at
essentially the same budget.

## What we learned

Plausible local improvements were not additive. The combined recipe failed to surpass the simpler D2 allocation, and
the longer task protocol confirmed a substantial source-model gap with less sampling noise. Larger evaluation sets
are valuable precisely because they make small apparent gains harder to overinterpret.

## Disposition

Not adopted over Experiment 022. Retained as the controlled synthesis and broader task-evaluation result.
