# Experiment 013: Gemma 3 270M stacked QKV factorization

## Status

**Completed.**

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/013-compress-and-benchmark-gemma-3-270m-it.py`
- Retained report:
  [`013-compress-and-benchmark-gemma-3-270m-it-quality.md`](../../Results/013/013-compress-and-benchmark-gemma-3-270m-it-quality.md)

## Question

Would jointly factorizing the shared-input query, key, and value projections improve quality at a matched low-bit
budget?

## What we did

We introduced stacked QKV factorization under a fixed budget and ran the complete 270M workflow against the
Experiment 010 baseline.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 194.054 | 1,409.141 |
| Mean task accuracy | 0.5208 | 0.3967 |

Effective BPW was **1.025051**.

## What we learned

Shared inputs did not make joint QKV factorization a quality win by itself. Both perplexity and task accuracy were
worse than Experiment 010, despite the modestly higher budget. Structural elegance and local sharing need an
end-to-end quality gate.

## Disposition

Rejected as a standalone change. Later combinations changed its objective weighting rather than adopting this result.
