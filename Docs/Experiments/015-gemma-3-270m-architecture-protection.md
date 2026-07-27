# Experiment 015: Gemma 3 270M architecture-protected allocation

## Status

**Completed.**

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/015-compress-and-benchmark-gemma-3-270m-it.py`
- Retained report:
  [`015-compress-and-benchmark-gemma-3-270m-it-quality.md`](../../Results/015/015-compress-and-benchmark-gemma-3-270m-it-quality.md)

## Question

Would architecture-aware protection repair the severe regression from pure reconstruction allocation?

## What we did

We protected Q/K/V/O and `down_proj` projections plus edge blocks, and tempered the sensitivity contribution to
0.25 while maintaining a similar overall bit budget.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 194.054 | 1,214.205 |
| Mean task accuracy | 0.5208 | 0.4100 |

Effective BPW was **1.025377**.

## What we learned

Architecture priors recovered most of Experiment 014's catastrophic loss, confirming that functional role must
constrain reconstruction-driven allocation. The result still did not beat Experiment 010's **1,133.506**
perplexity, so protection repaired the allocator but did not yet make it superior to the static baseline.

## Disposition

Retained as the best of the initial architecture-aware allocation candidates and the basis for Experiment 016.
