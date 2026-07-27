# Experiment 014: Gemma 3 270M reconstruction-aware allocation

## Status

**Completed.**

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/014-compress-and-benchmark-gemma-3-270m-it.py`
- Retained report:
  [`014-compress-and-benchmark-gemma-3-270m-it-quality.md`](../../Results/014/014-compress-and-benchmark-gemma-3-270m-it-quality.md)

## Question

Could ranks be assigned from measured reconstruction benefit alone, without architecture-specific preferences?

## What we did

We used reconstruction-derived sensitivity to redistribute a fixed rank budget across units. The unconstrained policy
allocated too little capacity to important projections, including `down_proj` ranks around 288.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 194.054 | 2,018.475 |
| Mean task accuracy | 0.5208 | 0.3858 |

Effective BPW was **1.025112**.

## What we learned

Local reconstruction benefit is not equivalent to functional importance. A globally “optimal” local allocator can
starve architecture-critical transformations and severely regress end quality. Allocation needs architectural floors
or priors in addition to measured error reduction.

## Disposition

Rejected. The failure directly motivated the protected architecture policy in Experiment 015.
