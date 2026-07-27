# Experiment 018: Gemma 3 4B tempered-allocation transfer

## Status

**Partial; no valid quality result.**

- Model: `google/gemma-3-4b-it`
- Launcher: `experiments/018-compress-and-benchmark-gemma-3-4b-it.py`
- Retained state: [`Results/018`](../../Results/018/)

## Question

Would the tempered architecture-aware policy from Experiment 017 improve the 4B model?

## What we did

We began a bounded-memory 4B compression using the transferred allocator. The retained run completed **30 of 170
physical units** and **5 of 34 blocks** before interruption.

## Results

There is no completed candidate or comparable quality report.

## What we learned

The experiment supplied additional bounded-memory execution experience, but not an allocation-quality answer.
Partial reconstruction statistics cannot validate a policy intended to improve model-level behavior.

## Disposition

Inconclusive. Do not cite it as evidence for or against the 4B transfer.
