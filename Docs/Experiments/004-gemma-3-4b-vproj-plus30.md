# Experiment 004: Gemma 3 4B v-projection rank +30%

## Status

**Completed.**

- Model: `google/gemma-3-4b-it`
- Launcher: `experiments/004-gemma-3-4b-it-vproj-plus30.py`
- Retained report:
  [`004-gemma-3-4b-it-vproj-plus30-quality.md`](../../Results/004/004-gemma-3-4b-it-vproj-plus30-quality.md)

## Question

Would adding 30% rank to every attention `v_proj` improve the Experiment 003 candidate enough to justify its storage
cost?

## What we did

We increased all 34 value-projection ranks after the existing tuned baseline, left 204 non-target projections exact
relative to that candidate, repacked the model, and evaluated it.

## Results

- Packed size increased **0.8795%**.
- Weighted target reconstruction error fell about **28%**.
- WikiText-2 perplexity changed from **84.107** to **84.370**.
- Mean task accuracy changed from **0.5117** to **0.5025**.

## What we learned

A large local reconstruction improvement did not translate into model quality. Because the rank overlay was applied
after tuning, the result also warned that a modified factorization should be retuned before drawing a definitive
architectural conclusion. Local weighted error alone was not a sufficient selection metric.

## Disposition

Rejected as a direct post-training overlay. Its stale-tuning limitation motivated a full-pipeline rank experiment.
