# Experiment 060: Gemma 3 1B best-methods compression at 2 BPW

## Status

Configured and awaiting the complete CUDA compression, export, and quality run.

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Launcher:
  `experiments/060-best-methods-2bpw-compress-and-benchmark-gemma-3-1b-it.py`
- Block-bounded campaign supervisor: `tools/run_experiment060_campaign.py`
- Baseline: Experiment 056, the completed physical-cap 1B run

## Question

How much quality can Experiment 059's validated best-methods recipe recover on
Gemma 3 1B when the complete quantized-layer representation is allowed up to
two charged bits per weight?

## Predeclared recipe

- exact-unit, same-campaign D2/KL allocation;
- calibration-weighted measured rank responses at the tuned 48-by-512-token
  operating point;
- a 1.5-times allocation response range with a strict physical-rank ceiling;
- stacked QKV with the retained 2-times value-member objective;
- the finite production-horizon `3e-5` binary learning rate;
- ordinary 800-step ADMM factorization, block tuning, post-block refit, and
  top-64 global distillation;
- 8.8% residual-selected INT8 columns with one BF16 scale per column, charged
  to the 2-BPW budget;
- at most 0.02 BPW of charged retry capacity, adding two residual columns only
  after a layer reaches its physical rank cap;
- complete logical/packed/GGUF export and the 64-by-128 WikiText plus
  six-task, 1,000-row quality benchmark.

The 1B projection shapes spend approximately 1.2576 BPW at full aligned
physical binary rank. The 8.8% outlier fraction is the largest simple
one-decimal-percent rate that keeps the conservative physical-rank plus
outlier plus retry bound below 2 BPW: approximately 1.9860 BPW. This is the
shape-matched counterpart to Experiment 059's 7% 270M policy.

## Deliberate exclusions

As in Experiment 059, product codebooks, over-complete binary rank, tail-aware
distillation, fixed mass-floor correction, bias correction, low-rank patches,
late structured sidecars, and direct binary search remain disabled because
their completed or held-out gates did not support general promotion.

## Completion and quality gate

The experiment is complete only after the resident journal has a coherent
26-block terminal prefix, strict artifact validation passes, the packed model
reloads, GGUF export succeeds, and the configured quality benchmark produces
finite results. Report actual effective BPW, WikiText perplexity, every task
score, wall time, memory, and artifact sizes. Compare with Experiment 056 and
the same-run BF16 reference; do not promote from reconstruction loss alone.
