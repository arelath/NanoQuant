# Experiment 059: Gemma 3 270M best-methods compression at 2 BPW

## Status

Configured and awaiting the complete CUDA compression, export, and quality run.

- Model: `unsloth/gemma-3-270m-it`
- Revision: `23cf460f6bb16954176b3ddcc8d4f250501458a9`
- Launcher:
  `experiments/059-best-methods-2bpw-compress-and-benchmark-gemma-3-270m-it.py`
- Block-bounded campaign supervisor: `tools/run_experiment059_campaign.py`
- Baseline: Experiment 021, the completed 270M exact-unit D2 run at 1 BPW

## Question

How much quality can the best retained production recipe recover on Gemma 3
270M when the complete quantized-layer representation is allowed up to two
bits per weight instead of one?

## Predeclared recipe

- exact-unit, same-campaign D2/KL allocation;
- calibration-weighted measured rank responses at the tuned 48-by-512-token
  operating point;
- a 1.5-times allocation response range with a strict physical-rank ceiling;
- stacked QKV with the retained 2-times value-member objective;
- the finite production-horizon `3e-5` binary learning rate;
- ordinary 800-step ADMM factorization, block tuning, post-block refit, and
  top-64 global distillation;
- 7% residual-selected INT8 columns with one BF16 scale per column, charged to
  the 2-BPW budget;
- at most 0.02 BPW of charged retry capacity, adding two residual columns only
  after a layer reaches its physical rank cap;
- complete logical/packed/GGUF export and the 64-by-128 WikiText plus
  six-task, 1,000-row quality benchmark.

The 270M projection shapes cannot spend two bits on ordinary binary rank alone:
full aligned physical rank is approximately 1.411 BPW before residual columns.
The fixed INT8 fraction uses the otherwise stranded capacity, while the retry
reserve keeps the worst-case represented layer payload below 2 BPW. The final
packed manifest remains authoritative because alignment and the allocator's
exact per-layer choices can leave additional slack.

## Deliberate exclusions

This is a best-evidence synthesis, not a union of every mechanism tested.
Product codebooks, over-complete binary rank, tail-aware distillation, fixed
mass-floor correction, bias correction, low-rank patches, and late structured
sidecars remain disabled. Their completed or held-out gates did not support
general promotion. The standard product-codebook and binary-search fallbacks
are also disabled so Experiment 059 measures the retained ordinary packed
factor path rather than introducing another unproven variable at 2 BPW.

## Completion and quality gate

The experiment is complete only after the resident journal has a coherent
18-block terminal prefix, strict artifact validation passes, the packed model
reloads, GGUF export succeeds, and the configured quality benchmark produces
finite results. Report the actual effective BPW, WikiText perplexity, each task
score, task mean, wall time, memory, and artifact sizes. Compare quality with
Experiment 021 and the same-run BF16 reference; do not promote the recipe from
local reconstruction loss alone.
