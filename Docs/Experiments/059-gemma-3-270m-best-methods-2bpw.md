# Experiment 059: Gemma 3 270M best-methods compression at 2 BPW

## Status

Completed successfully on 2026-08-12. The resident journal, transitive artifact
graph, packed export, GGUF, and long quality benchmark are complete. Strict
validation passed for all 18 blocks, 90 quantized layers, 108 active journal
records, and 582 reachable artifacts under one commit identity.

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

## Results

- Charged effective rate: `1.9751935 BPW` over 100,270,080 quantized
  parameters, below the 2.0-BPW ceiling.
- Charged payload: 137,134,080 binary-factor bits, 56,503,008 INT8 outlier
  value bits, 65,772 outlier-index bits, and 4,349,952 scale bits.
- Packed quantized-layer payload: 24,869,300 bytes.
- Complete GGUF: 209,833,888 bytes (60.87% smaller than the 536,223,056-byte
  BF16 checkpoint tensor payload).
- Resident block wall time: 7,181.14 seconds. The uninterrupted final slice
  spent 1,718.72 seconds in global distillation and 1,082.07 seconds in the
  quality workflow.
- Compression peak resource measurements: 4,490,002,432 GPU bytes and
  17,901,568,000 host bytes. The PyTorch quality-reference backend peaked at
  7,530,872,832 CUDA allocator bytes.
- Global top-64 distillation completed 2,048 steps; the final epoch mean loss
  was 1.836261.

| Benchmark | BF16 | Experiment 059 | Delta | Retention |
| --- | ---: | ---: | ---: | ---: |
| WikiText-2 perplexity (lower is better) | 194.053752 | 218.326427 | +12.51% | 1.1251x |
| PIQA acc_norm | 0.678 | 0.592 | -0.086 | 87.32% |
| ARC Easy acc_norm | 0.514 | 0.364 | -0.150 | 70.82% |
| ARC Challenge acc_norm | 0.265 | 0.203 | -0.062 | 76.60% |
| HellaSwag acc_norm | 0.437 | 0.384 | -0.053 | 87.87% |
| Winogrande acc | 0.524 | 0.500 | -0.024 | 95.42% |
| BoolQ acc | 0.578 | 0.499 | -0.079 | 86.33% |
| Six-task arithmetic mean | 0.4993 | 0.4237 | -0.0757 | 84.85% |

Experiment 021's directly comparable WikiText perplexity was 1,141.159795, so
Experiment 059 reduces that perplexity by 80.87%. Its six-task mean was 0.3983
versus 0.4237 here, but that task comparison is directional only because
Experiment 021 evaluated 200 rows per task while Experiment 059 evaluated
1,000. The WikiText input hash and protocol are identical.

## Disposition

The experiment validates the combined recipe and shows a large improvement
over the 1-BPW Experiment 021 baseline, but it does not establish BF16-quality
parity: WikiText remains 12.51% worse, and the largest task loss is 0.150 on ARC
Easy. Retain Experiment 059 as the current 2-BPW best-methods baseline and as a
source of allocator/outlier evidence; do not promote it as a quality-parity
production default without an explicit acceptance threshold that permits
these losses.

Published evidence:

- `Results/059/059-best-methods-2bpw-compress-and-benchmark-gemma-3-270m-it-quality.md`
- `Results/059/059-best-methods-2bpw-compress-and-benchmark-gemma-3-270m-it-summary.json`
- `Results/059/gemma-3-270m-it-nanoquant.gguf`
- `evidence/059/059-best-methods-2bpw-compress-and-benchmark-gemma-3-270m-it/validation-complete.json`
