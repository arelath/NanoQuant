# Experiment 005: maximum-rank `v_proj` double-check

## Question

Experiment 004 found that 30% more `v_proj` bits lowered local reconstruction error but did not improve quality.
Before rejecting that allocation direction, Experiment 005 requested twice the Experiment 003 `v_proj` bits.

## Physical limit

A true 2x allocation is not representable because every Gemma 3 4B `v_proj` has shape 1024 × 2560 and therefore a
maximum factor rank of 1024. The double request saturated all 34 layers at rank 1024 before reaching its requested
size.

- original ranks: 608–800, mean 714.35;
- candidate ranks: all 1024;
- aggregate achieved `v_proj` bit multiplier: 1.416214x;
- per-layer achieved multiplier: 1.270007x–1.652438x;
- complete packed decoder growth: 1.1762%.

This is the maximum-rank upper bound for the existing binary-factor representation.

## Integrity and artifacts

The candidate derives directly from the final globally tuned Experiment 003 state, not from Experiment 004.

- all 204 non-`v_proj` layers remained tensor-exact;
- all 34 expanded layers retained their original factors/scales as exact prefixes;
- source packed bytes: 402,802,128;
- candidate packed bytes: 407,539,832;
- GGUF bytes: 1,133,006,912;
- GGUF token embedding: Q8_0;
- GGUF SHA-256: `ee58e769c085f2a8643e5d6413406e1af88e334a66355e3c460f760133e1c2ab`.

## Reconstruction

All 34 target layers improved again.

| Metric | Experiment 003 state | Maximum-rank state | Relative change |
| --- | ---: | ---: | ---: |
| Mean weighted normalized error | 0.282931 | 0.183467 | -35.16% |
| Mean raw normalized error | 0.324355 | 0.209429 | -35.43% |
| Sum of weighted error | 1,504.654130 | 980.551789 | -34.83% |

## Matched quality

Experiment 003 and Experiment 005 used identical WikiText tokens, task rows, tokenizer, sample counts, sequence
length, and dense replay backend. Their BF16 results were identical.

| Benchmark | Experiment 003 | Maximum-rank `v_proj` | Delta |
| --- | ---: | ---: | ---: |
| WikiText-2 perplexity (lower is better) | 84.106649 | 85.882714 | +1.776065 (+2.11%) |
| PIQA `acc_norm` | 0.660 | 0.645 | -0.015 |
| ARC Easy `acc_norm` | 0.450 | 0.425 | -0.025 |
| ARC Challenge `acc_norm` | 0.300 | 0.285 | -0.015 |
| HellaSwag `acc_norm` | 0.480 | 0.465 | -0.015 |
| Winogrande `acc` | 0.535 | 0.585 | +0.050 |
| BoolQ `acc` | 0.645 | 0.645 | 0.000 |

Peak WDDM shared memory was 220,200,960 bytes, below the 805,306,368-byte limit.

## Decision

Reject uniform additive `v_proj` expansion for the base recipe. The maximum representable expansion strengthens the
negative result: it lowers local reconstruction error more than Experiment 004, yet degrades held-out perplexity
more severely and does not improve the task aggregate.

Future allocation experiments should optimize a downstream block/model loss reduction per stored byte, use held-out
selection, and keep total BPW fixed by trading bytes between layer families. Local weight reconstruction error is not
a sufficient allocation utility for this final globally tuned model.
