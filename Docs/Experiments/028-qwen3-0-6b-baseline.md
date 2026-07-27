# Experiment 028: Qwen3 0.6B baseline

## Status

**Completed.**

- Model: `Qwen/Qwen3-0.6B`
- Launcher: `experiments/028-compress-and-benchmark-qwen3-0-6b.py`
- Retained report:
  [`028-compress-and-benchmark-qwen3-0-6b-quality.md`](../../Results/028/028-compress-and-benchmark-qwen3-0-6b-quality.md)

## Question

Could the generic compression workflow produce and evaluate a deployment-valid Qwen3 0.6B model?

## What we did

We compressed all 28 blocks, exported a GGUF, and made llama.cpp evaluation deployment-authoritative. The evaluation
used the generic perplexity/task protocol and did not separately exercise Qwen3 thinking and non-thinking behavior.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 55.167 | 292.264 |
| Mean task accuracy | 0.5608 | 0.4483 |

- Effective BPW: **1.028017**
- GGUF size: **393,472,384 bytes**
- Storage reduction from BF16 tensor bytes: **73.83%**

## What we learned

The Qwen architecture, GGUF export, and deployment runtime worked end to end, but quality was substantially degraded.
More importantly, the generic benchmark was behavior-blind: it could not detect that Qwen3 defaults to thinking mode
or that a calibration recipe without coherent thinking traces could damage that mode. This gap directly motivated
Experiment 030.

## Disposition

Accepted as the Qwen3 deployment and generic-quality baseline, not as a thinking-capable quality result.
