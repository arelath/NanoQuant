# Experiment 025: Llama 3.2 1B replication

## Status

**Completed.**

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Launcher: `experiments/025-compress-and-benchmark-llama-3-2-1b-instruct.py`
- Retained report:
  [`025-compress-and-benchmark-llama-3-2-1b-instruct-quality.md`](../../Results/025/025-compress-and-benchmark-llama-3-2-1b-instruct-quality.md)

## Question

Could the current recipe complete end to end on Llama 3.2 1B and replace the interrupted Experiment 019?

## What we did

We ran all 16 blocks through calibration, allocation, resident compression, tuning, export, upload-capable artifact
handling, resume checks, and comparative quality evaluation.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 36.856 | 116.980 |
| Mean task accuracy | 0.6092 | 0.5067 |

Effective BPW was **1.021823**.

## What we learned

The workflow generalized operationally beyond Gemma and completed the previously interrupted model. Quality did not
generalize nearly as well: candidate perplexity was about 3.17 times BF16 and task accuracy was lower. Cross-family
support requires source-quality measurement, not just compatible layer discovery and export.

## Disposition

Accepted as the completed Llama 3.2 1B replication and operational baseline; not accepted as quality parity.
