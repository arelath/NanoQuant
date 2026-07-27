# Experiment 003: Gemma 3 4B baseline

## Status

**Completed.**

- Model: `google/gemma-3-4b-it`
- Launcher: `experiments/003-compress-and-benchmark-gemma-3-4b-it.py`
- Retained report: [`003-gemma-3-4b-it-quality.md`](../../Results/003/003-gemma-3-4b-it-quality.md)
- Memory investigation: [Excessive VRAM issue](../ExcessiveVRamIssue.md)

## Question

Could the resident workflow scale to Gemma 3 4B within bounded memory, and what quality would the baseline recipe
produce?

## What we did

We compressed all 34 transformer blocks using the resident/offload path, exported the model, and ran the common BF16
versus candidate quality evaluation.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 51.639 | 84.107 |
| Mean task accuracy | 0.7042 | 0.5117 |

## What we learned

The 34-block workflow and the memory-residency fixes made a full 4B run practical. Feasibility did not imply quality:
candidate perplexity was about 63% worse than BF16 and task accuracy dropped substantially. This became the 4B
baseline for targeted rank experiments.

## Disposition

Operational success and quality baseline; not accepted as source-model parity.
