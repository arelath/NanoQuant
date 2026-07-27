# Experiment 009: Gemma 3 270M publishable workflow

## Status

**Completed compression, export, and publication workflow.** No standalone retained quality report is present in the
numbered results directory.

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/009-compress-benchmark-and-publish-gemma-3-270m-it.py`
- Retained artifacts: [`Results/009`](../../Results/009/)

## Question

Could a numbered experiment produce a complete, standardized, publishable artifact set rather than only research
state?

## What we did

We ran all **18 blocks / 126 layers**, produced the logical and packed model forms, exported GGUF, and exercised the
publication path. An earlier misnamed 1B attempt under the same experiment number was interrupted and intentionally
discarded rather than mixed with the 270M identity.

## Results

The resident run and GGUF/export artifacts completed. The retained directory does not contain a self-contained
quality report, so this record makes no new perplexity or task-quality claim.

## What we learned

Compression, export, and publication can be treated as explicit durable stages. Experiment identity is part of the
artifact contract: changing model or configuration requires rollover rather than silently reusing old resident
state. Publication success must not substitute for retained quality evidence.

## Disposition

Accepted as workflow evidence. Experiment 010 supplied the comparable local quality baseline.
