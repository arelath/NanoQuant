# Experiment 019: First Llama 3.2 1B transfer

## Status

**Partial; no valid quality result.**

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Launcher: `experiments/019-compress-and-benchmark-llama-3-2-1b-instruct.py`
- Retained state: [`Results/019`](../../Results/019/)

## Question

Could the Gemma-derived resident recipe transfer to a different model family and architecture?

## What we did

We adapted layer discovery, rank policy, calibration, and export to Llama 3.2 1B. The run reached **22 of 80 physical
units** and **4 of 16 blocks** before interruption.

## Results

No completed compressed model or quality comparison was produced.

## What we learned

The first cross-architecture attempt exposed assumptions that needed more robust model adaptation and execution, but
it was too incomplete to characterize Llama quality. Experiment 025 later replaced it with a successful full
replication.

## Disposition

Superseded by Experiment 025; retain only as historical implementation evidence.
