# Experiment 026: Llama 3.2 3B scale-up

## Status

**Partial; preparation only.**

- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Launcher: `experiments/026-compress-and-benchmark-llama-3-2-3b-instruct.py`
- Retained state: [`Results/026`](../../Results/026/)

## Question

Would the completed 1B Llama workflow scale to the 3B member of the same family?

## What we did

Calibration/preparation and run initialization completed, but resident compression did not advance: the retained
state records **0 of 196 layers** and **0 of 28 blocks**.

## Results

No compressed candidate or quality result exists.

## What we learned

The experiment verifies only that the 3B model could be resolved and prepared under the then-current configuration.
It says nothing about compression throughput, memory sufficiency, resumability after a committed block, or quality.

## Disposition

Inconclusive. Do not treat preparation artifacts as model support evidence.
