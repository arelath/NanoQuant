# Experiment 005: Gemma 3 4B maximum v-projection rank

## Status

**Completed.**

- Model: `google/gemma-3-4b-it`
- Launcher: `experiments/005-gemma-3-4b-it-vproj-double-request.py`
- Retained report:
  [`005-gemma-3-4b-it-vproj-maxrank-quality.md`](../../Results/005/005-gemma-3-4b-it-vproj-maxrank-quality.md)

## Question

Was Experiment 004 simply too small a rank increase, and would the maximum physical `v_proj` rank produce a clear
quality gain?

## What we did

We requested double value-projection rank. The matrix dimensions capped the realized rank at 1,024, yielding about
1.416 times the original `v_proj` bits rather than a full doubling.

## Results

- Packed size increased **1.1762%**.
- Weighted target reconstruction error fell about **35%**.
- WikiText-2 perplexity changed from **84.107** to **85.883**.
- Mean task accuracy was **0.5083**, with mixed task-level movement.

## What we learned

Even the largest feasible value-rank overlay improved reconstruction while worsening perplexity. As in Experiment
004, the post-tuning overlay meant this was not a clean rejection of value rank inside a fully retuned pipeline. It
did reject reconstruction error as a stand-alone reason to publish the overlay.

## Disposition

Not adopted directly. The maximum-rank setting was carried into Experiment 006 only as a hypothesis to test with
full compression and tuning.
