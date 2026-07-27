# Experiment 016: Gemma 3 270M stronger architecture protection

## Status

**Completed.**

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/016-compress-and-benchmark-gemma-3-270m-it.py`
- Retained report:
  [`016-compress-and-benchmark-gemma-3-270m-it-quality.md`](../../Results/016/016-compress-and-benchmark-gemma-3-270m-it-quality.md)

## Question

Would stronger down-projection and edge-block weights plus greater sensitivity improve Experiment 015?

## What we did

We raised the `down_proj` weight to 1.5, the edge-block weight to 1.3, and sensitivity to 0.75, then compared both
cohort reconstruction and end quality.

## Results

Protected-cohort reconstruction improved versus Experiment 015: `down_proj` by **3.72%**, Q by **4.96%**, K by
**4.21%**, V by **6.14%**, O by **3.83%**, and edge blocks by **9.44%**. However:

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 194.054 | 1,369.485 |
| Mean task accuracy | 0.5208 | 0.4200 |

Effective BPW was **1.025280**.

## What we learned

Even broad reconstruction improvements across deliberately protected units did not guarantee better global language
modeling. Capacity moved away from gate/up projections, perplexity worsened, and only some tasks improved. Allocation
must be selected using matched end-to-end metrics, not cohort error alone.

## Disposition

Not selected over Experiment 015 for 270M quality, but its tempered form was tested at 1B in Experiment 017.
