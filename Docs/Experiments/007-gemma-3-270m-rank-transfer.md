# Experiment 007: Gemma 3 270M rank-policy transfer

## Status

**Completed.**

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/007-compress-and-benchmark-gemma-3-270m-it.py`
- Retained report: [`007-gemma-3-270m-it-quality.md`](../../Results/007/007-gemma-3-270m-it-quality.md)

## Question

Would the projection-specific policy from Experiment 006 transfer to the smaller Gemma 3 270M model?

## What we did

We adapted the same rank priorities to the 18-block 270M architecture and ran full compression, export, and quality
evaluation near the one-bit target.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 193.228 | 1,546.252 |
| Mean task accuracy | 0.5158 | 0.4092 |

Effective BPW was **0.997934**.

## What we learned

The policy transferred operationally but not in quality. At essentially one effective bit per weight, the smaller
model was much more fragile: perplexity rose to about eight times BF16. Model-size transfer requires fresh
measurement and likely different allocation rather than scaled copies of a larger model's recipe.

## Disposition

Retained as the first 270M baseline; superseded by Experiment 010.
