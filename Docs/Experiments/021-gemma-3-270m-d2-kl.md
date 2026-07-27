# Experiment 021: Gemma 3 270M exact-unit D2/KL allocation

## Status

**Completed.**

- Model: `google/gemma-3-270m-it`
- Launcher: `experiments/021-d2-kl-compress-and-benchmark-gemma-3-270m-it.py`
- Retained report:
  [`021-d2-kl-compress-and-benchmark-gemma-3-270m-it-quality.md`](../../Results/021/021-d2-kl-compress-and-benchmark-gemma-3-270m-it-quality.md)

## Question

Would the corrected, exact-unit D2 signal from Experiment 020 improve a fully compressed 270M model after global
tuning?

## What we did

We generated self-measured exact-unit D2 profiles, allocated the budget from those profiles under trust constraints,
and ran full compression and global tuning.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 194.054 | 1,141.160 |
| Mean task accuracy | 0.5208 | 0.3983 |

Effective BPW was **1.024928**. Versus Experiment 016, perplexity improved from **1,369.485** at slightly lower BPW,
while mean task accuracy fell from **0.4200**.

## What we learned

Corrected D2 survived promotion from a splice probe to a full globally tuned run and improved language modeling at a
matched budget. Its benefit was not uniform across downstream tasks, reinforcing that rank allocation must be judged
on more than one aggregate.

## Disposition

Accepted as positive D2 evidence and transferred to the stronger 1B baseline in Experiment 022.
