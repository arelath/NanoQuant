# Experiment 030: Qwen3 0.6B thinking-mode recovery

## Status

**Completed after rollover.** An earlier math-specialized dataset attempt was archived; the retained result uses
teacher-generated UltraChat responses for both modes.

- Model: `Qwen/Qwen3-0.6B`
- Launcher: `experiments/030-recover-qwen3-0-6b-thinking-quality.py`
- Retained report:
  [`030-recover-qwen3-0-6b-thinking-quality-quality.md`](../../Results/030/030-recover-qwen3-0-6b-thinking-quality-quality.md)
- Design: [Qwen3 thinking-mode quality](../36-qwen3-thinking-mode-quality.md)

## Question

Could coherent, full teacher responses in both thinking and non-thinking modes recover Qwen3 behavior after
compression?

## What we did

We replaced the math-heavy OpenR1 approach with teacher-generated UltraChat 200K responses. Each sample retained the
teacher's entire answer, including its thinking trace when thinking was enabled. The final mixture included both
teacher modes plus raw calibration data, with 528 prepared samples and eight distillation epochs. We then ran generic
quality, GGUF deployment, and mode-specific response-NLL checks.

## Results

| Metric | BF16 | Candidate |
| --- | ---: | ---: |
| WikiText-2 perplexity | 55.167 | 5.569 |
| Mean task accuracy | 0.5608 | 0.4233 |
| Thinking response NLL | 0.3485 | 1.7104 |
| Non-thinking response NLL | 0.3746 | 1.7172 |

- Thinking-mode NLL ratio: **4.9078 times BF16**
- Non-thinking-mode NLL ratio: **4.5844 times BF16**
- Effective BPW: **1.028092**
- GGUF size: **393,476,544 bytes**
- Storage reduction from BF16 tensor bytes: **73.83%**

The cross-mode guard passed because the two degraded modes were similarly bad. The extremely low candidate
WikiText-2 perplexity, combined with worse tasks and behavior NLL, is a warning for overlap, memorization, or protocol
contamination rather than evidence that the candidate surpassed its teacher.

## What we learned

Coherent teacher data was necessary but not sufficient to recover source-model behavior. A relative guard comparing
thinking with non-thinking can pass when both modes regress together. Each mode needs an absolute teacher-relative
gate on held-out prompts, and generation/evaluation datasets must be demonstrably disjoint. Generic perplexity can be
misleading under distillation and cannot replace behavior-specific validation.

## Disposition

The data-generation mechanism is reusable; the resulting model did not meet behavior recovery. Future confirmation
must use absolute per-mode gates and clean held-out evaluation.
