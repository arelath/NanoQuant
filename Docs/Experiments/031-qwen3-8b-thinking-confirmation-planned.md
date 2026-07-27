# Experiment 031: Planned Qwen3 8B thinking confirmation

## Status

**Not run.**

- Model: `Qwen/Qwen3-8B`
- Launcher: `experiments/031-confirm-qwen3-8b-thinking-quality.py`
- Retained results: none
- Design: [Qwen3 thinking-mode quality](../36-qwen3-thinking-mode-quality.md)

## Question

Would the teacher-generated dual-mode calibration and distillation approach scale from Qwen3 0.6B to Qwen3 8B?

## Intended method

The launcher reused coherent teacher outputs for thinking and non-thinking slices, retained raw calibration coverage,
and selected serialized deployment-quality evaluation suitable for the larger model.

## Results

The experiment was not run. In particular, it does not confirm that the data recipe scales, that 8B fits the
available execution envelope, or that either behavior mode meets its teacher-relative gate.

## What we learned

No empirical scaling lesson was obtained. In light of Experiment 030, a future run should first strengthen its
absolute per-mode gates and held-out-data controls rather than simply repeat the same relative cross-mode criterion
at larger scale.

## Disposition

Planned confirmation only.
