# Experiment 034 — Gemma 3 1B post-refit QKV covariance

**Status:** Ready to run

## Question

Does the same-rank post-refit covariance placement selected in Document 50
survive the complete Experiment 022 compression, global-distillation, export,
and retained-quality lifecycle?

## Method

Experiment 034 changes Experiment 022 only by enabling post-refit covariance
refinement for the fused-QKV owner in blocks 5, 11, 24, and 25. It retains
the diagonal calibration objective, D2 rank allocation, ranks, outliers,
factor format, global KD, export, and quality protocols.

The refinement captures 8,192 input rows for each selected owner and runs
after factorized tuning and post-block refit. No new representation field or
bit allocation is introduced.

## Gate

The experiment must complete strict validation and the mandatory packed,
checkpoint, GGUF, and retained-quality lifecycle. Promotion requires:

- no effective-BPW increase versus Experiment 022;
- exact pre-KD quality consistent with the bounded −6.79% perplexity result;
- final post-KD quality better than Experiment 022;
- valid resume, artifacts, and exported runtime format.

No conclusion is supported until the complete run finishes.
