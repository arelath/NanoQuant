# Experiment 058: matched 057 rate-matched INT8 outliers

## Question

Does reinvesting Experiment 057's BF16 outlier value rate into more
calibration-weighted INT8 columns improve the complete product-codebook model?

## Controlled change

Experiment 058 inherits Experiment 057's pinned model, calibration, exact-unit
KL allocation, product-k16 encoding, binary search, layer/block tuning,
post-block refit, top-64 distillation, 1.0-BPW factor budget, and retained
quality protocol. Its configuration delta is limited to:

- outlier storage changes from BF16 to symmetric per-column INT8 with one BF16
  scale per column;
- the global outlier fraction changes from `0.001` to `13 / 6912`, so every
  Gemma down projection receives exactly thirteen residual-selected columns.

The policy remains outside the charged 1.0-BPW factor ceiling, matching 057.
Nevertheless, scale and index tensors are included in reported logical bits.
The fraction applies proportionally to other projection shapes because the
current typed outlier policy is global; this is the smallest configuration-only
057 derivative that preserves outliers on every layer.

## Motivation and gate

The retained 057 follow-up screen found that merely quantizing its seven tuned
down-projection columns worsened KL. Spending the saved rate on six additional
columns selected by raw residual also failed. Selecting those additions with
the calibration-weighted objective improved held-out NLL by 1.11% and teacher
KL by 3.14% across all eight tested sequences. Experiment 058 tests whether that
signal survives selection before factorization, the full resident tuning path,
global distillation, packing, and the unchanged quality benchmark.

Promotion requires a completed and freshly validated run, exact replay of INT8
values/scales through the compact product-codebook artifact, no regression in
effective BPW versus the declared sidecar treatment, and protocol-matched
quality better than Experiment 057. The eight-sequence splice result is not a
promotion substitute.

## Implementation prerequisite

The resident layer retains a floating outlier master for tuning and requantizes
it at every durable freeze. INT8 cost includes one 16-bit scale per column. The
compact product-codebook overlay carries the same optional scale tensor as the
base packed layout. These changes advance the resident algorithm identity so
an older floating-sidecar commit cannot be adopted by Experiment 058.

## Status

Not run. The zero-argument launcher and CPU contract coverage are implemented;
a CUDA launch still requires the normal single-worker, lease, and device checks.
