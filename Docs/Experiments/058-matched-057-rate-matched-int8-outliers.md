# Experiment 058: matched 057 rate-matched INT8 outliers

## Question

Does reinvesting Experiment 057's BF16 outlier value rate into more
calibration-weighted INT8 columns improve the complete product-codebook model?

## Controlled changes

Experiment 058 inherits Experiment 057's pinned model, calibration, exact-unit
KL allocation, product-k16 encoding, binary search, layer/block tuning,
post-block refit, top-64 distillation, 1.0-BPW factor budget, and retained
quality protocol. Its configuration deltas are limited to:

- outlier storage changes from BF16 to symmetric per-column INT8 with one BF16
  scale per column;
- the global outlier fraction changes from `0.001` to `13 / 6912`, so every
  Gemma down projection receives exactly thirteen residual-selected columns.
- the product-codebook option screen changes from a single 100-step pass to a
  resumable 100/400/1,200-step multi-fidelity screen. The 100-step pass rejects
  dominated options, the surviving per-layer frontier is measured at 400
  steps, and at most the lowest-rate and lowest-error endpoints per layer are
  measured at 1,200 steps. Only the 1,200-step receipts enter the exact-bit
  global allocator.
- rank retry keeps its existing thresholds, attempt limit, rank growth, and
  extra-bit budget, but gains a one-column INT8 outlier fallback. If an attempt
  still misses its reconstruction threshold at the physical rank cap, the next
  attempt holds rank fixed and selects one additional residual outlier column.
  The fallback is not taken while rank can still grow and cannot exceed the
  layer input width or remaining retry-bit budget.

The policy remains outside the charged 1.0-BPW factor ceiling, matching 057.
Nevertheless, scale and index tensors are included in reported logical bits.
The fraction applies proportionally to other projection shapes because the
current typed outlier policy is global; this is the smallest configuration-only
057 derivative that preserves outliers on every layer. Production
factorization remains at 800 ADMM iterations. The binary sign search remains
the existing 8+8 search; Experiment 058 does not add the rejected 24+24 search
or a 1,600-step fitting arm.

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
base packed layout. Option receipts include their actual ADMM duration in both
their immutable payload and resumable journal identity, preventing a coarse
receipt from satisfying a final-stage allocation. These changes advance the
resident algorithm identity so older floating-sidecar or single-fidelity
commits cannot be adopted by Experiment 058. Retry summaries retain the
accepted attempt's actual outlier count, so packing and logical bit reporting
cannot fall back to the original thirteen-column plan after a capped retry.

## Status

Not run. The zero-argument launcher and CPU contract coverage are implemented;
a CUDA launch still requires the normal single-worker, lease, and device checks.
