# Calibration Behavior

The rewrite preserves the legacy statistic meaning while making accumulation typed and hook lifetimes explicit.

- Input importance is the mean squared linear input after robust row-norm clipping.
- Fisher output importance uses the mean squared output gradient. The legacy numerical stabilization is preserved:
  gradients are multiplied by `1e6` before clipping/squaring and the accumulated statistic by `1e-6` afterward.
- Online Fisher maintains a cumulative maximum clipping threshold. When a later batch raises that threshold, prior totals
  receive the same squared threshold correction as the legacy implementation.
- Two-phase Fisher first discovers fixed robust thresholds, removes those hooks, and then performs a deterministic second
  accumulation pass.
- Forward-only calibration collects input statistics and explicitly emits unit output importance.
- Shrinkage is `(1-s) * value + s * mean(value)` for `0 < s < 1`, matching the legacy implementation.

All hooks are removed in `finally` blocks. Calibration never attaches permanent buffers or replaces modules. Two-phase
calibration is invariant to equivalent equal-sized batch partitions; arbitrary unequal partitions and the online clipping
approximation will be reconciled explicitly before the 1B parity gate.
