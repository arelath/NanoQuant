# Experiment 054 factor-fusion retest

**Date:** 2026-08-03  
**Status:** completed screen and functional rejection; retain QKV as the only supported fusion

## Question

Does the completed Experiment 054 calibration objective change any earlier
factor-sharing decisions, or reveal a useful fusion beyond the adopted QKV row
stack?

The answer is no. QKV remains a large equal-bit reconstruction win. Every
previously rejected fusion retains its rejection, and the new reciprocal MLP
candidates produce matrix-objective false positives that fail held-out MLP
output and teacher KL.

## Protocol

- Model: pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Objective: the 182 already-shrunk diagonal input/output importance profiles
  retained by the completed Experiment 054 run, not the older July Fisher
  surrogate. The source is Experiment 054's `objective-specs` artifact
  `sha256-75cc889932d55840a249816924a45d88ef103f6be5ec5bfce0adfb4b6e2f73eb`.
- Budget: at most 1.0 physical BPW per compared source-weight inventory,
  including binary factors, 16-bit scales, padding, and an extra input-side
  scale profile when a reciprocal group combines distinct objectives.
- Fit: production ADMM, 400 outer by 5 inner iterations, cubic penalty schedule,
  regularization 0.03, wide-matrix transposition, and two scale-fit passes.
- Rank alignment: one, so rank-quantum rounding does not decide the comparison.
- Seed: zero.
- Functional gate: eight retained WikiText sequences of 512 tokens for paired
  teacher KL and four sequences for isolated MLP output error.

`tools/probe_factor_grouping.py` now accepts a resident `objectives.json`
artifact directly and enumerates all five partitions of gate, up, and
transposed down. `tools/probe_mlp_partition_functional.py` materializes selected
dense reconstructions in memory and restores the teacher after every isolated
splice. Neither tool creates a deployable compression artifact.

## Attention partitions: the old result reproduces

All 15 set partitions of Q, K, V, and `O.T` were repeated over all 26 blocks.

| Fixed topology | Experiment 054 weighted RMSE | Change vs QKV/O |
|---|---:|---:|
| QV / KO | 0.402678 | -0.972% |
| QKV / O | 0.406631 | baseline |
| QK / VO | 0.408050 | +0.349% |
| QKVO | 0.414938 | +2.043% |
| four singletons | 0.507064 | +24.699% |

The ordering is effectively unchanged from the earlier corrected-CCE result:
QV/KO moved from -1.035% to -0.972%, and QK/VO moved from +0.356% to +0.349%.
The 15-way per-block matrix oracle is -3.606%, also essentially unchanged from
-3.626%.

QV/KO is not revived by this reproduction. Its completed held-out gate already
raised full-attention teacher KL by 27.29% and isolated attention-output error by
about 19.5%. The Experiment 054 objective did not provide a materially different
matrix signal that could justify repeating that expensive negative gate.

QKV itself remains strongly supported: QKV/O is 19.81% lower weighted RMSE than
four separately factorized attention projections at equal physical bits.

## Adjacent-block sharing: every arm still loses

The same three depth pairs used in the original study were repeated with the
Experiment 054 objective.

| Shared group | blocks 0-1 | blocks 10-11 | blocks 24-25 |
|---|---:|---:|---:|
| QKV | +10.20% | +5.90% | +9.73% |
| gate | +3.25% | +2.63% | +1.41% |
| up | +0.99% | +0.23% | +0.50% |
| `down.T` | +2.33% | +0.17% | +2.06% |

Every value is a weighted-RMSE regression. These closely match the prior
corrected-CCE results. Statistical similarity between adjacent activations does
not translate into a shared binary factor basis worth tying, and cross-block
execution would not provide a normal fused matmul in any case.

## New reciprocal MLP partitions

Gate and up have shape `6912 x 1152`; `down.T` has the same shape. This permits
the MLP analogue of the reciprocal attention experiment. The five partitions
are:

- separate gate / up / `down.T`;
- gate+up / `down.T`;
- gate+`down.T` / up;
- up+`down.T` / gate; and
- gate+up+`down.T`.

The three-block exhaustive screen rejected gate+up again. The other reciprocal
groups showed late-block matrix gains, but no fixed all-block policy won:

| Fixed policy | Global weighted RMSE | Change vs separate | Blocks won |
|---|---:|---:|---:|
| separate | 0.519121 | baseline | — |
| gate+`down.T` / up | 0.523880 | +0.917% | 15/26 |
| up+`down.T` / gate | 0.522480 | +0.647% | 15/26 |
| gate+up+`down.T` | 0.531213 | +2.329% | 13/26 |

The apparently contradictory win counts come from depth and objective energy:
the reciprocal variants lose in early blocks and improve mainly in later
blocks. A per-block matrix selector would therefore look attractive, but the
attention QV/KO result showed that cross-role selection must be gated at the
nonlinear operator.

## Held-out MLP and KL gate

The strongest late-block cases were spliced into the clean BF16 teacher. All
reported matrix changes are favorable; positive output and KL changes are
regressions.

| Candidate | Block | Matrix RMSE | MLP output RMSE | Teacher KL | 95% KL delta interval (nats/token) |
|---|---:|---:|---:|---:|---:|
| gate+`down.T` | 10 | -1.07% | +4.61% | +39.52% | [+0.03634, +0.04946] |
| gate+`down.T` | 15 | -2.92% | +10.25% | +22.44% | [+0.01016, +0.03098] |
| gate+`down.T` | 20 | -4.29% | +30.23% | +17.12% | [+0.00285, +0.01964] |
| gate+`down.T` | 22 | -4.40% | +46.83% | +18.73% | [+0.00504, +0.01634] |
| up+`down.T` | 15 | -3.12% | +25.47% | +10.59% | [+0.00433, +0.01290] |
| up+`down.T` | 20 | -4.29% | +72.70% | +17.00% | [+0.00515, +0.01880] |
| up+`down.T` | 22 | -4.36% | +76.27% | +8.50% | [-0.00246, +0.01160] |
| up+`down.T` | 24 | -2.45% | +26.50% | +8.55% | [+0.00205, +0.00957] |
| three-way | 15 | -3.56% | +21.36% | +25.21% | [+0.01495, +0.02711] |
| three-way | 20 | -5.69% | +86.18% | +26.90% | [+0.00549, +0.03320] |
| three-way | 22 | -5.81% | +100.01% | +32.15% | [+0.00876, +0.02702] |
| three-way | 24 | -2.48% | +28.20% | +22.54% | [+0.01127, +0.02071] |

Eleven of twelve KL intervals are wholly worse; the remaining up/down block-22
interval crosses zero but has a positive point estimate and a 76% MLP-output
regression. Every isolated MLP-output comparison is decisively worse. No
reciprocal MLP topology passes the operator-level gate.

## Decision

- Retain within-block QKV row stacking. It remains the only fusion with both a
  strong equal-bit representation result and a natural fused runtime operation.
- Keep QV/KO and the remaining attention partitions rejected. Experiment 054
  reproduces the old matrix result rather than reversing it.
- Keep all adjacent-block factor tying rejected.
- Reject gate/down, up/down, and three-way reciprocal MLP factor ownership.
  Their matrix gains are caused by trading error across serial nonlinear roles,
  and that trade is destructive in the actual MLP.
- Do not build planner, artifact, tuning, packed, GGUF, or runtime support for
  these new MLP topologies.

The broader result is that shape compatibility and even a repeatable weighted
matrix gain are insufficient fusion criteria. A useful fusion must share an
actual runtime input (as QKV does) and survive the coupled operator metric.

## Evidence

- `evidence/m4/factor-grouping-probe/attention-partitions-experiment054.json`
- `evidence/m4/factor-grouping-probe/adjacent-experiment054.json`
- `evidence/m4/factor-grouping-probe/mlp-partitions-experiment054.json`
- `evidence/m4/factor-grouping-probe/mlp-gd-functional-experiment054.json`
- `evidence/m4/factor-grouping-probe/mlp-ud-gud-functional-experiment054.json`

The evidence JSON files are intentionally ignored analysis outputs. This
document and the reusable probes carry the durable repository record.
