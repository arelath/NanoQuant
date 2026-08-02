# Experiment 054: functional binary-factor learning rate

## Question

Can factorized block tuning move the binary signs far enough to improve the
functional objective without paying for additional rank or stored components?

The configuration schema already exposed separate binary, scale, outlier, and
bias learning rates. The resident rewrite nevertheless rejected unequal values
and reduced every factorized parameter to the scale learning rate. Experiment
054 makes those groups real and tests a higher binary rate while holding the
physical representation and every non-binary optimizer rate fixed.

## Implementation

- `tune_factorized` assigns binary latents, scales/patches, outliers, and bias to
  independent AdamW parameter groups.
- Cosine-scheduler resume restores every group's own learning-rate trajectory.
- The resident request, semantic identity, validation, shared-QKV path, ordinary
  layer path, and configuration adapter preserve all four effective rates.
- Equal rates still collapse into one parameter group, preserving the previous
  execution path.
- The resident algorithm version advances from 53 to 54 because changing a
  binary rate changes durable signs.

## Small functional gate

The first gate replayed the retained Experiment 024 block-0 `mlp.gate_proj`
rank-1152 factors. It fitted calibration rows 0-63 and measured rows 64-127.
All arms used 8 epochs and retained `1e-5` for scales, outliers, and bias.

| Binary LR | Sign changes | Held-out block loss | Weighted normalized matrix error |
|---:|---:|---:|---:|
| `1e-5` | 4,788 | 0.450137 | 0.174256 |
| `3e-5` | 15,774 | 0.394186 | 0.174459 |
| `1e-4` | 53,225 | 0.297845 | 0.176367 |
| `3e-4` | 133,507 | 0.226186 | 0.184203 |

This result is intentionally selected by the functional block objective. The
matrix metric gets worse as useful signs move, which confirms that diagonal
weight error is not a safe promotion metric for this refinement.

Evidence:
`evidence/054/block0-gate-binary-lr-fit64-held64.json`.

## Disjoint confirmation and splice gate

The confirmation fitted rows 128-191, measured held-out block loss on rows
192-255, and evaluated a dense single-gate splice through the intact Gemma
language model on 16 held-out 512-token sequences beginning at row 192.

| Metric | Current `1e-5` | Candidate `1e-4` | Candidate change |
|---|---:|---:|---:|
| Held-out block loss | 0.462244 | 0.302788 | -34.50% |
| Teacher KL, nats/token | 0.116941 | 0.105190 | -10.05% |
| Next-token NLL | 3.264908 | 3.259993 | -0.004915 |
| Sign changes | 4,737 | 52,854 | +48,117 |

The paired 95% bootstrap interval for the KL delta is
`[-0.018644, -0.004004]`, wholly below zero. The more aggressive `3e-4` arm
reduced block loss further, but its 16-sequence KL interval crossed zero and its
NLL regressed. It is rejected; `1e-4` is the promoted rate.

Evidence:
`evidence/054/block0-gate-binary-lr-confirm-1e4-kl16.json` and
`evidence/054/block0-gate-binary-lr-confirm-kl16.json`.

## Complete-run protocol

The numbered compression run is an Experiment 024 replay with exactly one
intended algorithmic change:

```text
block_tuning.factorized.learning_rates.binary: 1e-5 -> 1e-4
```

Scale, outlier, and bias rates remain `1e-5`; ranks, BPW accounting, outlier
policy, shared QKV with the 2x V objective, allocation, post-block refit, and
global distillation remain those of Experiment 024.

Promotion requires all of the following:

1. A complete, hash-valid resident run and export lifecycle.
2. No BPW or rank-budget increase attributable to the optimizer change.
3. Per-layer and per-block evidence showing whether the block-0 gate result
   generalizes across MLP, attention, and shared-QKV owners.
4. The standard retained WikiText-2 quality benchmark and task benchmark.
5. Comparison against Experiment 024 on quality, BPW, time, memory, and artifact
   bytes. A tiny or partial run cannot complete this experiment.

