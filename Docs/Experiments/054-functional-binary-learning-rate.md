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
NLL regressed. It is rejected.

Evidence:
`evidence/054/block0-gate-binary-lr-confirm-1e4-kl16.json` and
`evidence/054/block0-gate-binary-lr-confirm-kl16.json`.

## Production-horizon canary

The first complete-run attempt correctly failed rather than silently publishing
an unstable model. The small replay fitted 64 rows (8 optimizer steps per epoch),
whereas the production recipe fits 256 rows (32 steps per epoch). Applying
`1e-4` unchanged therefore made a substantially larger optimization excursion:

| Run | Gate rank | Left-sign changes | Fraction changed |
|---|---:|---:|---:|
| Experiment 024 uniform control, `1e-5` | 960 | 14,421 | 0.217% |
| Experiment 054 failed control, `1e-4` | 960 | 153,181 | 2.309% |

The candidate's block-0 factorized gate loss improved from Experiment 024's
0.501360 to 0.198145, but its propagated activations caused non-finite block-1
targets. The resident run stopped with a `FloatingPointError` after preserving a
hash-valid block-0 commit. This rejects `1e-4` at the production horizon and also
shows why a finite end-to-end canary is mandatory after a small functional probe.

The retry uses the already-screened `3e-5` arm. It improved the original 64/64
held-out block loss from 0.450137 to 0.394186, while limiting the per-step rate to
30% of the failed setting.

The production-horizon retry passed the boundary that rejected `1e-4`:

| Metric | Experiment 024 control | Experiment 054 `3e-5` control |
|---|---:|---:|
| Block-0 gate left-sign changes | 14,421 (0.217%) | 54,749 (0.825%) |
| Block-0 gate factorized loss | 0.501360 | 0.307358 |
| Block-0 final normalized loss | 0.008605 | 0.006247 |
| Block-1 entry | finite | finite |

The 54,749 production sign changes closely reproduce the roughly 53k excursion
selected by the 64-row probe instead of the failed run's 153k excursion. Block 0
improves its normalized loss by 27.4%, commits all MLP, shared-QKV, and O owners,
and propagates finite activations into block 1. This promotes `3e-5` past the
multi-block canary; complete-model quality remains the final gate.

The uninterrupted process later encountered a non-finite block-2 teacher
objective immediately after block 1. A fresh audit validated all 48 reachable
artifacts across the first two blocks (2.94 GB), and the committed block-1
activation pair contained no non-finite values. Reloading that exact activation
artifact in a clean process produced a finite block-2 target power of 284.977 and
continued tuning normally. The failure is therefore process-local carry-over,
not corruption of the durable boundary or evidence that `3e-5` itself is
non-finite at block 2.

As a diagnostic containment, Experiment 054 executed in clean, one-block
process slices. Every slice rehydrated the validated frozen prefix and canonical
activation generation, committed one additional block, and exited
intentionally. This isolated transient in-memory state while preserving the same
numerical recipe, rank budget, and committed model state.

The first rollback repair proved that best-state restoration itself was exact:
every selected parameter was byte-identical before refit and after rollback.
It also cleared gradients and AdamW scratch before releasing the failed refit.
A same-process canary crossed the block-13 rollback, entered block 14 with
finite target power 174,782.797, and committed both blocks. A longer replay
nevertheless failed after a later rollback, so end-of-epoch cleanup was
necessary but not sufficient.

The remaining bug was the rollback boundary. The tuner did not inspect
non-finite state until full epoch evaluation, after it had already run later
backward and optimizer steps from a poisoned step. Rollback is now
transactional at each logical optimizer step: it rejects a non-finite scalar
loss before backward, rejects non-finite accumulated gradients before AdamW,
and checks both selected parameters and AdamW moments, squared moments,
denominators, and Kahan compensation immediately after the update. The first
bad step is sanitized, training stops, and the finite entry/best parameter
snapshot is restored. A regression separately poisons only optimizer state,
while leaving parameters finite, to cover the hole that parameter-only checks
missed.

The legacy implementation was also audited before considering a loss-rescaling
workaround. Legacy intentionally backpropagates the summed weighted error
divided by calibration rows, so the rewrite retains that normalization rather
than hiding the rollback defect with a different objective scale.

The production replay in
`evidence/054/rollback-root-cause-diagnostic.stdout.log` then crossed the
non-finite post-refit rollbacks in blocks 18, 19, 20, and 21 in one process and
entered block 22 with finite teacher power. That was useful evidence for the
first-bad-step repair, but it did not establish that every rollback leaves a
healthy process.

A later candidate run reproduced the original failure at a new boundary:
block 1 reported `post_block_refit.nonfinite_rollback`, committed a hash-valid
activation generation, and block 2's dense teacher forward produced a
non-finite target immediately afterward. A fresh validation followed all 180
reachable artifacts (3.06 GB), and chunked audits found zero non-finite entries
in both 256 x 2048 x 1152 activation tensors. The durable boundary remains
sound; transactional rollback reduces the failure surface but does not fully
remove process-local contamination.

Experiment 054 therefore uses clean one-block process slices again. This is a
runtime containment, not an algorithmic change: every slice resumes from the
validated canonical activation generation and frozen prefix with identical
ranks, bit accounting, and numerical recipe. Campaign restarts include a short
cooldown so the Windows CUDA device lease is released before the next slice.

Failed-run evidence:
`evidence/054/054-d2-uniform-control-gemma-3-1b-it--archive-20154357dd00`,
inactive resident identity
`sha256:20154357dd0019eac6ec20c5c6cf67a9eeb26b3d434b51beed2faea2497b3b40`.

## Complete-run protocol

The numbered compression run is an Experiment 024 replay with exactly one
intended algorithmic change:

```text
block_tuning.factorized.learning_rates.binary: 1e-5 -> 3e-5
```

Scale, outlier, and bias rates remain `1e-5`; BPW accounting, outlier policy,
shared QKV with the 2x V objective, allocation recipe, post-block refit, and
global distillation remain those of Experiment 024. The same-run measured
profile may redistribute exact ranks as a downstream effect; target BPW and all
rank-budget policies remain fixed and the observed rank inventory will be
reported rather than assumed equal.

Promotion requires all of the following:

1. A complete, hash-valid resident run and export lifecycle.
2. No BPW or rank-budget increase attributable to the optimizer change.
3. Per-layer and per-block evidence showing whether the block-0 gate result
   generalizes across MLP, attention, and shared-QKV owners.
4. The standard retained WikiText-2 quality benchmark and task benchmark.
5. Comparison against Experiment 024 on quality, BPW, time, memory, and artifact
   bytes. A tiny or partial run cannot complete this experiment.
