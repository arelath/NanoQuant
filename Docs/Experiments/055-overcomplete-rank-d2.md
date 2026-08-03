# Experiment 055: bounded over-complete D2 rank

## Question

Does Experiment 054's algebraic-dimension rank ceiling prevent the equal-budget D2 allocator from buying useful
capacity, especially for `down_proj` owners that repeatedly saturate at rank 1152?

Experiment 054 allocated 32.77% of all quantized bits to `down_proj`, and 20 of 26 such owners reached the previous
hard ceiling. That is evidence of a binding constraint, not by itself evidence that all `down_proj` layers deserve
more bits. Earlier blanket promotions worsened final perplexity, so this experiment lets the measured response curves
and exact-unit KL objective decide which owners receive the newly available ranks.

## Controlled change

Experiment 055 replays Experiment 054 with the same model, retained exact-unit KL profile, target BPW, 2x V member
weight, functional binary learning rate, tuning, and distillation policy. It changes only the rank search space:

```text
allocation.bounds.ceiling_fraction_of_uniform: 1.4 -> 1.5
allocation.bounds.overcomplete_rank_ceiling_fraction: 1.0 -> 1.5
```

The hard ceiling is now the aligned value at 1.5 times the smaller matrix dimension. The allocator remains constrained
to the unchanged 1.0 target BPW, so selecting an over-complete owner requires an equal-budget trade against other
owners. For Gemma's MLP projections the measured ceiling is rank 1440, above the old rank-1152 limit; the independent
hard ceiling is rank 1728.

The retained profile is
`evidence/054/054-d2-uniform-control-kl-profile`, key
`sha256:4a67e45d5266763b09e3b487a3820f4ad8520201b144807241e2744d9c271bf9`. The uniform control that produced this
profile is numerically unchanged, so Experiment 055 reuses it and measures new response points above the old cap.
The campaign permits dataset-hub access only so 055 can create its required run-owned calibration receipt; model
resolution remains pinned to the local snapshot. Later slices reuse the validated receipt from this run.

The resident algorithm version advances from 54 to 55 because the new ceiling changes factor shapes and durable
commits. The packed layout and runtime are rank-agnostic; no format change is required.

## Promotion gate

Status: **Not run**.

Promotion requires:

1. New response probes actually sample ranks above the old algebraic-dimension ceiling.
2. The planned allocation remains within the same exact target-bit budget as Experiment 054.
3. At least one saturated owner earns an over-complete rank from measured marginal utility; otherwise the cap was not
   the active allocation problem.
4. The complete resident run and export pass fresh artifact validation.
5. The retained WikiText-2 protocol improves on Experiment 054 without a BPW increase, with rank, per-owner loss,
   time, memory, and artifact-byte comparisons reported.

A lower probe error or a partial block run is diagnostic evidence only and cannot complete the experiment.
