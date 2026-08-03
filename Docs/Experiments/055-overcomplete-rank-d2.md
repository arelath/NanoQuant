# Experiment 055: bounded over-complete D2 rank

## Question

Does Experiment 054's algebraic-dimension rank ceiling prevent the equal-budget D2 allocator from buying useful
capacity, especially for `down_proj` owners that repeatedly saturate at rank 1152?

Experiment 054 allocated 32.77% of all quantized bits to `down_proj`, and 20 of 26 such owners reached the previous
hard ceiling. That is evidence of a binding constraint, not by itself evidence that all `down_proj` layers deserve
more bits. Earlier blanket promotions worsened final perplexity, so this experiment lets the measured response curves
and exact-unit KL objective decide which owners receive the newly available ranks.

## Controlled changes

Experiment 055 replays Experiment 054 with the same model, retained exact-unit KL profile, target BPW, 2x V member
weight, functional binary learning rate, tuning, and distillation policy. It changes the rank search space:

```text
allocation.bounds.ceiling_fraction_of_uniform: 1.4 -> 1.5
allocation.bounds.overcomplete_rank_ceiling_fraction: 1.0 -> 1.5
```

The hard ceiling is now the aligned value at 1.5 times the smaller matrix dimension. The allocator remains constrained
to the unchanged 1.0 target BPW, so selecting an over-complete owner requires an equal-budget trade against other
owners. For Gemma's MLP projections the measured ceiling is rank 1440, above the old rank-1152 limit; the independent
hard ceiling is rank 1728.

Experiment 055 also enables the retained Experiment 052 control-then-tabu sign search for every factor owner. This is
an intentional second experimental variable requested before the compression run produced any complete block. Each
rank-response point and each final owner uses:

```text
scale fits:                 64 passes
control:                    8 outer passes
one-bit control:            16 full-vector passes
variable-depth control:     2 passes x length 64
tabu continuation:          8 outer passes
tabu per vector sweep:      2 passes x 256 steps
tabu tenure:                8 plus deterministic jitter 0-4
```

For final owners the search runs after factorized tuning and before freezing. The unchanged tuned incumbent, control,
and tabu are rescored after casting scales to their persisted dtype. The lowest-error state is selected, with strict
ties retaining the earlier state, so neither control nor tabu can replace the incumbent unless its stored
representation improves the exact diagonal residual objective. Post-block scale refit then proceeds normally.
Rank-response probes use the same search policy so the D2 allocator measures the capacity it will actually receive.

Experiment 052 showed why the final quality gate is binding: tabu improved the diagonal residual objective in all
nine sampled real owners, but broad composition worsened held-out KL; only block-25 QKV survived a disjoint functional
confirmation. Experiment 055 therefore tests broad tabu deliberately, but a better reconstruction curve is not a
promotion result.

The retained profile is
`evidence/054/054-d2-uniform-control-kl-profile`, key
`sha256:4a67e45d5266763b09e3b487a3820f4ad8520201b144807241e2744d9c271bf9`. The uniform control that produced this
profile is numerically unchanged, so Experiment 055 reuses it and measures new response points above the old cap.
The campaign permits dataset-hub access only so 055 can create its required run-owned calibration receipt; model
resolution remains pinned to the local snapshot. Later slices reuse the validated receipt from this run.

The resident algorithm version advances from 54 to 56. Version 55 introduced the over-complete ceiling; version 56
adds the post-tuning binary-search path and persisted search metrics. The packed layout and runtime are rank-agnostic;
no format change is required.

## Promotion gate

Status: **Not run under the tabu policy**. A pre-tabu attempt completed only part of the rank-response probe and no
block commits. Its probe-plan identity is incompatible with version 56 and will not be reused; the evidence is
preserved at
`evidence/055/aborted-pre-tabu-055-overcomplete-rank-d2-compress-and-benchmark-gemma-3-1b-it` rather than rewritten.

Promotion requires:

1. New response probes actually sample ranks above the old algebraic-dimension ceiling.
2. The planned allocation remains within the same exact target-bit budget as Experiment 054.
3. At least one saturated owner earns an over-complete rank from measured marginal utility; otherwise the cap was not
   the active allocation problem.
4. Search telemetry reports control/tabu errors, stored-dtype selection, updates, sign distance, and wall time for every
   selected owner; a nonzero tabu selection rate is required to claim that tabu was exercised.
5. The complete resident run and export pass fresh artifact validation.
6. The retained WikiText-2 protocol improves on Experiment 054 without a BPW increase, with rank, per-owner loss,
   time, memory, and artifact-byte comparisons reported.

A lower probe error or a partial block run is diagnostic evidence only and cannot complete the experiment.
