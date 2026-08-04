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

Experiment 055 also enables the retained Experiment 052 control-then-tabu sign search for every **final** factor
owner. This is an intentional second experimental variable requested before the compression run produced any
complete block. Each final owner uses:

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

Rank-response probes are deliberately cheaper approximations: they run ADMM only, with 100 outer iterations, five
inner iterations, the unchanged `3e-2` regularization, and the normalized cubic penalty schedule. They do not run
scale/control/one-bit/variable-depth/tabu refinement. ADMM has no learning-rate parameter, and the normalized schedule
still reaches the same terminal penalty, so no learning-rate adjustment is applicable. This makes the allocation
curve less representative of final post-search capacity but avoids repeating the expensive production search at all
three response ranks for all 130 units. Production tuning and binary learning rates are unchanged.

Experiment 052 showed why the final quality gate is binding: tabu improved the diagonal residual objective in all
nine sampled real owners, but broad composition worsened held-out KL; only block-25 QKV survived a disjoint functional
confirmation. Experiment 055 therefore tests broad tabu deliberately, but a better reconstruction curve is not a
promotion result.

The retained profile is
`evidence/054/054-d2-uniform-control-kl-profile`, key
`sha256:4a67e45d5266763b09e3b487a3820f4ad8520201b144807241e2744d9c271bf9`. The uniform control that produced this
profile is numerically unchanged, so Experiment 055 reuses it and measures new response points above the old cap.
The campaign permits dataset-hub access only so 055 can create its required run-owned calibration receipt; model
resolution remains pinned to the local snapshot. Later slices reuse the validated receipt from this run. The normal
zero-argument Experiment 055 launcher is the campaign controller: it starts a fresh child for each one-block slice,
recognizes the expected injected interruption only after the durable block count advances, and continues through the
final unbounded completion/export slice. The worker-only environment marker prevents recursive controllers. Ordinary
worker failures still stop the controller and surface the retained child stderr.

The resident algorithm version advances from 54 to 58. Version 55 introduced the over-complete ceiling; version 56
added the post-tuning binary-search path and persisted search metrics; version 57 makes reconstruction rank probes
ADMM-only; version 58 records the exact zero entry loss directly when the initial teacher and compressed activation
streams are the same tensor. The packed layout and runtime are rank-agnostic; no format change is required.

## Promotion gate

Status: **completed; rejected**.

The version-58 campaign completed all 26 blocks, global tuning, global distillation, logical and packed export, GGUF
publication, and the retained quality benchmark. Fresh validation audited 768 transitive artifacts and all 156 active
journal records without error.

The cap was active. The final allocation uses ranks through 1,440 and increases total rank from 111,456 in Experiment
054 to 112,192. Many early MLP owners, including `down_proj`, receive rank 1,440 rather than the old rank-1,152
physical-dimension ceiling. The result therefore answers the intended capacity question rather than merely repeating
the capped allocation.

| Measurement | Experiment 054 | Experiment 055 | Change |
| --- | ---: | ---: | ---: |
| Effective BPW | 1.024405 | 1.024416 | +0.000011 |
| Rank sum | 111,456 | 112,192 | +736 (+0.66%) |
| Block wall time | 8,331 s | 15,647 s | +87.8% |
| Peak GPU bytes | 7,650,410,496 | 7,696,547,840 | +0.60% |
| Peak host bytes | 9,056,464,896 | 7,806,263,296 | -13.8% |
| Resident artifact bytes | 9,770,129,117 | 10,628,673,408 | +8.79% |
| Packed quantized payload | 89,472,832 | 89,473,752 | +920 bytes |
| GGUF bytes | 417,332,736 | 417,333,696 | +960 bytes |
| WikiText-2 perplexity | 203.841821 | 285.496021 | **+40.06%** |

Task quality also fails to establish a compensating benefit: PIQA, ARC-Challenge, HellaSwag, Winogrande, and BoolQ
all decline; only ARC-Easy rises slightly from 0.377 to 0.386. The exact target-bit allocator nearly preserves BPW,
but the final representation is 7,680 bits above 054 because its changed rank distribution carries slightly different
scale overhead. This is immaterial in size and still violates the strict no-BPW-increase gate.

## Decision

Reject bounded over-complete D2 allocation and do not use Experiment 055 as the next baseline. Extra binary rank is
not equivalent to useful model capacity under the current response objective: it nearly doubles block processing
time, increases retained artifact storage, and substantially worsens held-out perplexity despite a measured
equal-budget redistribution.

Retain the run, export, profiles, and archived failed identities as evidence. Experiment 054 remains the closer
quality control for subsequent analysis.

Evidence:

- `evidence/055/055-overcomplete-rank-d2-compress-and-benchmark-gemma-3-1b-it`
- `Results/055/055-overcomplete-rank-d2-compress-and-benchmark-gemma-3-1b-it-quality.md`
- `Results/055/gemma-3-1b-it-nanoquant.gguf`
