# Experiment 071: two full-rank independent stages

Prepared 2026-11-11. **Not run.** Only input/configuration validation and CPU planning checks have been performed.

## Definition

Repeat the pinned `google/gemma-3-1b-it` Experiment 070 workload with these requested changes:

- ADMM: 600 outer iterations and two inner power iterations, instead of 800 and five.
- Block tuning: `compile_forward=True`, using the existing PyTorch compilation option.
- Exactly two independently scaled full-rank stages for every quantization owner, including the shared QKV owner.
- Reuse `Datasets/gemma-3-1b-it-allocation-070` and validate its teacher provenance against
  `Datasets/gemma-3-1b-it-kl-48x512`.

The parent is frozen in `experiments/recipes/references/experiment070.json` from the completed 070 manifest,
with semantic hash `sha256:97f03cb6e399438814dee7bdfeaec3108ec6bce519da842ada9099ca9ad4c509`.
It matches the 070 launcher except for that run's resolved KL-profile path and key. This prevents future base-recipe
changes from silently changing the comparison.

The five numerical configuration differences from the parent are ADMM outer/inner iterations,
`block_tuning.compile_forward`, `allocation.maximum_rank_layer_patterns=("*",)`, and
`allocation.bounds.overcomplete_rank_ceiling_fraction=2.0`. Dataset bindings and experiment identity/output paths
are execution/provenance differences. All other settings remain inherited, including pinned revision,
calibration, scale fitting, QKV grouping/multipliers, outlier policy, tuning epochs, batches, learning rates,
global distillation, export formats, quality backend and the 1,000-task evaluation limit.

## Ranks and represented bits

Maximum-rank promotion runs after ordinary allocation. The explicit total-rank ceiling is twice the physical
matrix rank, and the residual-stage fitter splits it into exactly two physical-rank terms. The nominal 2-BPW
allocation target remains the inherited planning input; promotion is intentionally allowed to exceed it.
No second allocation pass shrinks the promoted ranks to restore 2 BPW.

| Owner | Matrix shape | Primary rank | Residual rank |
| --- | --- | ---: | ---: |
| MLP gate / up | 6,912 × 1,152 | 1,152 | 1,152 |
| MLP down | 1,152 × 6,912 | 1,152 | 1,152 |
| Stacked attention QKV | 1,536 × 1,152 | 1,152 | 1,152 |
| Attention output | 1,152 × 1,024 | 1,024 | 1,024 |

All 26 blocks use this policy: 130 owners and 260 stages. Represented factor/sign/scale cost is
**1,754,947,584 bits / 697,761,792 quantized parameters = 2.51510989 BPW**. This excludes embeddings/output heads,
unquantized tensors, GGUF metadata and storage padding. It is a planned representation cost, not a measured file size.
The retained retry policy cannot increase ranks beyond this cap; residual-column increments remain disabled.
Unlike 070's variable stage count, this policy always uses two stages, including where 070 could fund a third stage.

## Dataset behavior

The allocation dataset supplies saved approximate rank-response curves and measured KL sensitivities. All ranks
are ultimately promoted to the fixed maximum, so the curves cannot trade capacity between final owners.
The dataset still bypasses uniform-control compression, fresh KL profiling and rank-response fitting.

The teacher dataset is loaded and validated against the packaged profile's model/protocol, input hashes and
baseline NLL. That profile already contains measurements from the same teacher inputs, so no redundant teacher
inference is scheduled. The 48×512 teacher dataset does **not** replace the separate global-distillation teacher
protocol; global distillation is unchanged from 070.

## Commands

Validate only, without starting the campaign:

```powershell
.\.venv\Scripts\python.exe tools\run_experiment071_campaign.py --dry-run
```

Start later, when desired:

```powershell
.\.venv\Scripts\python.exe tools\run_experiment071_campaign.py
```

The campaign runner keeps 070's one-block process slices, durable resume behavior and export/benchmark lifecycle.
The launcher selects both workspace datasets by default; `--allocation-profile-dataset` and `--kl-teacher-dataset`
can override their paths. No experiment directories or campaign logs are created by the dry-run wrapper.

The direct launcher is
`experiments/071-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it.py`.
One direct invocation retains the intentional interruption after one new block; the campaign runner resumes it.
Future artifacts belong under `evidence/071`, `outputs/071`, and `Results/071`.

## Acceptance

Setup checks passed the real two-dataset dry run. A separate CPU-only replay loaded 070's saved calibration
statistics/objectives, reconstructed its inventory, ran the dataset response/anchor path, and called the production
planner with 071's policy. All 130 resulting owners had the two expected full-rank stages and the cost above.
This was a planning diagnostic under `outputs/071-planning-check`, not a resident compression run.
The dry-run transcript is `outputs/071-setup-dry-run.log`; the per-owner shape/cost check is
`outputs/071-rank-shape-check.json`.
Focused setup tests passed, followed by the full suite: **1,598 passed, 53 deselected**.
Repository-wide Ruff and mypy's 224-file source check also passed. No Experiment 071 compression worker was launched.

Setup validation does not establish compression quality, compiler speedup, memory residency or full-run duration.
After a future run, compare 071 against 070 using the same quality protocol and report the larger BPW alongside
quality and elapsed time. More rank, shorter fitting and compilation all change this candidate, so it is not an
isolated speed comparison or an equal-bit quality comparison.
