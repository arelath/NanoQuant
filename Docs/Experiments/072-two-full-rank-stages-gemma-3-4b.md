# Experiment 072: two full-rank stages on Gemma 3 4B IT

Prepared 2026-11-12. The uniform control reached 12 of 34 committed blocks before a CUDA OOM on resume on 2026-11-13.

## Recipe

Transfer Experiment 071 to `google/gemma-3-4b-it`, pinned at
`093f9f388b31de276ce2de164bdc2081324b9767`. The pinned checkpoint is already cached locally.
The parent configuration is frozen in `experiments/recipes/references/experiment071.json`, with semantic hash
`sha256:2ae172330b1e6d7db5b1ff878a18f3c5b0a169706426a02088c8f9c9fbeb25f5`.

Preserved from 071: exactly two independently scaled full-rank stages per quantization owner, stacked QKV,
600 outer × 2 inner ADMM iterations, optional PyTorch compilation enabled, calibration, tuning epochs and learning
rates, global distillation, no residual columns, and a 1,000-example task limit. Maximum-rank promotion follows
allocation and may exceed the nominal 2-BPW allocation target. Embeddings/output heads are not given these stages.

The established 4B template supplies the model/revision and bounded-memory batch settings: nonfactorized tuning 4,
factorized tuning 1, post-block refit 1, microbatch 1, and resident forward batch 4. Inline quality is disabled;
final quality uses the existing 4B dense backend. Other 071 recipe settings, including its retry thresholds, remain
inherited. These batch/backend changes mean this is not an identical-execution comparison with 071.

The text decoder has 34 blocks, with 170 quantization owners and 340 independent stages:

| Owner | Matrix shape | Primary rank | Residual rank |
| --- | --- | ---: | ---: |
| MLP gate / up | 10,240 × 2,560 | 2,560 | 2,560 |
| MLP down | 2,560 × 10,240 | 2,560 | 2,560 |
| Stacked QKV | 4,096 × 2,560 | 2,560 | 2,560 |
| Attention output | 2,560 × 2,048 | 2,048 | 2,048 |

These are planned ranks derived from the pinned text configuration, not completed fit results. The vision tower
is not quantized by this experiment. The standard complete export workflow also exports the separate validated
BF16 `mmproj-BF16.gguf` required by the multimodal source.

## Single-pass completion update, 2026-11-13

The campaign now finishes the existing uniform-control run and uses it as the final model.
Every quantization owner is forced to maximum rank after allocation, so KL sensitivity measurements
cannot change its ranks. The control retains the same calibration, 600�2 fitting, tuning batches,
epochs, seeds and global distillation. Foldable MLP multipliers are disabled in both recipes.
The original control configuration, artifact identities and journals remain the source of resume state.
The complete export and quality workflow publishes the GGUF, statistics and reports under `Results/072`.
Reports retain the control run's actual provenance.

This removes the second 34-block compression pass, allocation KL profiling and candidate rank-response
probing from this campaign. The explicit reuse option requires maximum rank for all projections and
matching distillation. Allocation-profile overrides are rejected for this mode.
The KL teacher dataset remains available for future allocation experiments. This path does not consume it.
Global distillation retains its separate teacher protocol.

Resume from the repository root with the existing command.

```powershell
.\.venv\Scripts\python.exe tools\run_experiment072_campaign.py
```

`--plan-only` shows the selected mode. `--dry-run` prints the exact control configuration and publication
settings without loading weights or requiring an allocation dataset. The runner retains its 80-slice
ceiling and one-block process boundaries. Previously completed blocks resume from the saved CPU activation
boundary with `restore_completed_blocks=False`.

Slice 022 recorded block index 17 taking 1,262.7 seconds. Its logged factorized tuning spans totaled
506.0 seconds, ordinary tuning 153.2 seconds, post-block refitting 86.1 seconds and checkpoint commit
74.9 seconds. Fitting and scale/outlier stages totaled 276.4 seconds. Shared QKV processing includes
additional tuning and overlaps these categories, so these figures are not a complete additive breakdown.
Startup inventory and model loading together took 16.8 seconds.

At that block's rate, removing a second 34-block compression pass avoids approximately 11.9 hours.
That is an extrapolation from one block, excluding additional profiling and distillation savings.
It is not a measured end-to-end speedup. Eighteen blocks were committed when this investigation began,
and the following block had partial progress before a user interruption.

A CPU-only 2.5-GiB write/hash experiment compared serial and concurrent hashing twice each.
Serial runs took 24.20 and 23.48 seconds, concurrent runs 27.62 and 24.04 seconds.
Hashes matched, but concurrency gave no improvement and the proposed writer change was discarded.

The new path has CPU regression coverage for preserving the control configuration, skipping allocation
preparation, rejecting variable-rank reuse, and publishing an unnumbered control under numbered results.
The real campaign dry run exactly matched the stored control manifest. The full default suite passed
1,622 tests with 53 deselected; the final publication/control-focused check passed 15 tests. Ruff and
mypy passed. Full-model completion, exported quality and the final wall-time saving
remain to be measured when the campaign resumes.

## Setup validation

- The real `--plan-only` command passed; its output is `outputs/072-setup-plan.json`.
- CPU tests check recipe inheritance, two-stage planner promotion, no-compute setup behavior, and the 80-slice runner.
- Tiny CPU teacher tests cover both flat and nested configurations, dataset reuse, and matching KL consumer results.
- Model-size defaults were checked from the pinned local config without loading weights. A safetensors-header check
  verified all 170 owner shapes and 340 planned stages; see `outputs/072-rank-shape-check.json`.
- Full default test suite: **1,618 passed, 53 deselected** (CUDA/external-data/performance tests excluded).
  Repository-wide Ruff and mypy's 224-file source check passed.

Full-model memory use, compiled execution, compression quality and export/runtime acceptance remain unmeasured.
