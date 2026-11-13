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


## Block tuning profile, 2026-11-13

The profiling tool now reads the saved recipe's batch sizes, microbatch size, epoch count and
learning rates. It loads the source model on CPU and moves only block zero to CUDA. It supports
selecting factorized or nonfactorized tuning independently. The representative 4B replay uses the
retained gate projection's two full-rank stages, source BF16 remaining projections, saved output
importance and all 256 calibration sequences of length 2,048. Eight epochs at batch/microbatch one
produce 2,048 optimizer steps. The retained gate is already fitted and tuned, so this diagnostic
replays further tuning from a reproducible saved state rather than reproducing the original entry state.
Epoch checkpoint serialization is excluded from the isolated timing.

The synchronized phase profile took 71.28 seconds including initial compilation. The first epoch
took 20.73 seconds and subsequent epochs approximately 7.2 seconds. Median sampled CUDA spans were
8.61 ms for forward, 8.41 ms for backward and 10.44 ms for optimizer updates. Host batch staging
accounted for 37.36 seconds, including waits for queued CUDA work. These overlapped timings cannot
be added together or interpreted as 37 seconds of pure transfer overhead. The trainable gate uses
factorized matrix products in its forward path.

The accepted optimization computes AdamW's square root directly into its reusable denominator
buffer for optimizer groups whose tensors each contain at least 262,144 elements. Smaller or mixed
groups retain the batched copy/square-root path to avoid additional launch overhead. This removes
a full-size second-moment copy while preserving the elementwise recurrence,
BF16 rounding and Kahan compensation. A GPU test on the 4B gate's factor shapes matched all six
parameter/gradient/state buffer sets exactly across ten updates. Sixty-update timings were
0.648/0.609 seconds for the original and 0.568/0.565 seconds for the optimized update.
A compiled fused update was also tried and rejected because its BF16 buffers differed immediately.

The first uninstrumented compiled eight-epoch comparison took 57.994 seconds before and 56.390
seconds after warmup, a 2.77% reduction. All 22 parameter tensors, containing 133,734,912 values,
matched bit-for-bit, and both final losses were 1.2732211351394653. Repeated timings were 58.072
seconds before and 56.798 seconds after, a 2.19% reduction. The size probe found six-vector groups
of 2,560/10,240 elements slower with direct square roots, while 262,144/1,048,576-element groups
were faster; this motivated the conservative group-size threshold. The final guarded implementation
took 57.023 seconds, a 1.67% reduction versus the first baseline. All repeat and final outputs matched
the original 133,734,912 parameter values bit-for-bit and retained the identical final loss.
This measures one representative
gate-tuning replay, not complete campaign time or downstream language-model quality.

Evidence is under `evidence/072/tuning-profile-before`, `tuning-benchmark-before`,
`tuning-benchmark-after`, `tuning-benchmark-before-repeat`, `tuning-benchmark-after-repeat`,
`tuning-benchmark-final`, `optimizer-sqrt-probe.log`, `optimizer-size-probe.log` and
`tuning-optimizer-comparison.json`.
The numerical recipe and resume identity are preserved because the removed operation only copied
scratch data before the same square-root computation. The optimization applies when a new process
loads the updated optimizer module.

Reproduce the isolated factorized benchmark with the pinned local snapshot.

```powershell
.\.venv\Scripts\python.exe tools/profile_gemma_block_tuning.py `
  --run-output evidence/072/072-d2-uniform-control-gemma-3-4b-it `
  --snapshot C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767 `
  --output evidence/072/tuning-benchmark-local --mode compiled --stage factorized
```

Use `--mode micro` to collect phase counters. The tool takes the CUDA lease for the complete run.


Validation of the final optimization passed Ruff and mypy. The full default suite recorded 1,628
passes, 53 deselections and one Windows `WinError 32` failure in the unchanged concurrent run-registry
lock test. The registry, optimizer and tuning tests then passed together (43 passed, 2 deselected),
and the registry suite passed again independently (4 passed). The full-suite result is therefore
recorded with that intermittent failure rather than reported as an entirely passing run.
