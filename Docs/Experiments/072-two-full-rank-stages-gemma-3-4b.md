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


## Faster optimizer comparison, 2026-11-13

The profiler now offers experimental optimizer selection and reports post-training weighted and
unweighted MSE, gate reconstruction error, the epoch-loss trajectory and initial evaluated loss.
The legacy training-average metric alone can rank candidates differently from evaluation of the
final parameters. Each timed run follows a cold invocation and a reset to the identical starting
weights, with a fresh optimizer. All comparisons below use seed zero and 256�2,048 tokens,
eight epochs, factorized batch/microbatch one and the saved per-parameter learning rates.

The corrected comparison initializes both stages from saved continuous ADMM latents and fitted
scales. Production tuning uses continuous latents. The earlier frozen-sign replay is a different
starting point and its optimizer-ranking results are preliminary. The diagnostic reconstructs the
uniquely ordered factorization/scale-fit pairs between the primary factorization and layer commit,
validates their artifacts, and rejects ambiguous histories. Protocol JSON records all four artifact IDs.
The remaining block-zero projections use source BF16 weights. This remains an isolated gate-tuning
replay rather than the complete original block state after nonfactorized tuning.

| Optimizer | Warm seconds | Post-tuning weighted MSE | Unweighted MSE | Gate relative Frobenius error |
| --- | ---: | ---: | ---: | ---: |
| parity | 58.793 | 0.532435417 | 0.145310551 | 0.249616146 |
| torch-fused-scales-fp32 | 45.698 | 0.527085960 | 0.145346284 | 0.249524161 |
| torch-fused-fp32 | 52.771 | 0.528063416 | 0.145157367 | 0.249658123 |

The scale-only FP32 variant reduces warm tuning time by 22.27%. Its weighted MSE decreases by
1.00%, unweighted MSE increases by 0.025%, and gate relative Frobenius error decreases by 0.037%.
The full FP32-master variant reduces time by 10.24%, weighted MSE by 0.82%, and unweighted MSE by
0.105%, while gate relative Frobenius error increases by 0.017%. These are one-block, one-seed,
in-sample measurements. Other projections, held-out quality, full-model perplexity and task quality
have not been validated for these optimizer variants.

The best speed candidate uses native `torch.optim.AdamW(fused=True)` with betas `(0.9, 0.99)`,
epsilon `1e-6`, BF16 binary-factor master parameters and FP32 scale master parameters. Factor and
scale computations still cast to BF16 activations through the existing trainable forward path.
The diagnostic computes the weight-error metric after casting a temporary gate copy to BF16,
so the compared reconstruction uses BF16 storage precision.

The exploratory frozen-sign screen rejected plain fused AdamW with all-BF16 parameters. It ran
22% faster but gave post-training weighted MSE 5.13097 versus 1.27199 for the parity optimizer.
A compiled parity-update experiment initially hit coefficient specialization/recompilation limits.
Dynamic CPU scalar coefficients plus the Visual C++ developer environment allowed it to complete,
but its final weighted MSE was 392.46 on that same screen. It is unsuitable for adoption.
The native fused alternatives completed from the normal repository shell.

The production optimizer remains ParityAdamW. Experimental substitution is scoped to the profiler
and restored on exit, including errors. Native fused AdamW has different optimizer state, lacks
ParityAdamW's BF16 Kahan buffers, and FP32 scale masters need explicit persistence/storage handling.
A production option needs its own configuration identity and resume tests before use in the active
campaign. Existing 072 checkpoints were not modified by these diagnostic runs.

Reproduce the leading candidate with the local pinned snapshot.

```powershell
.\.venv\Scripts\python.exe tools/profile_gemma_block_tuning.py `
  --run-output evidence/072/072-d2-uniform-control-gemma-3-4b-it `
  --snapshot C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767 `
  --output evidence/072/optimizer-local --mode compiled --stage factorized `
  --optimizer torch-fused-scales-fp32 --initialization admm --seed 0
```

Use `--optimizer parity` for the control or `--optimizer torch-fused-fp32` for FP32 master factors
and scales. `compiled-parity` is retained as a rejected research comparison and requires a Visual
C++ developer environment on this Windows installation. Timing and error evidence is in
`evidence/072/optimizer-admm-comparison.json` and the `optimizer-admm-*` directories. Frozen-sign
screening evidence is in the other `optimizer-*` directories and must be compared within that protocol.

Validation passed: 45 focused tests (2 deselected), the full default suite (1,635 passed,
53 deselected), Ruff and mypy. These software checks do not establish full-model quality
for the experimental optimizers.


## Fused AdamW held-out check and continuation (2026-11-13)

Tested block 0 gate, up, down and attention output projections with seeds 0 and 1,
all 256 x 2048 training sequences, eight epochs and 2048 updates. Evaluation used
32 separate WikiText-2 test windows, excluded from training and epoch selection.
This is isolated projection reconstruction quality, not full-model perplexity.

| Projection | Warm tuning time reduction | Weighted held-out MSE change | Decision |
| --- | --- | --- | --- |
| Gate | 21.3-23.2% | -1.01% to +1.10% | Enable |
| Down | 23.4-24.1% | -7.80% to -5.23% | Enable |
| Attention output | 8.4-9.7% | -1.35% to +0.10% | Enable |
| Up | 19.8-21.0% | -0.73% to +4.77% | Keep parity optimizer |

The acceptance bound was 2% held-out MSE regression per seed, checking weighted
and unweighted error and Frobenius error. Results are in
`evidence/072/optimizer-heldout-comparison.json`. Native fused AdamW keeps FP32
scale masters during tuning and restores original storage precision afterward.

Experiment 072 enables the three accepted projections starting at zero-based
block 19; the partially saved block 18 finishes with the parity optimizer.
The explicit journal-pinned metadata continuation preserved all 111 journal
records, 18 completed blocks and three saved projections in block 18. Tensor
and activation references are unchanged. Original metadata is backed up under
`state/optimizer-continuation/`. Resume discovery found 111 valid records and
zero orphans, with the next cursor at block 18 self_attn.attn_qkv. A fresh audit
validated 693 artifacts (55.2 GB). No compression was started. Resume with the
existing `python tools/run_experiment072_campaign.py` command.

Validation: 1649 tests passed, 54 deselected in the full CPU suite; additional
focused provenance tests and the CUDA native checkpoint/resume test passed.
