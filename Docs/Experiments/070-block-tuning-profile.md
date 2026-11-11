# Experiment 070 block tuning profile

Measured on 2026-11-11 using the RTX 4000 Ada laptop GPU and the repository's PyTorch 2.12.1 CUDA environment.

The compiled block forward/backward candidate reduced warmed tuning time by 17% in the initial paired replay.
Its first invocation was 27% slower overall after including compilation. Production tuning remains on its current
execution path by default because the block-bounded campaign needs an end-to-end benefit and the compiled
updates have small numerical differences. Opt-in compilation is available through
`block_tuning.compile_forward: true`. This wraps each block's tuning forward callable for nonfactorized,
factorized and refit stages. It preserves batch staging, parameter selection, optimizer calls, schedules and
checkpoint ordering. The setting participates in run identity; resident algorithm version is 68.

## Workload

The diagnostic loads the pinned Gemma 3 1B BF16 model, generates block-zero inputs and teacher outputs from the
070 run's saved 256-by-2048 calibration tokens, and replaces `mlp.gate_proj` with the retained frozen gate factors
and independent residual stages. Remaining projections begin at BF16. This is a post-fit representative block
replay. It differs from the historical intermediate state before 070's first projection was tuned.

Factorized and nonfactorized tuning each start independently from the same block state. The comparison retains
eight epochs, batch and microbatch size eight, seed zero, 256 optimizer updates, the original parameter selectors,
learning rates, ParityAdamW, cosine schedule, and legacy training-loss selection. Timing comparisons use 070's
saved block-output importance weights. Epoch checkpoint serialization is excluded.

Compiled mode uses `torch.compile` for the block forward function, with `emulate_precision_casts=True` and
CUDA graphs disabled. It measures a first invocation, reloads the initial model state, and repeats with a fresh
optimizer and the same shuffled batch sequence. The warmed measurement excludes compilation.

## GPU profile

The initial operator traces used the same block and batches with unweighted MSE. Subsequent timing and parameter
comparisons used the saved output importance weights. Trace collection/export overhead is excluded from the
timing comparison below.

| Operator category | Factorized GPU self time | Nonfactorized GPU self time |
| --- | ---: | ---: |
| Matrix multiplication | 35.3% | 25.2% |
| Elementwise multiplication | 24.8% | 22.2% |
| Attention backward | — | 19.5% |
| All tensor copies | 14.4% | 12.7% |
| Host-to-device copies, included in copies above | 7.9% | 5.6% |

Trainable factors execute as two linear operations per stage. The hot path already avoids dense factor
reconstruction. Host transfers are bounded and prefetched through pinned buffers. Fusion of elementwise
operations was therefore selected for the first optimization trial.

A subsequent weighted micro profile measured the following CUDA-event intervals over all 256 updates.
It ran separately from the uninstrumented A/B timing measurements.

| Phase | Factorized | Nonfactorized |
| --- | ---: | ---: |
| Forward | 10.663 s | 10.792 s |
| Backward | 9.318 s | 23.806 s |
| Loss | 0.589 s | 0.579 s |
| Optimizer update | 0.954 s | 0.712 s |

Each stage staged 18 GiB across eight epochs. CPU batch-staging wall time includes waiting for reusable buffers,
so it cannot be interpreted as transfer execution time. The operator trace measures the copy-stream work directly.
The phase reports are in `outputs/tuning-profile/micro/*-phases.json`.

## Initial paired timing

| Stage | Existing path | Compiled first invocation | Compiled warmed | Warm reduction |
| --- | ---: | ---: | ---: | ---: |
| Factorized | 21.079 s | 30.019 s | 16.643 s | 21.0% |
| Nonfactorized | 32.469 s | 38.181 s | 27.888 s | 14.1% |
| Combined | 53.547 s | 68.200 s | 44.531 s | 16.8% |

A second process using the populated compiler disk cache measured 17.300/30.127 seconds for first factorized/
nonfactorized invocation and 15.871/27.706 seconds warmed. Its paired eager run measured 21.119/32.623 seconds.
That is an 11.8% improvement for the cached first invocation and an 18.9% warmed improvement. The first trial
includes an initially unpopulated compilation cache. Different blocks, ranks and trainable parameter sets can
trigger more compilation or compiler fallback, so this benefit still requires a full block/campaign measurement.

Final weighted factorized loss changed from 0.03609436 to 0.03610627, an increase of 0.033%.
Final weighted nonfactorized loss changed from 0.02061685 to 0.02063009, an increase of 0.064%.
Maximum parameter-tensor relative L2 differences were 0.143% and 0.351%, respectively. A second eager run reproduced
the factorized result exactly, while its nonfactorized final loss was 0.02060144 and its maximum parameter-tensor
relative L2 difference from the first eager run was 0.301%. Nonfactorized differences therefore include ordinary
run-to-run variation. These comparisons establish neither identical updates nor a quality regression from compilation.

These results do not establish a campaign-level speedup or model-quality parity. The proposed two-hour savings
from halving tuning remains an unmeasured projection. Before promoting compilation, replay complete block stages
including model replacement and process restarts, account for compilation/recompilation, and evaluate final quality.

## Reproduce

Run from the repository root with an idle CUDA device. The script acquires the project device lease and validates
the source artifact hashes. Select `eager`, `compiled`, `profile`, or `micro`. It writes timings, protocol metadata
and final parameters. Profile mode exports operator tables and Chrome traces; micro mode records forward/backward,
loss, optimizer and staging phases with CUDA events and transfer counters.

```powershell
.\.venv\Scripts\python.exe tools\profile_gemma_block_tuning.py `
  --run-output evidence\070\070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it `
  --snapshot C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752 `
  --output outputs\tuning-profile\replay --mode compiled
```

Initial local evidence is in `outputs/tuning-profile`: `baseline.json`, `compiled.json`, `compiled.log`,
`parameter-comparison.json`, and the two `*-operators.txt` / `*-trace.json` pairs. Profile timings include profiler
overhead and should be compared using the operator tables, while the eager and compiled modes provide wall times.
