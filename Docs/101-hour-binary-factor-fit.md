# Experiment 066: one-hour binary-factor fit

Date: 2026-09-13. Status: full one-hour-cap search and float64 audit complete.

## Result

The 59.64-minute search found **4.6056% lower weighted residual error energy**
and **2.3299% lower weighted NRMSE** than the fresh 800-step current-method
control. The best of three ordinary control seeds was seed zero, so the gain
also holds against the best ordinary start tested.

| Method | Weighted error energy | Weighted NRMSE |
| --- | ---: | ---: |
| Current 800-step fit | 5.085753142 | 0.488423971 |
| Current fit plus local search | 5.061083010 | 0.487237900 |
| Best within the hour | **4.851525611** | **0.477044064** |

The winner used BF16 latents with TF32-enabled solves, 51,200 ADMM outer
iterations, two scale-fit passes, and two control plus two tabu refinement
passes. Its ADMM stage took 807.46 seconds, versus 10.99 seconds for the
ordinary baseline including scale fitting and scoring. The campaign completed
25 joint fits and 236,656 outer iterations. Ninety percent of the final gain
was available after 20.82 minutes; 99% after 40.98 minutes. Extra starts did not
beat the winner. The curve shows diminishing returns.

The winner and baseline were reconstructed in float64; the winner's error
differed from its selection score by only 6.56e-10 relative. All 110 saved
factor states passed a fresh SHA-256 check. Peak fitting workspace was
0.834 GiB allocated / 0.916 GiB reserved CUDA memory; observed Windows peak
working set was 1.662 GiB.

See the [complete report and plots](../Results/066/report.md),
[machine-readable comparison](../Results/066/summary.json), and retained
`evidence/066/hour-fit/best.safetensors`. This result changes no production
policy or model-quality gate. The protocol below was frozen before the run.

The question is how much reconstruction headroom an additional hour of search
can find for one full-size binary factorization at unchanged storage. This is
a layer replay, not a complete compression experiment or a model-quality gate.
It produces an actual reusable binary-factor fit and a time/error curve. It
cannot establish a global optimum for this discrete nonconvex problem.

## Fixed case and comparison

- Pinned `google/gemma-3-1b-it`, revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Middle block 12 `mlp.down_proj`, full 1,152 by 6,912 matrix, rank 1,152.
- Retained Experiment 041 production replay supplies diagonal Fisher weights
  and seven removed columns: 429, 1791, 3043, 3886, 4002, 5534, 5891. Its
  immutable owner loader preserves the original removed-column importance
  policy. The same residual target and importance vectors are used by all arms.
- Representation: two unrestricted sign matrices and three scale vectors;
  outlier positions/values, rank, objective, and storage do not change.
- Current-method control: today's production ADMM, 800 outer iterations, five
  power iterations, cubic penalty, regularization 0.03, native orientation,
  BF16 latents with FP32 solves, then two FP32 IEEE scale-fit passes. Seed 0 is
  the main control. Seeds 1 and 2 run inside the hour to show start variability.
- The retained historical tuned factors are scored only for context. They do
  not initialize the timed search. The incumbent begins at the fresh control.

Experiment 060 uses these ADMM settings and this physical rank, but its
resident calibration artifacts have been removed from this checkout. This
experiment compares its current *fitting algorithm* on a retained older
objective; it does not claim to reproduce Experiment 060's final weight errors,
8.8% INT8 outliers, block tuning, global distillation, or model quality.

## Speed and search design

Reuse the established GPU factorizer and exact-objective local-search kernels.
Avoid full-model loading, activation collection, export, and model evaluation.
After an eight-step warmup for each mode, time matched 64-step seed-101 pilots:
BF16 or FP32 latents with IEEE or TF32 matrix multiplication, plus PyTorch's
FP32 `medium` mode. All candidates receive the same IEEE scale fit and common
IEEE reconstruction score. Select the fastest pilot whose error energy is at
most 5% above the FP32 IEEE pilot. This is the fastest measured eligible mode,
not a claim to have exhausted possible kernel optimizations.

The additional 3,600-second clock starts after the baseline and speed pilots.
Scoring, local refinement, checkpoint writes, and additional control seeds all
count against the hour. Input preparation and the final independent audit do
not. The runner records actual elapsed time; it reserves ten seconds at the
end and can finish slightly early when another bounded operation cannot fit.

The predeclared schedule is:

1. Refine the control with eight one-bit/variable-depth outer passes followed
   by eight tabu outer passes. Each bounded call uses eight scale passes,
   16 one-bit passes or two 256-step tabu sweeps, with tabu tenure 8 and jitter
   4. This is a checkpointable local-search diagnostic, not an exact replay of
   the older 64-scale-pass 8+8 research recipe.
2. Run two additional unmodified 800-step control seeds.
3. Pair the fastest mode with a distinct latent precision at 800, 1,600,
   3,200, 6,400, 12,800, 25,600, and 51,200 outer iterations, seed 0. Retaining
   FP32 candidates matters because a short speed pilot cannot expose long-run
   BF16 rounding limitations.
4. Spend remaining time on new seeds at 6,400, 25,600, and 51,200 iterations,
   alternating the two precisions. Reduce the final horizon using measured
   iteration time and a 20% allowance instead of discarding a long partial fit.
5. Apply two control and two tabu passes to candidates within 1% error energy
   of the incumbent. Keep only full-precision improvements as the incumbent.

Each ADMM iteration checks the deadline through the existing recorder API.
Incomplete fits cannot replace the incumbent. Every completed fit has a
hash-bound safetensors checkpoint. The report and event journal record starts,
progress, actual iterations, precision, seed, elapsed time, and accepted best
states. Resume validates the runner and fixture hashes and preserves the
original deadline; interruption cannot extend the one-hour budget.

## Metrics and interpretation

The primary metric is the exact same diagonal-Fisher residual error energy for
all arms. Report weighted NMSE and NRMSE separately: relative energy reduction
is not relative root-error reduction. Raw NRMSE is diagnostic. Final baseline
and winner scores use independent float64 reconstruction and reduction.

The final report must include the control seed range, local-only improvement,
joint-search horizon/precision curve, best checkpoint, actual time, represented
bytes, peak CUDA allocator memory, and the comparison's scope limits. No test
or held-out token slice is opened or used for selection. A matrix-fit gain
requires a later fresh functional comparison before changing production policy.

## Run

```powershell
.\.venv\Scripts\python.exe tools\probe_hour_factor_fit.py
```

Defaults write `evidence/066/hour-fit/fixture.json`, `fixture.safetensors`,
`events.jsonl`, `report.json`, all completed factor states, and
`best.safetensors`. Rerunning the same command resumes the same deadline or
returns the completed comparison. `--prepare-only` captures the CPU fixture.
`--output outputs/066/canary --seconds 30` is explicitly a lifecycle canary;
its result does not satisfy the one-hour experiment.

No resident algorithm, persisted production schema, deployment format, or
runtime changes. Therefore this analysis does not increment the resident
algorithm version or require a GGUF export.
