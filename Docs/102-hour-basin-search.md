# Experiment 067: another hour searching binary-factor basins

Date: 2026-09-13. Status: completed, including the user-requested 5% amendment.

## Result

The 59.9208-minute wall-clock campaign completed 330 proposals and 1,823
refinement passes. The final fit has **4.63468% lower weighted error energy**
than the original 800-step baseline and **0.03052% lower error** than the
previous hour's winner. Weighted NRMSE reductions are 2.34483% and 0.01526%,
respectively. The additional hour found very little extra weighted-fit headroom.

| Fit | Float64 weighted error energy | Weighted NRMSE |
| --- | ---: | ---: |
| Ordinary 800-step baseline | 5.0857531418 | 0.4884239712 |
| Previous hour winner | 4.8515256113 | 0.4770440644 |
| Continued local control | 4.8504661676 | 0.4769919747 |
| Second-hour best | **4.8500448898** | **0.4769712602** |
| Best completed 5% proposal | 4.8505008966 | 0.4769936823 |

The 41 completed 5% proposals did not beat the incumbent. They initially
changed 462,952-465,723 signs and ended 75,743-95,106 signs away from their
parent after refinement (median 86,164; raw corresponding-entry distance).
Their best result remained 0.00940% worse than the overall winner. The larger
perturbations started at 37.28 minutes after a checkpointed pause; that pause
counts against the original deadline.

An independent float64 audit of all 9,289,728 individual sign flips found
zero improving flips for the overall winner with all scales fixed, at absolute
energy tolerance 1e-10. This is fixed-scale one-bit stationarity only, not a
certificate for scale-refitted or joint moves. The best 5% result still had
two improving single-bit flips, with largest gain about 1.08e-9 energy.

All 1,823 saved states and the preserved pre-amendment source/report snapshots
passed fresh hash checks. The winner is `evidence/067/basin-search/best.safetensors`,
SHA-256 `f8a9d27b3ef4987bd3a3b59f6705aec2c65384742a985272274e9c7f562054e3`.
The evidence retains 15.835 GiB of factor checkpoints. Ninety percent of the
new gain arrived by 10.42 minutes and 99% by 28.32 minutes.

This campaign generated **zero fresh independent ADMM starts**. It directly
refined retained factors; the inherited winner originally came from 51,200
ADMM iterations. The current ADMM API cannot initialize from a candidate, so
candidate-initialized ADMM refinement remains untested. Distinct final sign
hashes (305 here) are not independent basins. Earlier timings suggest roughly
300-370 fresh 800-step starts per hour or 4-5 starts at 51,200 steps, before
extra refinement/checkpoint overhead. Broad random-start coverage remains a
separate unresolved experiment.

See the [complete report and plot](../Results/067/report.md),
[machine-readable summary](../Results/067/summary.json), and
[independent large-proposal audit](../Results/067/large-flip-audit.json).
The original protocol below was frozen before the run and is retained verbatim
in `evidence/067/basin-search/protocol.md`; the amendment is described at the end.

Validation: 41 focused checks passed; the full default suite passed 1,483 tests
with 49 deselected in 109.71 seconds. Repository-wide Ruff passed, and mypy
passed for all 217 source files. The full suite ran after the timed search.

The question is whether another hour spent escaping the previous solution can
find a better fit than Experiment 066. The original 800-step production-method
baseline and the previous hour's winner are both retained and independently
rescored. This is a warm-start experiment: the first hour's computation is
inherited, and the new 3,600-second budget is additional.

## Fixed case

Use Experiment 066's hash-verified full 1,152 by 6,912 Gemma block-12
`mlp.down_proj` fixture at rank 1,152. Target, diagonal Fisher weights, seven
outlier columns, all sidecars, representation, and storage remain identical.
The fixture derives from retained Experiment 041 calibration; it is not a
reproduction of the missing Experiment 060 calibration or final model quality.
See [Experiment 066](101-hour-binary-factor-fit.md) for exact input provenance.

## Search design

Reuse the existing GPU binary-search kernels with IEEE FP32 arithmetic and
exact common-objective acceptance. ADMM has no retained warm-start latent or
dual state, so this experiment spends the new hour on direct factor search.
Large batched populations and exponential joint windows would spend much of
the budget on full-size matrix products; a serial portfolio keeps the working
set small and makes each completed refinement checkpointable.

1. Continue ordinary refinement from the previous winner, alternating one
   descent pass and one tabu pass. Freeze the best result as a separate
   comparator after 240 seconds or four successive relative improvements
   below 1e-6. The clock includes this control. It is a bounded refinement
   comparator, not a proof that its neighborhood is exhausted.
2. Cycle through twelve proposal types: randomly chosen low-cost sign flips
   at 32, 256, 2,048, or 8,192 flips per factor; uniform flips at probabilities
   0.0001, 0.001, or 0.005; coupled flips in 2 or 8 rank components at 2%, 10%,
   or 25%; a sparse pair/component-replacement pass; and diverse retained
   Experiment 066 ADMM states selected by mode/seed. Low-cost proposals sample
   from the lowest 16 times the requested number of exact fixed-scale costs.
3. Each proposal receives up to eight bounded refinement passes. Descent uses
   32 one-bit passes plus two length-64 variable-depth passes. Every third
   local pass uses two length-256 tabu sweeps, tenure 8 and jitter 4. All calls
   use eight scale-fit passes. Broad moves use one pair pass with pool 8 and
   four component replacements with four alternating steps. Stop unpromising
   proposals after two passes if still more than 0.3% above the incumbent;
   stop after three passes if relative improvement falls below 1e-7.
4. Two out of every three proposals start at the best state. The other starts
   at an exploratory walker, whose Metropolis temperature decreases linearly
   from 0.1% to 0.005% of best error energy. The walker may accept worse results;
   the recorded best only accepts verified improvements. Reset the walker to
   the best every 18 proposals. Deterministic proposal seed is 67000 + index.
5. Reserve the final 120 seconds for additional incumbent descent, tabu, and
   sparse broad moves. Each operation is admitted using its measured maximum
   duration plus a 40% allowance. Work completed after the deadline is retained
   but cannot replace the best. The run may finish a few seconds early when
   another bounded call cannot fit.

All proposal scoring, refinement, factor hashing, checkpoint writes, and
control work count against the hour. Input checks and the final read-only
audit do not. Resume checks input and numerical-source hashes and preserves
the original deadline. Interrupted proposals are skipped; completed states
remain valid. Canonical hashes remove sign gauges for diversity reporting,
but do not prove that two states inhabit distinct attraction basins.

## Comparison and audit

Report weighted error energy, NMSE, NRMSE, raw NRMSE, time/error trajectory,
proposal outcomes and uphill walker accepts, factor-state bytes, and CUDA
allocator peaks. Compare the new best with the ordinary baseline, previous
hour winner, and continued-local-refinement control. Preserve every completed
state and independently verify its hash at the end.

Reconstruct all four comparisons in float64. An independent float64 audit
computes the exact error change of every individual sign flip with scales and
the opposite factor fixed. It reports the number of improving flips and best
single-flip gain at absolute energy tolerance 1e-10. A zero count certifies only
this fixed-scale one-bit neighborhood. It does not certify joint flips,
scale-refitted flips, a strict local minimum in the full parameter space, or a
global optimum. The audit never modifies the selected state outside the budget.

No production algorithm, model tuning, export format, GGUF, or model-quality
gate changes. The runner is an analysis-only experiment.

```powershell
.\.venv\Scripts\python.exe tools\probe_hour_basin_search.py
```

Defaults retain evidence in `evidence/067/basin-search`. A lifecycle canary uses
`--output outputs/067/canary --seconds 60`; it cannot satisfy the hour-long run.

## User-requested amendment during the run

After approximately 35 minutes, the user requested much larger sign flips,
suggesting 5%, followed by refinement. The current worker was stopped and its
last complete report and exact runner source were preserved as
`pre-large-flip-report.json` and `pre-large-flip-runner.py`. The best committed
fit remains the incumbent. Both snapshots are hash-bound in `protocol_changes`.

The remaining original wall-clock budget uses independent Bernoulli 5% flips
in each factor, about 464,486 changed signs per proposal. Each candidate may
receive up to 64 refinement passes, with at least 12 passes before pruning for
an error more than 1% above the best or relative progress below 1e-7. Existing
descent/tabu kernels and the final 120-second incumbent-polish reserve remain
the same. The pause to implement and validate this amendment counts against
the original deadline; it does not grant extra search time.

The resumed command is:

```powershell
.\.venv\Scripts\python.exe tools\probe_hour_basin_search.py --amend-large-flips
```

The final report separates the larger proposals, shows their recovery and
remaining sign distance from their parent, and reports the exact amendment
time. CUDA allocator counters cover the resumed worker; the stopped worker's
allocator peak was not retained. Host peak working set is sampled in both
phases. No new independent ADMM starts are added by this amendment.
