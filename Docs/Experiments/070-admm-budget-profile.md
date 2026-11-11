# Experiment 070: ADMM iteration budgets

Measured on 2026-11-11.

## Question and protocol

Can shorter ADMM schedules or fewer inner power iterations approach the 800-outer,
5-inner fit while reducing elapsed time? Measurements use the RTX 4000 Ada laptop
GPU, PyTorch 2.12.1+cu130, and the pinned Gemma 3 1B snapshot.

The diagnostic tool is `tools/profile_admm_budgets.py`. It uses blocks 0, 12 and 25,
with gate, up, down, attention output and stacked QKV owners. Each arm keeps the
retained 070 primary/residual stage ranks, saved Fisher importance (including QKV
member multipliers), BF16 ADMM arithmetic, regularization, scale-fit policy and
represented-bit budget. Every owner is evaluated with both seed 0 and seed 1;
results are paired within owner and seed, with no best-seed selection.

Targets are the original BF16 model weights, not the historical intermediate
weights after 070's preceding block tuning. Each arm starts from fresh random
factors using the same seed. Later stages use the production residual-stage seed
derivation and fit the remaining error after earlier stages, including two
alternating scale-fit passes. The next residual subtracts the FP32 reconstruction
of the fitted scales. Final scoring uses hard signs and BF16-rounded scales.
This reproduces the fitting structure but is not an exact historical-run replay.

The cubic penalty schedule spans the requested outer budget. A 400-step arm is
a complete compressed schedule, not the first 400 steps of the 800-step arm.
Changing inner iterations changes the power approximation in initial, iterative
and final projections. All other ADMM parameters are retained.

Metrics are weighted weight NMSE, relative Frobenius error, and mean per-sequence
linear-output NMSE on eight WikiText validation windows of 128 tokens, offsets
256 through 263. These inputs are captured from the BF16 teacher and are not used
to fit factors or scales. Scoring reconstructs and multiplies using FP32 tensors
with the production TF32 allowance to isolate factor approximation; it does not
exercise native packed kernels.
The activation metric is local to each projection;
it does not measure error propagated through the complete compressed model.

Timings synchronize CUDA around each fit and separate ADMM and scale fitting.
The total includes residual assembly and frozen-precision reconstruction, but
excludes model loading, activation capture, artifact reading and metric scoring.
Arm order reverses across owners/seeds to reduce systematic ordering bias.
Instrumented phase profiles are separate from ordinary timing comparisons.

## Outer-iteration results

Completed 120 fits: 15 owners, two seeds and four budgets. Each row totals time
over the same 30 owner/seed pairs. Error changes are means of paired percentage
changes, giving each pair equal weight; these are relative increases in NMSE,
not percentage-point changes or model-accuracy changes.

| Outer / inner | Total fit time | Time saved vs 800x5 | Weighted NMSE increase | Activation NMSE increase | Worst activation increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800 / 5 | 412.30 s | — | — | — | — |
| 400 / 5 | 213.43 s | 48.24% | 4.27% | 4.27% | 9.55% |
| 200 / 5 | 108.01 s | 73.80% | 11.20% | 10.92% | 21.49% |
| 100 / 5 | 53.50 s | 87.02% | 24.43% | 23.87% | 37.11% |

The 400-step arm raises raw relative Frobenius error by 1.91% on average.
Its weighted NMSE increases by 2.18–6.77% across individual pairs. Means by
owner type are up 2.80%, gate 3.39%, down 3.64%, QKV 5.62%, attention output
5.91%. Its worst activation regression is block 0 attention output, seed 0.
Thus halving the outer count is a measurable speed/fit tradeoff, rather than
an equivalent result. The shorter 100/200 schedules lose substantially more.

Evidence: `outputs/admm-budgets/screen/{protocol,results,summary}.json` and
the saved validation token/input tensors. The first screen predates the tool's
explicit `--arms` interface and summary addition; its fitting/scoring path is
the same. Its protocol records the original tool hash.

## Inner-iteration and combined results

A second series completed 90 fits with an interleaved fresh 800x5 control for
every owner and seed. All 30 repeated control metric records reproduced the
first series exactly. Use this series' own timing control: the laptop GPU reached
87 C during the run and timing differed between series. Reversed arm ordering
reduces ordering bias, but these timings remain hardware/thermal-state dependent.

| Outer / inner | Total fit time | Time saved vs matched 800x5 | Mean weighted NMSE increase | Mean activation NMSE increase | Worst activation increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800 / 5 | 447.61 s | — | — | — | — |
| 800 / 2 | 399.23 s | 10.81% | 0.133% | 0.012% | 4.05% |
| 600 / 2 | 303.64 s | 32.16% | 1.552% | 1.489% | 5.49% |

At 800x2 the median weighted increase is only 0.020%, but block 0 QKV, seed 0,
regresses by 3.36%. Its largest activation regression is block 12 attention output,
seed 1. Small average differences therefore do not establish equivalence for
every projection.

At 600x2, weighted NMSE increases by 0.77–2.74% across all pairs, and raw relative
Frobenius error increases by 0.70% on average. The worst activation increase is
block 0 attention output, seed 1. Activation changes range from -2.79% to +5.49%.
Error need not improve monotonically when changing inner projection accuracy.

Evidence: `outputs/admm-budgets/inner-screen/{protocol,results,summary}.json`.
Across both series, 210 ordinary timed fits were measured; the 30 common 800x5
controls were repeated, not pooled to manufacture extra independent samples.

## ADMM phase profile

A separate instrumented replay of block 0 gate, seed 0, covers two rank-1152
stages at 800 outer iterations each. In the 800x5 control, CUDA-event intervals
over all 1,600 outer iterations break down as follows:

| Phase | CUDA interval | Share of outer-loop interval |
| --- | ---: | ---: |
| Left and right linear solves combined | 14.222 s | 84.4% |
| Sign projections combined | 2.133 s | 12.7% |
| Dual updates | 0.447 s | 2.7% |
| Entire outer loop | 16.852 s | 100% |

These are event intervals, including launch gaps/waits, rather than sums of
kernel-active time. They explain why reducing inner projection iterations
delivers a modest overall gain: the repeated Gram construction, Cholesky/linear
solves and associated synchronization dominate. Splitting those solve operations
further is the next kernel-level profiling target if an unchanged iteration
count is required. This study does not change solver failure handling or defer
Cholesky status checks.

Both instrumented 800x5 and 800x2 replays reproduced their ordinary-run metrics
exactly. Instrumented wall times are excluded from the speed comparison above.
Evidence: `outputs/admm-budgets/phases/*.phases.json`.

## Recommendation

Use **600 outer / 2 inner** as the next full compression-quality candidate. It
offers about one-third less fitting time for a small, measured local-error cost.
Use **800 outer / 2 inner** when preserving the average fit more closely matters
more than the additional speed. The 400x5 option nearly halves fitting time but
has a clearly larger error penalty.

Both controls are already exposed by the existing configuration. For the
600x2 candidate, the configuration fields are:

```yaml
factorization:
  admm:
    outer_iterations: 600
    inner_iterations: 2
```

Keep all ranks, calibration data, scale fitting, tuning batches/updates and final
quality protocols fixed in the follow-up comparison. Check final teacher KL,
perplexity and downstream quality, especially sensitivity of attention output
and QKV. Production defaults remain 800x5 for the 070 recipe pending that gate.
The measured percentage savings apply to fitting, not to the entire campaign.

## Reproduce

Use an idle CUDA device; the tool acquires the project cross-process lease.
Output directories must be new, and the source run is read-only.

```powershell
.\.venv\Scripts\python.exe tools\profile_admm_budgets.py `
  --run-output evidence\070\070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it `
  --snapshot C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752 `
  --output outputs\admm-budgets\outer-screen
```

Use `--arms 800x5 800x2 600x2` for the matched inner-iteration comparison, or
`--blocks 0 --owners mlp.gate_proj --seeds 0 --arms 800x5 --profile` for a separate
phase profile. Protocol JSON records model/token/tool hashes, source identity,
parameters and workload. Results are saved atomically after each fit. Summary
JSON requires complete matching owner/seed sets against the 800x5 baseline.

## Interpretation limits

This is a pre-tuning fit diagnostic. It does not establish final perplexity,
teacher KL, downstream task accuracy, or full campaign speedup. A promising
setting must still pass a matched compression/tuning and model-quality comparison
before becoming the default. Existing configuration already exposes both
iteration counts; this investigation does not change production defaults or
the resident algorithm version.

## Validation

The full suite passed: 1,588 tests, 53 deselected. A subsequent strengthened
focused check passed all three cases, comparing the probe directly with the
production residual-stage service in FP32 and BF16 and checking matched-pair
summary accounting. Ruff passed repository-wide; mypy passed for all 224 checked
source files. The saved CUDA comparisons and instrumented metric equality above
provide the real-workload evidence beyond those small regression fixtures.
