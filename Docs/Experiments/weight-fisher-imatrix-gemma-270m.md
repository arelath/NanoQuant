# Gemma 3 270M per-weight Fisher versus activation IMatrix

**Recommendation: retain the current production IMatrix path.** Across two
completed screens, full weight Fisher is substantially worse at 3 bits. At
4 bits it improves observed-token NLL and beats the current row-and-column
IMatrix on KL, but does not demonstrate a KL advantage over cheaper activation
controls. The 4-bit NLL signal merits narrower research, not a general switch.

## Question and scope

Does output-aware, per-weight sensitivity improve quantization enough to justify
its collection and storage cost? This is an analysis-only full-model perturbation
screen, not a numbered compression experiment, a deployable export, or a change
to the resident algorithm. No production default changes on this evidence.

The baseline is the row-and-column IMatrix used by Experiment 075:
`Datasets/gemma-3-270m-it-imatrix-eaddario-huge-rows/imatrix.gguf`, SHA-256
`3deceafea3aadd5fb6b75e22dfa0123bb99f22e0be95816354b374e80ebcc0c9`.
Its source is `unsloth/gemma-3-270m-it`, revision
`23cf460f6bb16954176b3ddcc8d4f250501458a9`. The probe uses that same checkpoint,
rather than silently substituting the Google snapshot. Its corpus contains
6,810,112 token slots in 13,301 contexts of length 512; collection took
3,193.828 seconds. Those historical CUDA seconds cannot be compared directly
with this CPU probe's timings.

## Predeclared protocol

`tools/probe_weight_fisher_imatrix.py` freezes a protocol, hashes source weights
and IMatrix, and retains exact token IDs, sampled positions, statistic tensors,
collection timings, teacher distributions and per-sequence quality values.
Completed statistic stages can be reused only with matching protocol and hashes.

- All 126 decoder projections are quantized; embeddings, output head and norms
  remain at the same FP32 values in every arm. FP32 avoids the previously
  observed Gemma 270M FP16 overflow and BF16 numerical comparison floor.
- Calibration uses 32 seeded, nonoverlapping 128-token windows spread across
  the existing IMatrix corpus. Two distinct target positions per context yield
  64 independently squared gradients per Fisher estimator.
- The activation control observes all 4,096 context slots from those same
  windows. The existing IMatrix retains its larger corpus budget. The two
  comparisons distinguish sample-budget effects from the kind of statistic.
- The six arms are uniform, existing input columns, existing rows × columns,
  matched-sample rows × columns, observed-token empirical Fisher, and
  teacher-sampled-token model Fisher.
- Fisher is `mean_t (d log p(t|context) / dW)^2`, using a separate reverse pass
  for each target. Autograd sums a weight's contributions across contextual
  uses **before** squaring; targets are averaged **after** squaring. No
  sequence-averaged-gradient shortcut or layer-local outer-product substitute
  is used. The teacher-sampled target distribution makes the latter estimator
  a Monte Carlo diagonal Fisher for the model's predictive distribution.
- Every arm uses exactly the same signed uniform 3-bit and 4-bit quantizer,
  8×8 tiles, one representable FP16 scale per tile, and 31 scale ratios from
  0.5 to 1.25. Selection minimizes `sum(I * (W_quantized - W)^2)`. Tiles span
  both axes so row weights cannot cancel as they would within a single row.
  Nominal projection payload is 3.25 or 4.25 bits/weight including scales.
  These are logical code budgets, **not measured GGUF sizes or whole-model BPW**.
- Evaluation uses 24 WikiText-2 test windows of length 128, with 32 seeded
  positions in each: 768 next-token targets. KL is exact across the full
  vocabulary at each selected position, and NLL uses the actual next token.
  This test prefix is an already-retired exploratory monitor; it is not
  relabeled as a fresh gate. It is separate from the calibration source,
  but corpus-level textual deduplication has not been established.
- Paired 95% percentile bootstrap intervals resample the 24 sequences 5,000
  times. These are exploratory intervals without multiple-comparison
  correction. A ≥5% KL gain motivates further study; it does not authorize
  production adoption. No quantizer setting is fitted to evaluation outcomes.

The probe runs on CPU with four threads because Experiment 075 owns CUDA.
This preserves the active run and provides same-device collection costs for
the matched activation/Fisher comparison. Background workloads and the small
sample count limit timing precision. It does not establish GPU throughput.

## Reproduction

```powershell
.\.venv\Scripts\python.exe tools/probe_weight_fisher_imatrix.py
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_weight_fisher_imatrix_probe.py
```

Default evidence destination:
`evidence/probes/gemma270m-weight-fisher-imatrix/`.
The runner refuses incompatible reuse rather than overwriting a protocol.
Partial Fisher stages restart; completed activation/Fisher tensors are audited
and reused. Teacher scoring and quantization/evaluation are deterministic
recomputations when the command is rerun.

## Interpretation limits

This quantizer exposes arbitrary per-weight importance without changing
NanoQuant's separable row/column objective or factorization contracts. A gain
here would establish a useful screening signal, not that the current binary
factor solver can exploit the full matrix. Before adopting it, compare larger
calibration samples/seeds, longer contexts, a feasible factorized or blocked
statistic, actual NanoQuant fitting at equal payload, and a fresh untouched
evaluation slice. Then validate the complete compression/export lifecycle.

Full diagonal Fisher omits cross-weight interactions. Observed-token Fisher
and model Fisher also target different distributions. Neither is an exact
finite-quantization loss predictor.

## Initial 64-target result

The clean FP32 model's sampled NLL was 5.069273 nats/token. Lower KL and NLL
are better. All entries below perturb the same 100,270,080 projection weights.

| Importance | 3-bit KL | 3-bit NLL | 4-bit KL | 4-bit NLL |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 6.090759 | 10.317042 | 1.172376 | 6.094473 |
| Existing IMatrix columns | 3.168474 | 7.683124 | **0.623860** | 5.365239 |
| Existing IMatrix rows × columns | 2.772509 | 7.037333 | 0.739234 | 5.432637 |
| Matched-sample activation rows × columns | **2.610292** | **6.877334** | 0.690977 | 5.324360 |
| Observed-token weight Fisher | 3.311726 | 7.502443 | 0.706150 | **5.110544** |
| Teacher-sampled weight Fisher | 3.425901 | 7.531229 | 0.676440 | 5.154769 |

Both Fisher methods lose clearly at 3 bits. At 4 bits, teacher-sampled Fisher
improves KL by 8.49% relative to the existing row-and-column IMatrix, but the
existing column-only IMatrix is better still. The 4-bit teacher-Fisher paired
KL difference versus rows × columns is -0.062794 nats/token; its exploratory
95% interval excludes zero. NLL favors observed-token Fisher at 4 bits, so
prediction agreement and observed-token likelihood do not rank all arms alike.

Matched activation collection took 5.968 seconds; empirical Fisher took
33.384 seconds (5.59×), and model Fisher took 29.909 seconds (5.01×), for only
two sampled target backwards per context. One full Fisher safetensors file is
401,093,440 bytes versus 988,608 bytes for the current full-corpus row/column
IMatrix (405.7×). The probe's peak process memory was 4.48 GiB, including both
Fisher matrices, model, source copies and evaluation workspaces; this is not
an isolated single-estimator peak.

Because 64 targets may leave the gradient estimate noisy, the unchanged
quantizer and six arms are repeated with 128 calibration contexts and four
targets each (512 total). The larger run uses the same seed and corpus; its
evaluation positions are newly drawn after the longer calibration-position
draw sequence, within the same 24 already-retired WikiText windows. Comparisons
are paired within each run, not directly across their aggregate NLL/KL values.

## Larger 512-target result

This run completed all six arms at both widths. Its clean-model sampled NLL
was 5.201249 nats/token. The different baseline NLL reflects different sampled
evaluation positions, not a change in model weights.

| Importance | 3-bit KL | 3-bit NLL | 4-bit KL | 4-bit NLL |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 6.195003 | 10.487625 | 1.165418 | 6.253347 |
| Existing IMatrix columns | 3.201177 | 7.610350 | **0.636724** | 5.494794 |
| Existing IMatrix rows × columns | 2.851935 | 6.993601 | 0.755024 | 5.589291 |
| Matched-sample activation rows × columns | **2.694335** | **6.870793** | 0.676392 | 5.460212 |
| Observed-token weight Fisher | 3.646642 | 7.824960 | 0.688687 | 5.342509 |
| Teacher-sampled weight Fisher | 3.462511 | 7.617947 | 0.685153 | **5.273036** |

Teacher-sampled Fisher versus the existing rows × columns:

- At 3 bits, KL is **21.41% worse**, with paired difference +0.610576
  nats/token, exploratory 95% interval [+0.447884, +0.776972].
- At 4 bits, KL is **9.25% better**, with paired difference -0.069871,
  interval [-0.130642, -0.011974]. However, its KL is 7.61% higher than the
  column-only IMatrix point estimate; that paired difference's interval
  [-0.015812, +0.109490] includes zero. It also does not establish an advantage
  over matched activation rows × columns: difference +0.008761,
  interval [-0.044898, +0.060595].
- The 4-bit NLL improvement has paired intervals excluding zero on this monitor:
  -0.316256 nats/token versus existing rows × columns, interval
  [-0.400053, -0.232842]. It also improves NLL versus columns alone by
  -0.221759, interval [-0.320927, -0.121753]. This is a useful positive signal
  for likelihood-oriented follow-up, despite the inconclusive KL comparison
  with the cheaper controls.

| Collection on identical 128×128 contexts | CPU seconds | Relative to activation |
| --- | ---: | ---: |
| Activation rows and columns | 22.517 | 1.00× |
| Empirical Fisher, 512 targets | 203.725 | 9.05× |
| Model Fisher, 512 targets | 269.223 | 11.96× |

This samples four of the 127 possible next-token targets in each context.
It does not measure the cost of collecting per-token Fisher across the whole
6.81-million-token corpus. Timings are single, ordered observations on a
machine also running Experiment 075; no precise CPU/GPU throughput or full
corpus extrapolation is claimed. The larger probe peaked at 4.66 GiB process
memory. The per-estimator statistic file remains 401.09 MB, about 406× the
existing 0.99 MB IMatrix.

Increasing the gradient budget eightfold did not rescue the 3-bit failure or
produce a clear 4-bit KL advantage over activation controls. Preserve the
existing production behavior. A future trial should focus on whether the
4-bit NLL gain survives a fresh evaluation, longer contexts, and the actual
NanoQuant representation at equal exported size. It should also test whether
row/column or block approximations retain that gain without storing 100
million independent importance values. These scalar results do not justify
changing Experiment 075's row policy or the resident factorizer.

## Retained evidence and implementation checks

Small, stable receipts are in
[`Results/gemma-3-270m-it-weight-fisher-imatrix/`](../../Results/gemma-3-270m-it-weight-fisher-imatrix/):
`pilot-protocol.json`, `pilot-summary.json`, `larger-protocol.json`,
`larger-summary.json`, `paired-comparisons.json`, and `environment.json`.
They include per-sequence values, hashes, CPU/dependency versions, and
comparisons to both current and matched activation controls. Large tensors
remain in the two local evidence directories.

The CPU unit checks cover summing shared contextual uses before squaring,
keeping opposing target gradients separate, exact model-Fisher versus KL
curvature on a tiny enumerated distribution, equal-rate tile reconstruction,
weighted quantizer objective improvement, and paired bootstrap behavior.

Validation completed: four focused tests passed; the default full suite passed
with 1,793 tests and 54 expected marker deselections (252.62 seconds). Repository
Ruff passed, and mypy passed across 228 source files. CUDA-marked tests and a
complete NanoQuant compression/export comparison were not part of this screen.
