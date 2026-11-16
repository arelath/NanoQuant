# Experiment 080: reusable three-stage rank responses

## Purpose and coverage

Measure rank versus reconstruction error for every physical quantization unit in
the pinned Gemma 3 1B model, using the current 080 IMatrix objectives and independent
residual stages. Embeddings are never loaded for fitting or modified.

Dataset directory: `Datasets/gemma-3-1b-it-staged-rank-response-080-v1`.
The existing `Datasets/gemma-3-1b-it-allocation-070` is preserved.

The old 070 dataset already contains independent residual stages. It has only
three measured ranks per unit and a two-segment fitted curve. MLP measurements
reach 2.5 full-stage equivalents; QKV and attention output stop earlier. It does
not provide uniform coverage through three full stages, nor per-member QKV error
at every rank. Its probe recipe also differs from 080's IMatrix recipe.

The new dataset covers 26 blocks × five physical units: MLP gate, up and down,
attention output, and shared QKV. This is 130 units representing 182 projections.
For each unit let `D = floor(min(rows, columns) / 32) * 32`, using the stacked QKV
shape when appropriate. Requested total-rank fractions are:

```text
0.25, 0.5, 0.75, 0.875, 1,
1.125, 1.25, 1.5, 1.75, 1.875, 2,
2.125, 2.25, 2.5, 2.75, 2.875, 3
```

Ranks round to the nearest multiple of 32 using Python's ties-to-even rule. Save
the actual rank and actual fraction; requested fractions need not be exactly
representable. All five shapes yield 17 distinct points, including exact one-,
two- and three-full-stage boundaries: **2,210 measured options**. Each point saves
per-member errors, giving **3,094 logical-projection observations**.

Each total rank is split into full stages followed by a partial final stage,
e.g. `[1152, 1152, 288]`. This measures the production full-prefix/tail policy;
it is not an exhaustive search over arbitrary ways to divide rank among stages.

## Fitting and metrics

Use original pinned source weights and the validated 080 run's committed input
and output importance. Shared QKV uses the persisted member objective multipliers,
including V's multiplier of two. The source has no outlier or low-rank-patch
sidepaths; the builder rejects nonempty sidepaths rather than dropping them.

Match 080's actual ADMM settings: 600 outer iterations, two inner iterations,
cubic penalty, regularization 0.03, no early stopping and no wide transpose.
Fit scales with two alternating passes and production rollback behavior. Inputs
are BF16 with production CUDA numerical settings. Unit/stage seeds are fixed and
recorded. Each tail rank gets its own fit; the dataset does not truncate 080's
tuned export. Identical full-stage prefixes are reused under the same target,
objective and seed, which avoids recomputing identical fits.

Later stages fit the residual of the FP32 reconstruction of optimized prior
scales. Separately reconstruct every fitted stage with BF16-rounded scale vectors
and sum in FP32 to score the stored representation. Save both optimized-precision
and stored-representation metrics. This follows the pre-freeze fitting/final-scale
storage distinction; it does not include activation tuning, block refitting or KD.

For target `W`, stored approximation `Q`, input importance `v`, output importance
`u`, and `v_mean_one = v / mean(v)`, save:

- `raw_nmse = sum((Q-W)^2) / sum(W^2)`.
- `objective_nmse = sum((Q-W)^2 * u * v) / sum(W^2 * u * v)`.
- `imatrix_noise_nmse = sum((Q-W)^2 * v_mean_one) / sum(W^2)`.

The square root of the third metric is the column-weighted noise-equivalent
relative error used by the mixed-format sensitivity analysis. It differs from
weighted NMSE because the denominator is the unweighted source norm. Save the
underlying sums and individual Q/K/V metrics, not just an aggregate group score.

`payload_bits` sums the production `factor_bit_cost` for all stages with 16-bit
scales and rank alignment 32. Count QKV's shared right factor and input scales
once. This is exact logical packed factor/scale payload cost, not whole-GGUF
size: metadata, container overhead, embeddings and other fixed tensors are outside
the per-unit cost. Adding a stage charges its new outer scales.

## Curves and reuse

Retain every measured point. For each metric and logical member, fit equal-weight
decreasing isotonic regression to log NMSE, then interpolate linearly between
the fitted log-error knots. Thus the curve is piecewise exponential in error and
preserves every sampled rank boundary. Raw measurements remain unchanged even
where sampling/fitting noise violates monotonicity. In-sample log RMSE quantifies
the isotonic adjustment, not predictive accuracy on independent data.

Publish the existing content-addressed `allocation-profile-dataset` schema only
after all units are complete. Standard `evidence.response_points` contains all
17 measurements, and `evidence.response_curve` contains all 16 intervals; the new
builder does not call the old min/baseline/max two-slope fitter. Baseline rank is
two full stages. `staged_measurements` adds explicit ranks, costs, per-member
metrics, seed/protocol hashes and all three fitted curves. The compact final
artifact contains the scalar evidence needed for reuse after the source run is
retired. Per-unit measurement artifacts and their protocol are also retained.

Use `load_allocation_profile_dataset` to validate and read the dataset. For example:

```python
from nanoquant.infrastructure.allocation_profile_dataset import load_allocation_profile_dataset

dataset = load_allocation_profile_dataset("Datasets/gemma-3-1b-it-staged-rank-response-080-v1")
unit = next(u for u in dataset.payload["units"] if u["evidence"]["unit_id"] == "0:mlp.gate_proj")
points = unit["staged_measurements"]["points"]
curve = unit["staged_measurements"]["curves"]["imatrix_noise_nmse"]
```

`curve_error` in the builder interpolates these curves and rejects extrapolation.
The existing resident approximate-dataset consumer can use the saved multi-knot
allocation curve, but its own endpoint extrapolation remains possible and must
be checked against the recorded measured range. The measured range is 0.25–3D.

No old KL anchors are silently copied. This dataset measures reconstruction error,
not per-rank KL; the default artifact has no `kl_profile`. Consequently the D2
convenience launcher that requires a packaged KL profile will not accept it alone.
A separately measured compatible profile can be supplied with `--kl-profile`
when building a distinct dataset identity. For mixed-format planning, combine the
per-member column-weighted errors with the separately versioned sensitivity
measurements, then validate conditional and joint model quality. The recent refit
regression remains evidence that lower local error is not a model-quality gate.

## Checkpointing and reproduction

`measurement-manifest.json` binds the model file, source manifest/journal, audit,
recipe, code, numerical library versions and sample grid. Each rank result is
written atomically with a content checksum under `measurements/`. The model file
hash must match the committed source-weight hashes. Transient full-prefix tensors under
`work/` support resuming the current unit and are removed after its completion;
completed scalar results remain. Output and GPU leases prevent concurrent writers.
An invocation with different semantic identity rejects the existing directory.

```powershell
.\.venv\Scripts\python.exe tools/build_staged_rank_dataset.py `
  --run-output evidence/080/080-d2-uniform-control-gemma-3-1b-it `
  --model C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752/model.safetensors `
  --audit outputs/080/080-mixed-refit-resident-audit.json `
  --output Datasets/gemma-3-1b-it-staged-rank-response-080-v1
```

The initial invocation used `--max-units 1` for a complete 17-point feasibility
test under `Datasets/gemma-3-1b-it-staged-rank-response-080`. That evidence is
preserved separately. Source-file identity and per-unit checksum checks were then
strengthened, so the final `-v1` run starts with its own code identity and repeats
the first unit. The numerical fitting recipe is unchanged. The canary took 68.7
seconds of measured fitting/scoring. Block-0 gate objective NMSE was 0.0848671,
0.0221013 and 0.00582447 at one, two and three full stages respectively. These are
fresh pre-tuning reconstruction results, not claims about a compressed model.

## Software validation

Seven focused CPU tests pass: stage-boundary coverage, isotonic curve/allocator
agreement at interior ranks, weighted metric semantics, real ADMM and BF16-scale
reconstruction, checkpoint resume and corruption rejection, source-file identity,
and portable publication through the existing allocation dataset loader.
Ruff and mypy pass. The full default suite reports 1,877 passed, 54 deselected
and the same three existing Experiment 079 launcher/identity failures. Full log:
`outputs/080/staged-rank-response-pytest.log`.
