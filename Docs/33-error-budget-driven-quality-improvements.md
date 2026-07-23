# Error-Budget-Driven Quality Improvements

**Status:** Torch/runtime implementation complete; operating-point campaign in progress

**Primary evidence:** [Error Anatomy](ImprovementSuggestions/ErrorAnatomy.md) (measured 2026-07-19),
[Next Quality Levers](ImprovementSuggestions/NextQualityLevers.md) (idea catalog)

**Companion designs:** [Reconstruction-Informed Rank Planning](30-reconstruction-informed-rank-planning.md),
[Stacked Shared-Input Factorization](31-stacked-shared-input-factorization.md)

**Scope:** Turn the measured error anatomy of the adopted recipe (stacked Q/K/V + activation-importance
weighting on pinned `google/gemma-3-1b-it`) into five concrete pipeline changes: a KL splice-evaluation
harness as a first-class planning input, KL-calibrated allocation sensitivities, closed-form output bias
correction, member-weighted group objectives, and a targeted low-rank fp16 patch for `o_proj`. Scale-only
distillation is scheduled, not built — it already exists.

## Implementation and evidence status

- D1 through D5 are implemented in the application, resident-compression, artifact, packed-runtime,
  and validation paths. Bias and low-rank patch sidecars remain intentionally unsupported by the
  stock llama.cpp/GGUF path, as described in §§4.3 and 6.3.
- Experiments 021 and 022 exercise the self-measured, exact-unit D2 workflow without imported response
  constants or rank plans. Experiment 023 measures the interaction-corrected D2 variant.
- The resumable 270M phase campaign is driven by `tools/run_error_budget_campaign.py`. It now enforces
  the corrected D2 contract: exact physical-unit arms, current-run measured responses, untempered KL,
  no type/block fallback, no legacy KL/error proxy, and no rank-trust dependency on Experiment 016.
  Cross-model rank direction remains diagnostic only.
- The remaining work is measured adoption: run the paired D2 gate, then the D3 bias comparison, D4
  member-multiplier sweep, D5 patch-rank sweep, existing scale-only distillation, packed quality, and
  CUDA sidecar parity. The campaign checkpoints every completed run/profile and fails closed at each
  identity, bit-budget, and quality gate.
- Bias/patch campaign arms reserve 0.01% of the nominal factor budget. This is an alignment guard, not
  extra capacity: it prevents a charged sidecar from consuming a sub-rank remainder and ending a few
  thousand bits above the retained baseline after ranks are rounded to the required multiple.
- The 270M D2 arm passed its paired 48-sequence gate (2.81752 → 2.59082 nats/token; −8.05%, paired
  95% interval [−9.07%, −7.03%]). The first all-unit D3 arm was correctly funded at 1.024821 BPW but
  failed its `o_proj` splice gate (1.65693 → 2.01585; +21.66%, paired 95% interval [+19.23%, +24.15%]).
  D3 is therefore not adopted for this model. The campaign continues D4 from the last accepted D2
  operating point; rejected phases never become inputs to later phase measurements.
- The first no-bias D4 sweep appeared to select `α_v=2`: QKV splice KL moved from 1.15965 at
  `α_v=1` to 0.94728 (−18.31%, paired 95% interval [−22.14%, −14.65%]); `α_v=4` reached 1.01596.
  Those values are preliminary only. Inspection showed that each arm regenerated non-identical
  online-Fisher statistics, so calibration noise was confounded with the multiplier response. The D4
  arms must be rerun with the exact same validated calibration/objective artifacts before selection.
- Initial D4 timing exposed a campaign performance bug: every arm spent about one hour recomputing all
  90 measured rank-response units before roughly 72 minutes of actual block compression. Rank-probe
  reuse is now a validated execution cache, paired with exact calibration/objective reuse. Calibration
  reuse validates and imports the donor's content-addressed statistics and transitive tensors but always
  builds a new arm-specific rank plan. Probe reuse imports only physical units whose full baseline unit
  plans, model/calibration identities, probe protocol, source hashes, rank, and deterministic seed
  match. D4 multiplier arms reuse all probes because their uniform baseline probe plans are unchanged;
  D5 patch arms re-probe `o_proj` only. Every mismatch fails closed and is recomputed.

## 1. Decision summary

The anatomy study measured, end-to-end on held-out text, where the adopted recipe's error actually lives:

- MLP projections carry **72%** of output KL; `up_proj` alone is 28.5% of the type budget, while all of
  Q/K/V together is 13%. The within-layer functional ranking (`up` 0.45, `v` 0.32 … `down` 0.20, `k` 0.17)
  **inverts** the Frobenius ranking the current planner is calibrated on.
- Damage is front-loaded in depth: blocks 0–10 carry 65% of total KL, blocks 18–24 carry 7.8%.
- Per-block KLs are sub-additive (sum 5.17 vs whole-model 4.67, ratio 0.90): local fixes add up, and
  splice measurements are trustworthy planning evidence.
- The weighted residual is low-rank in activation space (a rank-4 real correction removes up to 46% of
  `o_proj` error energy) and has a non-trivial mean (up to 33% of `o_proj` error energy at deep blocks).

Decisions, in dependency order:

1. **D1 — Promote the splice harness to a supported evaluation mode.** Reconstruction splicing + NLL/KL
   against the bf16 teacher becomes a resumable workflow, producing a persistent, keyed
   **sensitivity profile artifact** (per-type and per-block KL, plus per-unit KL where affordable).
2. **D2 — Allocate rank with measured KL sensitivities.** The Docs/30 planner keeps its probe pass and
   response curves but takes `ReconstructionAllocationUnit.sensitivity` from the D1 profile
   (`s_u = KL_u / E_w,u²`) instead of activation-magnitude proxies. This moves bits attention→MLP
   (especially `up`) and deep→early — both reversals of the current behavior.
3. **D3 — Closed-form output bias correction on every factorized unit.** Requires one new calibration
   statistic (per-layer input mean). The packed format, runtime, tuning, and distillation paths already
   support an additive bias; only the producer is missing.
4. **D4 — Member multipliers on group output importance.** One config knob to up-weight `v` rows inside
   the stacked Q/K/V objective (v is both worst-fit and most exposed). Pure objective change, no format
   impact.
5. **D5 — Optional low-rank fp16 patch, `o_proj` only.** A per-layer side pair `(P_left, P_right)` of
   fp16 tensors, rank 4–16, fitted in closed form against calibration activations, stored and costed like
   outliers. Format change gated behind D1 evidence per model.

Scale-only distillation (`run_global_topk_distillation` in
[global_distillation.py](../src/nanoquant/global_distillation.py) — `_selected_parameters` at line 234
already restricts training to `scale_pre/mid/post`, `outlier_values`, `bias`, and norm vectors) is the
final recovery pass, scheduled after D2+D3 land. No new training code is required; D3 matters first so
the bias exists as a trainable parameter.

Explicit non-decisions, per the anatomy findings: error-feedback (propagated) calibration is **not**
built (compounding ratio 0.90 — premise absent); sparse row/outlier extensions are **not** extended
(no row concentration); MLP-side patches are **not** planned (poor bits-per-error at measured ceilings).

## 2. D1 — KL splice-evaluation harness and sensitivity profile

### 2.1 What exists

- Session evidence scripts (`collect_all.py`, `quantize_all.py`, `anatomy.py`, `error_budget.py`)
  produced the numbers in ErrorAnatomy.md by splicing dense reconstructions into the HF model. They
  should be committed under `evidence/error-anatomy-2026-07-19/` as provenance, then productionized.
- [infrastructure/frozen_model_loader.py](../src/nanoquant/infrastructure/frozen_model_loader.py) and
  [infrastructure/live_reconstruction.py](../src/nanoquant/infrastructure/live_reconstruction.py)
  already materialize per-layer reconstructions from run artifacts.
- [application/distillation.py](../src/nanoquant/application/distillation.py) already implements
  teacher top-k logit capture (`cache_topk_teacher_targets`, line 218) with the exact
  teacher-forcing loop the harness needs.

### 2.2 New module: `application/kl_budget.py`

One workflow class (pattern: `quality_evaluation_workflow.py`) with a request naming a completed run, an
evaluation dataset slice, and an arm list:

```
KlBudgetRequest(run_ref, dataset, sequences, arms)
  arm := "full" | "type:<layer-type>" | "block:<index>" | "unit:<block>:<path>"
KlBudgetProfile(arms: dict[arm, ArmResult(nll, kl_nats_per_token, n_tokens)], baseline_nll, provenance)
```

Mechanics, copied from the validated session script:

- Load the pinned bf16 teacher once; compute and cache teacher log-probs for the KL subset. Retain
  per-sequence statistics and use a paired confidence interval for adoption decisions. The corrected
  270M D2 measurement showed that 12×512 was insufficient for a 1% gate, while 48×512 resolved that
  candidate; sample sufficiency remains an interval-based decision rather than a fixed constant.
  Chunk the KL reduction at 128 tokens as in the session script. For large-vocabulary models, use
  on-the-fly teacher evaluation when a persistent fp16 cache would consume excessive host memory.
- For each arm: copy the arm's reconstructions over `module.weight.data`, run the eval subset, restore
  from clean CPU copies. Reconstructions come from the run's committed factors via
  `live_reconstruction`, not from re-fitting.
- Checkpoint after every arm (the session run measured 33 arms in ~3 minutes on a 12 GB GPU; the full
  per-unit matrix for Gemma-1B is 130 arms ≈ 10 minutes — affordable but resumability is still required,
  matching the Docs/30 probe-pass convention).
- Persist as an artifact keyed by (model revision, recipe hash, dataset slice) so planning can reject a
  stale profile — same invalidation rule as the Docs/30 probe profile.

Also computed per unit arm while the weights are spliced (free): the unit's normalized weighted
squared reconstruction error `E²_w` from the run record, so the profile stores the pair
`(KL_u, E²_w,u)` that D2 consumes. The persisted field must be explicitly dimensionless; an absolute
weighted-error energy is not interchangeable with this quantity.

### 2.3 Placement

Domain math (KL/NLL reductions) has no new domain code — reuse
[domain/metrics.py](../src/nanoquant/domain/metrics.py) conventions. The workflow is application-layer;
model access goes through the existing `ports/model_adapter.py` / infrastructure loaders. CLI entry:
a `kl-budget` run command beside the existing evaluation commands in
[cli/run_commands.py](../src/nanoquant/cli/run_commands.py).

## 3. D2 — KL-calibrated allocation sensitivities

### 3.1 What changes

[domain/planning.py](../src/nanoquant/domain/planning.py) needs **no change**:
`ReconstructionAllocationUnit` remains the allocation primitive, but the production D2 path no longer
interprets a transferred sensitivity proxy as KL:

- Today, [application/planning.py](../src/nanoquant/application/planning.py) line 129 derives
  `sensitivity = input_summary.mean × output_summary.mean` — an activation-magnitude proxy. Finding F/G
  showed this misranks units end-to-end.
- The allocation anchor is the directly measured exact physical-unit `KL_u`; no reconstruction-error
  denominator from the profile is used.
- Every physical unit is factorized at its aligned lower bound, baseline rank, and aligned upper bound
  in the current run. These probes use the current calibration-weighted objective. Piecewise log-error
  slopes are fitted from those points, and a noisy interval that does not demonstrate improvement is
  conservatively assigned zero gain.
- The allocator uses `sensitivity_strength = 1`. Its common geometric normalization does not change
  priorities, so the optimized score is proportional to
  `KL_u × E_weighted,u(r) / E_weighted,u(r_baseline)` rather than a tempered pseudo-KL quantity.
- Exact physical-unit arms are mandatory. Type×block fallback remains available only to the historical
  sensitivity-proxy mode and cannot be used by the measured-unit-KL path.

### 3.2 Wiring

- Join the exact D1 arms to the physical rank-probe units by `unit_id`/`profile_key`. Shared QKV is one
  physical arm and one rank-response curve; logical Q/K/V measurements are diagnostics only.
- Add an allocation-strategy value `kl_calibrated` in [config/schema.py](../src/nanoquant/config/schema.py)
  beside the existing `sensitivity` / `utility_profile` strategies; selecting it without a joinable,
  fresh D1 profile is a validation error. With no command-line arguments, each self-measured D2
  experiment creates or resumes its own uniform control and exact-unit profile under its numbered
  evidence directory, then verifies both the canonical control recipe and the profile's source commit
  identity. Experiment 021 applies this workflow to Gemma 3 270M and Experiment 022 applies it to the
  pinned Gemma 3 1B model. `--kl-profile` and `--kl-control-run` remain paired overrides for selecting
  explicit same-campaign inputs.
- `response_source = measured`, `objective_mode = calibration_weighted`, exact KL granularity, no
  imported rank-trust reference, and untempered sensitivity are a single fail-closed configuration
  contract. Configured response constants from earlier experiments are not inputs to Experiments 021
  or 022.
- Experiment 021 also disables inherited architecture multipliers and protected cohorts. Rank movement
  is determined by its current exact-unit KL anchors and current per-unit response probes, subject only
  to the declared rank bounds and bit budget.
- The protected-cohort logic and floors/caps in `allocate_reconstruction_rank_budget` are unchanged.
  Cross-model rank-direction expectations are diagnostics, not adoption gates. The corrected 270M
  exact profile moved rank toward attention and improved splice KL, but its original NLL comparison
  mixed static and tuned modes; the matched static result regressed. Experiment 021 must decide the
  globally tuned outcome without treating historical rank direction as a gate.

### 3.3 Gate

Re-run the D1 harness on the re-allocated plan. Success requires the upper bound of a paired 95%
confidence interval to show at least the predeclared relative KL improvement at equal or lower bits.
Then run the exact retained packed quality protocol. A rank redistribution matching historical
cross-model expectations is reported diagnostically but is not a success condition.

## 4. D3 — Closed-form output bias correction

### 4.1 Math

For a unit with input samples X (calibration) and reconstruction Ŵ:
`b = mean(X) · (W − Ŵ)ᵀ`, added to the layer output. Removes exactly `N·‖b‖²` of activation-space
error energy — measured at 22% mean / 33% peak for `o_proj`, 12–17% for deep k/v, ~7–10% elsewhere.

### 4.2 Producer changes

- **Calibration statistic.** [application/calibration.py](../src/nanoquant/application/calibration.py)
  accumulates clipped second moments per layer input/output
  (`OnlineClippedAccumulator` / `FixedClippedAccumulator` from
  [domain/calibration_math.py](../src/nanoquant/domain/calibration_math.py)). Add a plain
  `MeanAccumulator` (running sum + count; no clipping — the mean estimate must stay unbiased) beside
  them, snapshotted/restored through the same checkpoint path
  (`OnlineAccumulatorSnapshot`, calibration.py line 113). Only *input* means are needed.
- **New stage.** `BiasCorrectionStage` in
  [application/quantization_stages.py](../src/nanoquant/application/quantization_stages.py), running
  after `ScaleFitStage` (line 347): read the fitted reconstruction (already materialized at
  scale_fit time via `reconstruct(...)`, line 378), the target weight, and the input-mean tensor;
  emit the bias vector plus before/after activation-error deltas as events. For shared-input groups the
  group bias is computed once against the stacked target and partitioned into member row slices, same
  ownership pattern as `s_post` in Docs/31 §1.
- **Bit accounting.** `out_features × scale_bits` per unit; add a `bias_bits` field to the existing
  `BitCost` breakdown in [domain/models.py](../src/nanoquant/domain/models.py) so
  `effective_bpw` (domain/planning.py line 329) stays honest. For Gemma-1B this is ~0.4% of the model
  budget — fund it globally, not per-unit.

### 4.3 Consumers (already done)

- Packed format: `PackedLayerState.bias` with `bias_storage="separate-additive-tensor"`
  ([runtime/packed.py](../src/nanoquant/runtime/packed.py) lines 103, 240) — set `spec.has_bias=True`.
- Tuning/distillation: `TrainableFactorizedLinear.bias` is already a selected parameter in
  `_selected_parameters` (global_distillation.py line 241), so the distillation pass will refine it.
- **Open consumer: GGUF.** Gemma-3 has bias-free linears in stock llama.cpp graphs. The project already
  maintains a modified reader (Docs/31 §1 requires it for shared ownership), so emitting bias tensors
  from [infrastructure/gguf_export.py](../src/nanoquant/infrastructure/gguf_export.py) and adding them
  in the modified reader is the same class of change — but until that lands, bias correction is
  torch/CUDA-runtime only. The stage must therefore be recipe-gated, and
  [runtime/validation.py](../src/nanoquant/runtime/validation.py) parity checks must compare against
  the bias-inclusive reference.

## 5. D4 — Member multipliers on group output importance

Docs/31 §1 already specifies "concatenated Q/K/V output-importance vectors" as the group objective, and
`factorize_admm` applies output importance as row whitening
([domain/factorization.py](../src/nanoquant/domain/factorization.py) line 191). Up-weighting `v` is
therefore one multiplication before the concatenated vector is stored:
`imp_out[v_rows] *= α_v²` (squared because whitening takes the square root).

- Config: per-member multiplier map on the group topology entry (the Docs/30/31 topology config in
  [config/schema.py](../src/nanoquant/config/schema.py)), default all-ones. Normalize the concatenated
  vector to preserve its mean so the ADMM stabilizers and `shrink_importance` behavior are unaffected.
- Sweep `α_v ∈ {1, 2, 4}` (and optionally `α_q = α_k < 1`) using the D1 harness restricted to
  `type:qkv` arms — three cheap runs, decided per model family, recorded in the experiment definition
  under `experiments/recipes` (Docs/26/29 layout).
- Evidence for the ordering: `v` has the worst functional error in the stack (0.32 vs q 0.23 / k 0.17)
  and its error passes linearly to the attention output while q/k errors are softmax-compressed.

## 6. D5 — Low-rank fp16 activation-space patch (`o_proj` only)

### 6.1 Evidence and scope

Measured ceilings (ErrorAnatomy Finding B): rank-4 removes 27–46% of `o_proj` functional error energy;
`o_proj` inputs are near-low-rank (93% of error inside the top-256 of 1024 input directions). MLP-side
patches are explicitly out of scope (rank-16 ceiling ~25% of energy at ~26% of the layer's bit cost —
worse than spending the same bits on rank). Attention q/k/v stay out until D2+D3+D4 are re-measured.

### 6.2 Fit (closed form, application layer)

Given input covariance Σ = XᵀX/N with Cholesky Σ = LLᵀ (per-layer input covariance is already computed
by [application/covariance.py](../src/nanoquant/application/covariance.py) for tuning), and residual
Δ = W − Ŵ (post scale-fit, post bias):

1. M = Lᵀ Δᵀ  (in_features × out_features)
2. thin SVD M = P S Qᵀ, truncate to k
3. patch C = Q_k S_k P_kᵀ L⁻¹  — store `P_right = P_kᵀ L⁻¹` (k × in, fp16) and
   `P_left = Q_k S_k` (out × k, fp16)

C is the rank-k minimizer of ‖X(Δ − C)ᵀ‖ — the exact quantity whose ceiling was measured. Ridge-damp
L (`+λI`, λ = 1e-2·tr(Σ)/n) before inversion; the anatomy sample (4096 tokens ≫ k) makes overfit mild
but the acceptance gate is held-out regardless.

### 6.3 Representation and cost

- Artifact/runtime: a per-layer optional pair `(patch_left, patch_right)` following the outlier
  side-tensor pattern (`outlier_values`/`outlier_scales` through
  [domain/models.py](../src/nanoquant/domain/models.py), `PackedLayerState`, and the runtime layers).
  Forward: `y += (x @ patch_right.T) @ patch_left.T` — two thin GEMMs, negligible at k ≤ 16.
- Bits: add `patch_bit_cost(out, in, k, value_bits=16) = 16·k·(out+in)` beside `outlier_bit_cost`
  (domain/planning.py line 206) and route it through `ReconstructionAllocationUnit.fixed_bits`
  (line 29) so the allocator pays for the patch by shaving that unit's rank — the equal-bits
  comparison is then automatic, not manual.
- Tuning: patch tensors join `_selected_parameters` as trainable side tensors (same treatment as
  `outlier_values`).
- GGUF: same status as bias (§4.3) — modified-reader work; torch/CUDA runtime first.

### 6.4 Gate

Per model: D1 harness with arms {no patch, k=4, k=8, k=16} on `type:o`; adopt the smallest k whose
held-out KL gain survives, else drop the patch. Expected from ceilings: k=8 at ~1.3% of `o_proj` bits
for roughly 15–25% of its functional error.

## 7. Rollout, ordering, and measurement discipline

| Phase | Items | Format impact | Gate |
|---|---|---|---|
| 1 | D1 harness; commit evidence scripts | none | reproduces ErrorAnatomy §2 numbers from run artifacts |
| 2 | D2 measured-unit-KL allocation | none | paired whole-model KL improves at equal/lower effective BPW; re-measure profile |
| 3 | D3 bias + D4 v-weighting | packed bias flag only | additive KL gains on `type:o`/`type:qkv` arms; parity vs reference runtime |
| 4 | D5 o_proj patch | side tensors | §6.4 |
| 5 | scale-only distillation (existing) | none | end-to-end ppl vs Phase-4 static recipe |

Measurement discipline, from the anatomy findings:

- **Never rank by plain Frobenius again** — report the run's weighted error and, for adoption
  decisions, D1 KL. Frobenius inverted the true ranking.
- **Re-measure the profile after every phase.** Sensitivities are operating-point-dependent; the
  harness is minutes, stale profiles fail closed (§3.2).
- **Importance robustness before scaling beyond wikitext** (NextQualityLevers §4): the profile and the
  importance vectors must be reproduced on a second calibration corpus before non-Gemma adoption.

## 8. Risks

- **KL response is still an operating-point model.** The corrected path removes the cross-run error
  denominator and transferred Frobenius slopes, but assumes local KL scales with the newly measured
  calibration-weighted response. The paired whole-model KL and packed-quality gates remain mandatory.
- **Dense-splice harness ≠ packed runtime.** D1 splices bf16 reconstructions; packed-runtime rounding
  (`runtime/packed.py` dtype rules) is not in the loop. Planning evidence tolerates this; phase gates
  3–5 must additionally run [infrastructure/packed_evaluation.py](../src/nanoquant/infrastructure/packed_evaluation.py).
- **GGUF/llama.cpp lag** for bias and patch tensors confines D3/D5 benefits to the torch/CUDA runtime
  until the modified reader lands. Ship them recipe-gated.
- **Teacher log-prob cache** for the KL subset is ~3.2 GB CPU for Gemma-1B and scales with vocab and
  subset size; for larger models fall back to on-the-fly teacher passes (2× forwards, as in the session
  script) — the workflow must support both.
- **Absolute quality remains far from usable at strict 1 bpw** (ppl 53 → 7262 with all current levers).
  This plan closes measured, attributable gaps; it does not promise a usable 1-bpw model. The honest
  summary metric for every phase is nats/token against the same held-out slice.
