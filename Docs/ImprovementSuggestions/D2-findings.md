# D2 Campaign Review — Findings (2026-07-20)

**Scope:** Post-hoc audit of the Experiment 020 D2 (KL-calibrated allocation) arms on
Gemma-3-270M, answering three questions: did we do anything wrong, do we have enough KL
samples, and is there a bug in the code. Reviewed: `src/nanoquant/application/kl_budget.py`,
`src/nanoquant/infrastructure/kl_splice.py`, `src/nanoquant/domain/metrics.py`, the persisted
profiles under `evidence/020/016-kl-budget-profile*/`, and the campaign record in
`evidence/020/README.md`.

**Verdict in one line:** the exact-unit D2 arm failed primarily because of a sensitivity-metric
bug (fixable in one line), the adoption gate is under-sampled at 12 sequences, and the
remaining shortfall is expected-headroom overstatement from porting 1B anatomy to an
already-allocated 270M baseline. The campaign's gate discipline itself worked: every bad arm
was rejected before adoption.

---

## Finding 1 — BUG: sensitivity uses the absolute squared error, squared again

**Where:** `src/nanoquant/application/kl_budget.py:261`

```python
sensitivity = exact.kl_nats_per_token / (exact.weighted_error**2)
```

`weighted_error` is populated from `final_reconstruction.export_weighted_error`
(`src/nanoquant/infrastructure/kl_splice.py:96` and the group variants at lines 120/230/263).
Two compounding problems:

1. **Wrong metric.** `export_weighted_error` is the *absolute* weighted squared error —
   `weighted_squared_error()` in `src/nanoquant/domain/metrics.py:25` returns an unnormalized
   sum of squares. Doc 33 §3.1 specifies `s_u = KL_u / E_w,u²` with `E_w` the *normalized*
   (relative) weighted error; the codebase field for that is `export_weighted_normalized_error`
   (`metrics.py:89`), which is already a squared ratio (`normalized()` at `metrics.py:54` is a
   plain quotient of squared quantities — no square root).
2. **Double squaring.** Because the stored field is already squared, the additional `**2`
   makes the computed quantity `KL / E⁴_abs` where the design wanted `KL / E²_rel`.

**Evidence from the persisted profile** (`016-kl-budget-profile-v2/kl-budget-profile.json`):

- Unit-arm `weighted_error` spans **0.32 → 37.3** across the 90 units. A normalized relative
  error cannot materially exceed 1; a 100× spread is only possible for an absolute quantity
  whose scale tracks matrix size and raw importance magnitude. After the extra squaring, the
  cross-unit sensitivity distortion reaches ~10⁶×.
- The six highest computed sensitivities are all **late-block small-error units**
  (`unit:12/13/14/16:self_attn.o_proj`, `unit:17:mlp.gate_proj`, `unit:17:mlp.up_proj`),
  while every `mlp.down_proj` and `attn_qkv` unit — the largest absolute errors — computes
  to s ≈ 0.
- The measured per-unit **KLs themselves look sane** (min 0.018, median 0.079, max 0.370;
  the worst units sit in blocks 3 and 11, consistent with the independent block arms). The
  input data is fine; only the conversion is broken.

**Consequence:** this is the primary cause of the exact-unit arm's failure (full KL
3.1843 vs baseline 2.7455, +15.98%; ranks moved *away* from blocks 0–10 toward 11–17). The
campaign README attributes it to the Doc 33 §8 "quadratic estimate outside its operating
regime" risk; that risk is real but secondary — the allocator was fed sensitivities that were
wrong by orders of magnitude in a systematically depth-correlated direction (late-block
matrices are smaller and carry smaller importance mass, hence tiny absolute errors, hence
exploded sensitivities).

**Fix:**

```python
sensitivity = exact.kl_nats_per_token / exact.weighted_error   # weighted_error := export_weighted_normalized_error
```

i.e. populate the arm's `weighted_error` from `export_weighted_normalized_error` and divide
once (the field is already E²). Add two guards:

- profile validation rejects unit arms whose `weighted_error` is not dimensionless
  (e.g. > 2.0), so an absolute-metric regression can never reach the planner again;
- a planner-side sanity check that `s_u × E²_u` reproduces the measured `KL_u` ordering
  (rank correlation threshold) before the plan is accepted.

The exact-unit D2 arm should be re-run after this fix; nothing else about it was wrong.

## Finding 2 — Not enough KL samples for the adoption gate (fine for profiling)

Every arm was evaluated on **6,132 tokens = 12 sequences × 511 scored positions**
(`token_count` in every arm record; `kl_budget_workflow.py` defaults,
`--sequence-length 512`).

- **Adequate:** coarse type/block profiling (arm KLs 0.20–3.13 nats) and arm *ranking*.
- **Marginal:** per-unit arms — the smallest is 0.018 nats/token over 6k tokens.
- **Not adequate:** the 1%-relative full-KL adoption gate. The campaign's own data proves
  this empirically: the trust-region arm measured **+0.168%** on the 12-sequence KL harness,
  while the retained static quality benchmark on the *same candidate* showed
  **+0.105 NLL (+1.5%)** on WikiText and a 0.21 absolute BoolQ drop (0.575 → 0.365) with
  exact packed/reference parity, so no runtime confound. The small harness under-reported a
  real regression by roughly an order of magnitude. (The two evaluations also use different
  token sets — profile-slice baseline NLL 6.452 vs benchmark 7.222 — so part of the gap is
  distribution, which is itself an argument that 12 sequences is not a representative gate.)

Clean aspects verified: the KL evaluator scores the wikitext **test** split
(`quality_evaluation._wikitext_tokens`, `load_pinned_dataset_split(..., "test")`) while
calibration uses **train** — no leakage; v1 vs v2 profile repeatability on identical arms is
±0.005 nats, so evaluator determinism is not the problem; token/label alignment is correct
(511 scored positions per 512-token sequence).

**Recommendations:** 48+ sequences for any adoption-gate measurement; report a bootstrap
confidence interval over sequences next to the point estimate and gate on the CI, not the
point; keep the retained quality benchmark as the final arbiter (the campaign already does
this — it is what caught the regression).

## Finding 3 — Expected headroom was overstated for this workload (design, not code)

The type×block fallback arm executed correctly (all four intended rank movements: MLP +992,
attention −1248, blocks 0–10 +1152, blocks 11–17 −1408) and still regressed 3.2%. Two
design-level reasons, neither a bug:

- **The reference is not uniform.** ErrorAnatomy's reallocation gains were measured against
  *uniform ranks on Gemma-3-1B*. The Experiment 016 baseline is already Docs/30
  reconstruction-allocated, so much of the reallocation headroom was pre-harvested. D2's
  realistic prize on this baseline was always small.
- **The 270M error landscape differs from the 1B anatomy.** The fresh 020 profile shows
  `down_proj` as the worst type (3.13 nats; `up_proj` is 1.72), and a much flatter depth
  profile (block arms 0.20–0.85, worst is block 3) versus 1B's 65%-of-KL-in-blocks-0–10.
  Porting 1B-derived expectations to 270M overstated both the direction sharpness and the
  magnitude. The Doc 33 "re-measure the profile per model / per phase" discipline caught
  this; the lesson is to *set the gate expectation from the fresh profile*, not from the
  anatomy of a different model.

Also consistent with under-modelled response: the fallback arm's QKV type-KL regressed 32.4%
after attention lost 1,248 ranks — the exponential rank-response plus locally-quadratic KL
mapping underestimates the cost of large steps, which is exactly why the predeclared 0.25
trust region was the right correction (its arm landed at +0.17%, i.e. neutral within the
harness's resolution).

## What worked and should be kept

- Gate discipline: all three D2 arms were rejected before adoption; negative evidence is
  recorded with provenance (`formal/d2-kl-type-block-trust-025-static/…`,
  `d2-type-block-projection.json` with `adoption_evidence: false`).
- Provenance hashing (dataset fingerprint, slice hash, model revision, recipe hash) made this
  audit possible and cheap.
- The per-unit KL measurements are reusable as-is once the sensitivity conversion is fixed.

## Recommended next steps, in order

1. Fix `kl_budget.py:261` to divide `KL_u` once by `export_weighted_normalized_error`; add
   the dimensionless-error profile guard and the `s×E² ≈ KL` ordering sanity check.
2. Re-run the exact-unit D2 arm with the fix, keeping the 0.25 trust region.
3. Widen the gate evaluation to ≥48 sequences with a bootstrap CI; keep the quality benchmark
   as the final gate.
4. Re-set the D2 expected-gain gate from the fresh 270M profile (flat depth, down_proj-heavy)
   rather than the 1B anatomy; if the corrected exact-unit arm is also neutral, record D2 as
   exhausted on Docs/30-allocated baselines and move the campaign's weight to D3/D5 (bias and
   patches target error the allocator cannot reach).