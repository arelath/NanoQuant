# Error Anatomy of the Adopted Recipe (Stacked + Weighted, ~1 bpw)

**Date:** 2026-07-19
**Model:** Gemma-3-1B (google/gemma-3-1b-it), wikitext-2 calibration (24×512 tokens),
held-out evaluation (sequences 24–47).
**Recipe under study ("adopted"):** stacked q/k/v + input-importance weighting on every
linear, uniform ranks (r = mn/(m+n), stack rank from equal bits). Baseline for
comparison ("old"): separate q/k/v, unweighted, same bits.
**Question:** where does the remaining error actually live — so the idea catalog in
[NextQualityLevers.md](NextQualityLevers.md) can be re-ranked by measured error mass
instead of plausibility.

Scripts: `collect_all.py`, `quantize_all.py`, `anatomy.py`, `error_budget.py`
(session scratchpad). Reconstructions cached per unit; all numbers below are measured,
none modeled.

## 1. Within-layer anatomy (blocks 0/12/24, real activations)

For each unit, with X = 4096 real input tokens and ΔW = Ŵ−W:
functional error `E_func = ‖XΔWᵀ‖/‖XWᵀ‖`; the fraction of functional error *energy*
removable by an optimal rank-k real correction (SVD of XΔWᵀ); the fraction of error
inside the top-256 activation directions (spectral projector of XᵀX); row
concentration; and the bias-removable fraction (mean error).

Per-type means over blocks 0/12/24, adopted recipe:

| unit | E_func | E_frob | patch r16 | patch r64 | err in top-256 in-dirs | sig in top-256 | worst-5% rows | bias frac |
|---|---|---|---|---|---|---|---|---|
| q    | 0.229 | 0.62 | 0.38 | 0.62 | 0.81 | 0.97 | 0.13 | 0.11 |
| k    | 0.172 | 0.40 | 0.42 | 0.72 | 0.81 | 0.97 | 0.08 | 0.12 |
| v    | **0.316** | 0.41 | 0.41 | 0.71 | 0.81 | 0.88 | 0.08 | 0.12 |
| o    | 0.238 | 0.52 | **0.54** | **0.76** | **0.93** | 0.98 | 0.11 | **0.22** |
| gate | 0.282 | 0.58 | 0.33 | 0.55 | 0.79 | 0.95 | 0.10 | 0.10 |
| up   | **0.449** | 0.59 | 0.33 | 0.55 | 0.79 | 0.88 | 0.09 | 0.10 |
| down | 0.199 | 0.61 | 0.25 | 0.46 | 0.63 | 0.95 | 0.07 | 0.07 |

### Finding A — the functional ranking inverts the Frobenius ranking

Functionally worst: **up_proj (0.45)** and **v_proj (0.32)**. Functionally best:
k_proj (0.17) and down_proj (0.20). Under Frobenius, up/gate/down all looked equally
bad (0.55–0.63) and q looked worst of the attention side. Consequences:

- **Any allocation derived from Frobenius calibration is mis-aimed.** The
  equal-weight allocator's instinct to drain down_proj was directionally *right*
  (down is functionally the second-best fit); its Frobenius numbers were just the
  wrong justification. Weighted-objective re-calibration (NextQualityLevers §2) is
  mandatory, and should be expected to shift bits toward up and v.
- up_proj is 1.6× functionally worse than gate_proj at identical geometry and
  *identical input activations* — the difference is purely in how each weight matrix
  aligns with the input distribution. This is a per-matrix property that only
  activation-space calibration can see.
- v is the worst-fit member of the qkv stack AND the member whose error passes
  linearly to the output (no softmax compression) — two independent reasons to
  up-weight v rows inside the stack fit (NextQualityLevers §5).

### Finding B — the weighted residual is strongly low-rank in activation space

An optimal rank-16 real-valued correction removes 25–54% of functional error energy;
rank-64 removes 46–76%. o_proj is extreme: at block 24 a **rank-4** correction removes
46% of error energy (its inputs — attention outputs — are nearly low-rank, with 98% of
signal in the top-256 of 1024 directions).

- Diagonal importance weighting is therefore leaving large *correlation* structure on
  the table: the answer to the NextQualityLevers §7 decision experiment is
  "**large gap — covariance-aware methods have real headroom**".
- The sharpest cheap exploit is a targeted low-rank fp16 patch on o_proj:
  rank-4 costs ~12% of o's bit budget for up to ~30–46% of error energy
  (~16–27% of functional error) at deep blocks. A patch on down/gate/up is *not*
  obviously worth it (r16 fraction 0.25–0.33 but 16 fp16 ranks cost ~26% of budget).
  §10 should be retargeted from "patch everything" to "patch o (and maybe q/k/v)".
- Caveat: patch ceilings are measured in-sample on the calibration activations;
  a fitted patch must be validated held-out (rank ≪ token count, so overfitting
  should be mild, but verify).

### Finding C — bias correction is real, and concentrated at o_proj

The constant-offset (mean-error) share of functional error energy: ~22% on average
for o_proj (**33% at block 24**), 12–17% for k/v at deep blocks, ~7–10% elsewhere.
A per-layer output bias computed in closed form from calibration statistics
(`b = mean(X)·ΔWᵀ`) is nearly free in bits and recovers ~3–18% of functional error
depending on unit. NextQualityLevers §8 is promoted: implement unconditionally.
(Note bias is a special case of the rank-k patch — the rank-1 direction along
mean(X) — so a fitted patch subsumes it; implement bias first, it's one line.)

### Finding D — error is NOT concentrated in rows; it IS concentrated in input directions

- Worst-5%-of-rows carry only 7–17% of error energy (uniform = 5%): no fat row tail,
  so sparse row-patching and row-outlier handling are dead ends. Output-side gains
  must come from *importance* (which rows matter downstream), not error spikes.
- 63–93% of error energy lies inside the top-256 activation input directions —
  almost as concentrated as the signal itself. The residual error lives *in* the
  subspace the model actually uses, not in the unimportant tail. This is why further
  diagonal reweighting alone cannot fix it (the weighting already prioritized these
  directions; what remains is *within-subspace* misfit that only more capacity —
  rank, patch, or covariance-aware fitting — can remove).

## 2. End-to-end error budget (held-out KL and perplexity)

Measured by splicing cached reconstructions into the live model; 24 held-out
wikitext-2 sequences for headline arms, 12 for splice arms. KL is nats/token vs the
bf16 teacher.

### Finding E — absolute quality reality check

| arm | nll | ppl | KL |
|---|---|---|---|
| bf16 baseline | 3.971 | 53.0 | 0 |
| adopted recipe, whole model | 8.891 | 7262 | 4.675 |
| old baseline (separate, unweighted) | 19.915 | 4.5e8 | 16.805 |

Two readings, both important:

- **The adopted levers are end-to-end confirmed, and they are not marginal.** The old
  recipe destroys the model outright — NLL 19.9 is *worse than a uniform distribution
  over the vocabulary* (ln 262144 ≈ 12.5), i.e. confidently-wrong predictions.
  Stacking + weighting removes 12.1 nats/token of KL. This is the difference between
  noise and a damaged-but-structured model, and it is the first perplexity-level
  validation of the activation-space methodology.
- At a strict uniform ~1 bpw with no allocation, no bias correction, and no recovery
  pass, the adopted model is still **far from usable** (ppl 53 → 7262). The per-layer levers are
real but the end-to-end bar is high: closing this gap will take the full stack of
measures (weighted allocation, bias/patch corrections, and almost certainly a
scale-only distillation pass — NextQualityLevers §15), and/or a somewhat higher bit
budget. Every decision below should be read as "which measures buy the most nats."

### Finding F — the MLP side carries 72% of the damage; up_proj is the worst single type

Quantizing one type model-wide (adopted recipe), KL in nats/token:

| type | KL | share of type-sum |
|---|---|---|
| up   | 1.520 | 28.5% |
| gate | 1.231 | 23.1% |
| down | 1.083 | 20.3% |
| o    | 0.793 | 14.9% |
| qkv  | 0.697 | 13.1% |

- **All three attention projections together cause less damage than any single MLP
  matrix.** The stacked+weighted attention side is already the cheap part; further
  attention-side refinement attacks only 13% of the budget.
- up_proj is confirmed end-to-end as the worst unit type, matching its within-layer
  functional error (Finding A). gate close behind. **Bits should flow
  attention → MLP (especially up), the opposite of what Frobenius calibration
  suggested.**

### Finding G — damage is front-loaded in depth

Per-block KL (all units of one block quantized): blocks 0–10 carry **65%** of the
total; blocks 18–24 carry **7.8%**. Worst blocks: 0 (0.520), 3 (0.474), 5 (0.394),
10 (0.349), 17 (0.336). Mid-late blocks 12–15 and 19–24 are nearly free
(0.03–0.13). Depth-uniform allocation is therefore badly wrong end-to-end: bits
should move from deep blocks to blocks 0–10. (Block 25 is an exception to the deep
trend at 0.214 — last-block effects; keep it funded.)

### Finding H — errors are ~additive across blocks: local fixes add up, error feedback demoted

Sum of per-block KLs = 5.170 vs whole-model KL = 4.675 — ratio 0.90, i.e. slightly
**sub**-additive. There is no catastrophic compounding down the depth; interactions
mildly cancel. Consequences:

- The budget decomposition above is trustworthy: a lever that removes X nats in a
  splice arm removes ≈0.9X in the full model.
- NextQualityLevers §3 (error-feedback / propagated calibration) is **demoted**: its
  premise was superlinear compounding, and at this operating point compounding is
  absent. Revisit only after total KL drops an order of magnitude.

### Finding I — KL-calibrated allocation (new, sharper form of §2)

The splice harness makes *measured end-to-end KL* available as an allocation weight:
`s_u = KL_u / E_func,u²` converts each unit's activation-space error into nats, and
the existing water-filling then minimizes Σ s_u·E_u(r)² instead of Frobenius mass.
The full per-block × per-type sensitivity matrix costs 130 splice evals ≈ 10 min on
this GPU. This subsumes both Finding F (type shares) and Finding G (depth profile)
in one allocator run and replaces proxy weights with measured ones.

## 3. Consequences for the idea catalog

Measured re-ranking of [NextQualityLevers.md](NextQualityLevers.md):

| Idea | Verdict from anatomy |
|---|---|
| §2 weighted allocation re-derivation | **Confirmed mandatory** — functional ranking inverts Frobenius; bits should flow toward up/v, away from down/k |
| §8 bias correction | **Promoted** — up to 33% of o's error energy at deep blocks, closed-form |
| §10 low-rank patch | **Promoted & retargeted** — patch o_proj (rank 4–16); skip MLP-side patches |
| §7 diagonal-vs-covariance gap | **Answered: large** — §12 (input Hadamard) and §13 (weighted ADMM) promoted to test-worthy |
| §5 v-row weighting in stack | **Strengthened** — v is worst-fit AND most exposed |
| Sparse/row-outlier ideas | **Confirmed dead** — no row concentration |
| §1 output-importance weighting | Unchanged (anatomy measures error location, not downstream sensitivity) |
| §3 error feedback | **Demoted** — compounding ratio is 0.90 (sub-additive); premise absent at this operating point |
| §2 allocation weights | **Upgraded to measured KL sensitivities** (Finding I) — bits flow attention→MLP-up and deep→early, both reversals of the Frobenius allocator |
| §15 scale-only distillation | **Promoted to necessary** — Finding E shows the static recipe alone is far from usable at strict 1 bpw |
| §16 perplexity harness | **Done** — `error_budget.py` is the harness (splice, NLL, KL, checkpointed) |

## 4. Suggested next experiments, in order

1. **KL-weighted allocation** (Finding I): measure the 130-arm sensitivity matrix,
   re-run the water-filling with `s_u` weights and the depth profile, validate
   end-to-end. This is the largest measured, fully-plumbed win available.
2. **Bias correction everywhere** (Finding C): closed-form, then re-measure the
   budget.
3. **Rank-4..16 fp16 patch on o_proj** (Finding B), held-out validated.
4. **v-row up-weighting in the stack** (Finding A/§5): free, one sweep.
5. **Scale-only distillation** (Finding E): the recovery pass the end-to-end numbers
   say is required.
