# Next Quality Levers for Binary Factorization

**Date:** 2026-07-19
**Status:** Idea catalog — everything here is UNTESTED unless it cites a measurement.
**Prerequisites:** [ReconstructionHeadroom.md](ReconstructionHeadroom.md) (per-layer fit is within ~0.3% of
practical optimum; falsified-strategy list), [StackedFactorization.md](StackedFactorization.md) (stacked
q/k/v and rank allocation, verified and adopted).

> **2026-07-19 update:** [ErrorAnatomy.md](ErrorAnatomy.md) measured where the error actually
> lives (within-layer + end-to-end) and re-ranked this catalog. Headlines: §2 upgraded to
> KL-measured allocation weights (bits must flow attention→MLP-up and deep→early); §7
> answered ("large gap"); §8 and §10 promoted (o_proj bias/patch); §3 demoted (errors are
> sub-additive); §16 built (`error_budget.py`); §15 promoted to necessary. Read that doc
> first — its §4 execution order supersedes the one at the bottom of this file.

## Context

Two levers from the 2026-07-18 experiments were adopted and confirmed at scale:

1. **Stacked q/k/v factorization** — shared B_R/pre basis, −14.9% attention-side Frobenius error
   model-wide on Gemma-3-1B, survives activation-space evaluation (−7 to −18% functional error).
2. **Input-importance weighting** — fit `W·diag(√imp)` then `pre /= √imp`; −20 to −30% functional
   error, free via the diagonal-absorption identity. Combined with stacking: −36 to −44% functional
   error at identical bits (0.21–0.23 vs 0.33–0.40 baseline on blocks 0/12/24).

Two consequences of adoption reshape the priority list below:

- **The objective changed.** We now optimize (and must evaluate in) activation space:
  `‖X(Ŵ−W)ᵀ‖_F / ‖XWᵀ‖_F`. Frobenius and functional rankings demonstrably diverge — weighted fits
  look *worse* in plain Frobenius while being far better functionally. Every allocation decision,
  calibration curve, and falsified result derived under plain Frobenius is now suspect until
  re-checked under the weighted objective. Some previously-lost ideas may win under the new metric
  (§10); some previously-won calibrations may be misallocating bits (§2).
- **The absorption identity is a pattern, not a one-off.** Any per-row or per-column reweighting is
  free because `diag(pre)` and `diag(post)` absorb it. Ideas that reduce to "choose a better
  diagonal" (§1, §5 within-stack weighting, §12 weight exponent) cost nothing at runtime and reuse
  the entire existing pipeline.

Ideas are ordered by expected payoff per unit effort. Each entry states the mechanism, a concrete
test, expected gain, effort, and how it composes with the adopted levers.

---

## Tier 1 — finish making the adopted levers coherent (do these first)

### 1. Output-importance weighting

**Mechanism.** The mirror of the adopted input lever. Fitting under a per-row weight is exactly an
unweighted fit of `diag(√imp_out)·W`, with `post /= √imp_out` afterward — same identity, other
side. Nothing in ADMM, scale ALS, or export changes. Both sides compose:
fit `diag(√imp_out) · W · diag(√imp_in)`, then rescale both `post` and `pre`.

**Where imp_out comes from.** Candidates, cheapest first:

- For o_proj and down_proj (rows write into the residual stream): the input importance E[x²] of
  the *readers* of that stream — the next block's LayerNorm-scaled channel variances. Already
  collected by the activation harness.
- For q/k/v within the stack: structural sensitivity (see §5).
- General: E[grad²] per output channel from a short calibration backprop (a few dozen sequences,
  teacher-forced LM loss). More faithful, slightly more machinery.

**Test.** Extend `activation_eval.py` with `sep-w-io` / `stack-w-io` arms using next-block reader
importance for o/down and gradient-based importance elsewhere; evaluate all three metrics on
blocks 0/12/24. **Success criterion:** activation-space error improves over input-only weighting.

**Expected gain:** unknown, but the input side alone bought 20–30%; the output side targets
matrices (o_proj, down_proj) the input lever helps least. **Effort:** low (a day).
**Risk:** row importance may be flatter than column importance (residual stream is normed), in
which case the gain is small — that itself is worth knowing and documents why.

### 2. Re-derive rank allocation under the weighted objective

**Mechanism.** The current water-filling allocator minimizes plain Frobenius error — a metric we
have now shown ranks fits *incorrectly*. The allocator's known pathology (draining down_proj to
E≈0.75 at the 0.5× bound) is likely an artifact: down_proj sits behind the gate/up activations
whose channel variances span 500–8000×, so its *functional* sensitivity is badly represented by
unweighted Frobenius mass.

**What changes.** Only the calibration inputs: per-matrix `E_u` and `β` measured in activation
space on *weighted* fits, and per-matrix Frobenius mass replaced by activation mass
`‖XWᵀ‖²`. The greedy heap water-filling, piecewise-β handling, and clamps in `wide_bounds.py`
are reused as-is.

**Test.** Recalibrate E_u/β for the 5 unit types on 3 blocks (15 ADMM runs at 2 ranks each),
re-run the allocator, validate on blocks 0/12/24 in activation space against the current
composite. **Success criterion:** weighted-composite functional error ≤ current composite, and
down_proj no longer pinned at its lower bound (or, if it still is, we now trust that).

**Expected gain:** unknown in magnitude but this is a *correctness* item — the ~9% composite
claim is not adoption-ready until the allocation is re-derived under the objective we actually
care about. **Effort:** low-medium; all machinery exists. **Composes:** this IS the composition.

### 3. Sequential calibration with error feedback (GPTQ-style propagation)

**Mechanism.** Today every layer is calibrated against activations from the *clean* model. Errors
therefore compound multiplicatively down the depth. If instead blocks are quantized in order and
calibration activations for block i+1 are collected by running the already-quantized prefix
[0..i], each layer's weighted fit sees the *actual* distribution it will receive at inference and
partially compensates upstream error. This is standard PTQ practice (GPTQ, BRECQ lineage)
precisely because it is cheap and reliably positive.

**What changes.** Only the calibration loop ordering in the pipeline — quantize block, splice it
into the model, re-run the forward hook for the next block. No format change, no extra bits.
~26 forward passes of a 1B model over the calibration set instead of 1; minutes, not hours.

**Test.** Two-arm comparison on Gemma-3-1B, stacked+weighted format: (a) clean-activation
calibration, (b) propagated-activation calibration. Compare end-to-end perplexity (this test
wants §16's harness) or, cheaply, final-block activation error vs the clean model.
**Expected gain:** in the PTQ literature this is worth more at aggressive bit widths; at ~1 bpw
with 20–40% per-layer error, compounding is severe, so plausibly several percent of functional
error. **Effort:** low-medium. **Composes:** with everything; orthogonal to format.

### 4. Importance-vector robustness (validation, not a lever)

**Mechanism.** The adopted weighting lever is only as good as the E[x²] estimates. Before scaling
adoption, verify the importance vectors are *stable*: across calibration datasets (wikitext vs
C4-ish text), across sample counts (1k vs 4k vs 16k tokens), and across sequence positions.
Channel variances spanning 500–8000× suggest a few dominant channels — if those are dataset-
dependent, weighted fits overfit the calibration set.

**Test.** Collect imp vectors under 3 datasets × 3 sample sizes; report pairwise cosine/rank
correlation per layer; refit one block with each and cross-evaluate activation error on held-out
text. **Success criterion:** cross-dataset functional error within a few percent of same-dataset.
**Effort:** trivial (the harness exists). **Risk mitigated:** silent quality regression on
out-of-domain text — the failure mode importance weighting is most likely to have.

---

## Tier 2 — new objectives (cheap tests, potentially large)

### 5. Structural output weighting inside the q/k/v stack

**Mechanism.** The stacked fit currently treats all 1536 output rows equally, but the three
members have different *functional* sensitivity: q and k errors pass through a softmax
(compressive — logit errors partially wash out), while v errors propagate linearly into the
attention output. The literature consistently finds v (and o) more sensitive than q/k. A per-
member row weight (scalar per member, or per-head) is a special case of §1 and therefore free.

**Test.** Sweep relative v-weight ∈ {1, 2, 4, 8} (q=k=1) in the stacked weighted fit; evaluate
attention-output error `‖softmax(QKᵀ/√d)V_hat − softmax(QKᵀ/√d)V‖` on real activations, not just
layer-output error. **Expected gain:** small-to-moderate, but free and permanent if positive.
**Effort:** trivial on top of §1. **New idea — not previously catalogued.**

### 6. Weight exponent tuning (imp^α, α ≤ 1)

**Mechanism.** Raw E[x²] weights with 8000× dynamic range may over-concentrate the fit on a few
channels, sacrificing everything else — classic risk of unregularized importance weighting (GPTQ
uses Hessian dampening for the same reason). A tempered weight `imp^α` with α ∈ [0.5, 1]
interpolates between unweighted (α=0) and fully weighted (α=1); the optimum is often interior.
Still free via the identity (√(imp^α) absorbs into pre).

**Test.** One-dimensional sweep α ∈ {0.5, 0.75, 1.0} on blocks 0/12/24, activation-space metric,
held-out calibration split (important: the α=1 arm will always win in-sample).
**Expected gain:** 0 to a few percent; also acts as insurance for §4's robustness concern.
**Effort:** trivial. **New idea — not previously catalogued.**

### 7. Measure the diagonal-vs-full-covariance gap (decision experiment)

**Mechanism.** Input weighting uses only the diagonal of the activation second moment. The full
objective is `‖(Ŵ−W)Lᵀ‖_F` with `L = chol(XᵀX)` — off-diagonal correlations matter when
activations are strongly correlated (they are: residual streams have large principal components).
The full-covariance fit does NOT absorb into `pre` and would need a weighted ADMM (medium build).
Before building it, measure whether the prize justifies it.

**Test (cheap, no new fitter).** For existing fits, compare three numbers per layer:
(a) achieved activation error; (b) the same fit's error lower-bounded by projecting the residual
onto X's principal subspace; (c) the floor: rank-r SVD *of W·Lᵀ* mapped back — the best any
rank-r method could do under full covariance. The gap (a)−(c) decomposed into "diagonal captured"
vs "correlation left on the table" tells us whether a weighted ADMM is worth building.
**Decision rule:** if diagonal weighting captures ≥80% of the covariance-aware headroom, close
this door and record it; otherwise §13 (weighted ADMM) gets promoted.
**Effort:** low for the measurement; the fitter itself is medium.

### 8. Bias correction

**Mechanism.** The quantization error has nonzero mean in activation space:
`b_corr = E[x]·(Ŵ−W)ᵀ` is a constant output offset that can be folded into a bias vector
(Gemma linears are bias-free, but adding one fp16 vector per layer is negligible bits — for
q_proj, 1024×16 bits against a 1.2M-bit budget, ~1.4%; for down 6912-input it's the output dim
1152 that counts). Closed-form from statistics the harness already collects; zero interaction
with the fit itself.

**Test.** Compute `E[x]` per layer, apply the correction, measure activation-space error delta on
blocks 0/12/24. **Expected gain:** small (activations are roughly centered post-norm), but it is
one line of math and permanently free. Worth doing just to record the number.
**Effort:** trivial.

### 9. Attention-product objective for q/k (research-y)

**Mechanism.** What attention actually consumes is the bilinear form `q·kᵀ = x W_qᵀ W_k x'ᵀ`
(per head, with RoPE interleaved). Fitting W_q and W_k independently ignores that their errors
can cancel or compound in the product. A product-aware objective would fit Ŵ_q, Ŵ_k to preserve
`W_qᵀW_k` per head. RoPE makes this exact only per relative position, which is why this is
research-y rather than an engineering item.

**Test (measurement first).** Before any fitter work: measure per-head logit error
`‖X Ŵ_qᵀ Ŵ_k Xᵀ − X W_qᵀ W_k Xᵀ‖` for existing fits and check whether it is dominated by the
individual weight errors or shows systematic compounding. If errors already partially cancel
(plausible — they're independent), the ceiling here is low.
**Effort:** low for the measurement, high for the fitter. **New idea — not previously
catalogued.**

---

## Tier 3 — format probes (cheap scripts, uncertain payoff)

### 10. Retest the fp16 low-rank residual under the weighted objective

**Mechanism.** Multi-stage residual fitting LOST under plain Frobenius
(ReconstructionHeadroom.md: 0.4233/0.4326 vs 0.4164 at equal bits) — the unweighted residual is
white-ish (kurtosis ≈ 3.16), so nothing low-rank to grab. But the *weighted* residual is
concentrated in the few high-importance activation directions, which is exactly what a small
real-valued rank-4..16 correction is good at (EoRA-style). The objective change invalidates the
old falsification; it must be re-run, not assumed.

**Test.** At equal bits (fund the fp16 correction by shaving binary ranks: rank-8 fp16 on q_proj
costs ~278k bits ≈ 127 binary ranks), compare stacked+weighted vs stacked+weighted−127r+fp16-r8
in activation space. **Decision rule:** one script, one afternoon; if it loses again under the
weighted metric, mark it double-falsified and stop.

### 11. Cross-block stacking for partner-less matrices

**Mechanism.** o_proj and down_proj have no within-block partner. The stacking-win condition
(`r_stack ≪ shared input dim`) is *numerically* satisfied by stacking the same matrix type across
adjacent blocks: two stacked down_projs form 2304×6912 with r_stack ≈ 1729 ≪ 6912. The physics is
weaker than q/k/v (different blocks' MLP activations are different spaces, only statistically
similar) and runtime fusion is unnatural (two blocks must share a packed tensor). Extension: a
2-block × q/k/v six-way stack (3072×1152) pushes the same idea harder on the attention side —
but adjacent blocks' residual inputs differ by one block of computation, so basis sharing is
plausible but unproven.

**Test.** `admm_err` probe on down_proj pairs (blocks 12+13, 0+1, 24+25) and one six-way qkv
pair, equal bits, Frobenius first (cheap screen), activation-space if it survives.
**Expectation:** honest coin-flip. Treat as curiosity probe; do not build runtime support unless
the win is >5%.

### 12. Input-side-only Hadamard rotation + diagonal weighting (synthesis)

**Mechanism.** Rotation was falsified for this format, but the damage was measured to be
*output-side* — `diag(post)` exploits head-structured row magnitudes that rotation flattens
(ReconstructionHeadroom.md §9). An input-side-only Hadamard leaves `post` untouched, and its
specific virtue under the NEW objective is different from the old test: rotating X decorrelates
and Gaussianizes the activation distribution, which (a) makes the *diagonal* importance
approximation closer to optimal — directly attacking the gap §7 measures — and (b) flattens the
8000× channel-variance spikes that worry §4/§6. `pre` absorbs any per-column scale after
rotation. Cost: one Hadamard transform on the activation path per layer group (fusable into the
preceding norm in many positions).

**Test.** Only worth running AFTER §7: if §7 finds a large diagonal-vs-covariance gap, test
input-Hadamard + diagonal weighting vs plain diagonal weighting on 3 blocks, activation metric.
If §7 finds a small gap, skip permanently. **New synthesis — the old falsification does not
cover this configuration.**

### 13. Full-covariance (weighted) ADMM

**Mechanism.** The real fitter for §7's objective: ADMM where the least-squares subproblems carry
`XᵀX` (or its Cholesky) as a metric. Right-side updates become generalized least squares;
binary projections stay elementwise. Medium build, touches `factorization.py` math.
**Gate:** only if §7 shows ≥20% headroom left by diagonal weighting AND §12's cheap
transform doesn't close it.

### 14. Per-head factorization of attention matrices (expected to lose)

For the record: splitting q/k/v per head before factorization is the semantic version of column
blocks, which lost decisively (0.4398/0.4656 vs 0.4164) because rank breadth beats scale/block
freedom at 1 bpw. Stacking (the opposite direction: merge, don't split) is what wins. Listed
here so nobody re-derives it; do not test without a new mechanism argument.

---

## Tier 4 — recovery passes and infrastructure

### 15. Scale-only distillation fine-tune

**Mechanism.** Binaries are frozen (per-layer fit is ~0.3% from optimal and STE cannot improve
it), but pre/mid/post and the LayerNorm gains are continuous and tiny in count (a few hundred K
parameters model-wide). A short KL-to-teacher fine-tune on calibration text, updating only
scales+norms, is the cheapest end-to-end recovery pass and directly optimizes the thing we
actually ship: model output. Unlike per-layer STE (falsified), this optimizes a *global*
objective with *continuous* parameters only — neither failure mode of the falsified experiment
applies.

**Test.** 100–500 steps of Adam (lr ~1e-4) on wikitext, KL to the bf16 teacher, scales+norms
only; measure perplexity before/after. Requires §16. **Expected gain:** in the literature this
class of pass recovers a meaningful fraction of PTQ degradation at extreme bit widths.
**Effort:** medium (needs the quantized model runnable end-to-end in torch).

### 16. End-to-end perplexity harness (the gating item)

**Mechanism.** Not a lever — the measurement everything else is gated on. Every result so far is
Frobenius or single-layer activation error; the adoption question ("does −36–44% functional
error move perplexity?") remains open. Build: reconstruct Ŵ for all layers from a chosen recipe
(stacked+weighted+allocated), splice into the HF model as dense bf16 (no packed runtime needed
for measurement), run wikitext-2 perplexity vs the bf16 baseline and vs the old uniform recipe.

**Effort:** low-medium — reconstruction code exists in the scratchpad scripts
(`reconstruct()` in `fit_gap_experiment.py`); the splice is mechanical. **This unlocks §3, §15,
and turns every activation-space claim into a shippable one. If only one item from this document
gets done, it should be this.**

### 17. Calibration accumulator plumbing (adoption engineering)

The production path (`application/quantization_stages.py`, `domain/scale_fit.py`) already
supports `input_importance`/`output_importance` in scale ALS, but the *fit itself* now needs
importance at ADMM time (pre-scaling of W), stacked targets need fused calibration accumulators,
and the planner (`domain/planning.py`) needs stack-aware units with piecewise β. Tracked in
StackedFactorization.md §4–5; listed here only for completeness of the adoption picture.

---

## Already falsified — do not retest without an objective-change argument

From ReconstructionHeadroom.md / StackedFactorization.md, all at equal bits, plain-Frobenius
objective unless noted:

| Idea | Result | Objective-sensitive? |
|---|---|---|
| Per-layer STE/Adam on binaries | never beats ADMM at any lr | No — also fails from ADMM init |
| Basin hopping / larger flip kicks | returns to incumbent; big kicks worse | No |
| Multi-stage residual (binary stages) | 0.4233/0.4326 vs 0.4164 | **Yes → retest as §10 (fp16, weighted)** |
| Column-block splits | 0.4398/0.4656 vs 0.4164 | Unlikely (§14) |
| Sparse fp16 outlier patch | poor bits-per-error; residual is Gaussian | Partially → subsumed by §10 |
| Both-side incoherence rotation | 0.4164→0.4344 (hurts) | **Input-side-only + weighting untested → §12** |
| Rank-group pre/post scales | +0.1–0.7% vs +2–12% for same bits in rank | No |
| Ternary {−1,0,+1} at equal bits | 0.4499 vs 0.4164 | Unlikely |
| gate+up stacking | loses at 1.0 AND 0.5 bpw (basis mismatch) | Unlikely — mechanism, not budget |
| Wider allocation bounds (>1.4×) | +0.2% only — saturated | Re-check falls out of §2 anyway |

## Suggested execution order

1. **§16 perplexity harness** — everything is gated on it, and it's cheap.
2. **§1 + §2 together** (output weighting + weighted allocation) — one calibration campaign
   feeds both; makes the adopted composite honest.
3. **§4 + §6** robustness/exponent sweep — trivial, de-risks the adopted lever.
4. **§7 decision experiment** — cheap; decides the fate of §12/§13.
5. **§3 error-feedback calibration** — measured against §16's harness.
6. **§5, §8** free micro-levers, batched into any of the above runs.
7. **§10, §11** curiosity probes when a GPU hour is spare.
8. **§15** once §16 exists and the static recipe is frozen.
