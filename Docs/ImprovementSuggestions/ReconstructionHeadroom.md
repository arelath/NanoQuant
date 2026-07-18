# Reconstruction Headroom: Is the ADMM Fit Stuck in a Local Minimum?

**Status:** Findings (experiments run 2026-07-18)

**Audience:** Algorithm researchers deciding where to spend effort on reconstruction quality

**Related:** [21-weight-reconstruction-overview.md](../21-weight-reconstruction-overview.md), [16 optimization catalog / experiment 003 quality-gap notes](../15-profiler-design.md), `src/nanoquant/domain/factorization.py`, `src/nanoquant/domain/scale_fit.py`

## 1. Summary

**The per-layer fit is not meaningfully stuck.** The exported ADMM solution is ~0.3% (relative Frobenius error) above the best solution that scale ALS, exhaustive one-bit coordinate descent, STE/Adam, and basin hopping can collectively find — and every one of those methods, from every starting point tried, lands in the same place. The large remaining gap to the information-theoretic floor (0.42 achieved vs ≈0.22 possible at 1 bpw) is a property of the `diag(post)·B_L·diag(mid)·B_R·diag(pre)` format, not of the optimizer.

**Same-budget format rearrangements tested so far all lose.** Multi-stage residual fitting and column-block splits are strictly worse at equal bits; sparse fp16 outlier patching pays poorly because the residual is almost exactly Gaussian. Rank breadth is the most valuable thing the bit budget buys.

What would actually improve reconstruction, in order of expected payoff, is in §6.

## 2. Experimental setup

All numbers are on real weights from `google/gemma-3-1b-it`, layer 12, fitting the **original (unweighted) parameters** with unit importances. Rank is set to the ~1.0 bpw budget `r = mn/(m+n)`. Errors are relative Frobenius: `‖W − Ŵ‖ / ‖W‖`.

- `q_proj`: 1024×1152, r = 542
- `down_proj`: 1152×6912, r = 987

`factorize_admm` ran with production defaults (400 outer iterations, cubic schedule, `transpose_wide=True`).

## 3. Evidence on the local-minimum question

| Probe | Result (q_proj) |
|---|---|
| ADMM as exported (E0), seeds 0/1/2 | 0.4174, 0.4175, 0.4174 |
| + scale-only ALS (`fit_scales`, unit importances) | 0.4169 |
| + greedy one-bit coordinate descent on B_L/B_R, interleaved with ALS, to convergence | 0.4163–0.4164 (all seeds) |
| Per-layer STE/Adam from ADMM init (lr ∈ {3e-4, 1e-3, 3e-3}, ALS every 200 steps) | never beats 0.4170; degrades monotonically |
| Per-layer STE/Adam from random init (6000 steps) | plateaus at 0.7427 |
| Basin hopping: flip 0.1% / 0.5% / 2% of bits, re-descend (3 trials each) | 0.4163 / 0.4162 / 0.4164 — returns to incumbent |
| Basin hopping: flip 5% / 15%, re-descend | 0.4193 / 0.4608 — strictly worse |

`down_proj` shows the same shape: 0.5556 → 0.5550 (ALS) → 0.5545 (flips).

Interpretation:

- **Technically yes, the export is not even a one-flip local minimum** — a few thousand single-bit flips (out of ~1.4M bits) plus scale refits recover ~0.3%. That is the entire recoverable amount.
- **Practically no, it is not a trap.** Three independent ADMM seeds agree to the fourth decimal; perturbations of up to 2% of bits descend back to the same error; larger kicks only lose ground; and a gradient-based optimizer with full freedom to move many bits at once finds nothing better from either warm or cold starts. The 0.4163 level is the representative attractor of this parameterization on this matrix, not one basin among many.
- The rank-1 SVID export of the scales costs almost nothing: closed-form scale ALS recovers only ~0.1%.

## 4. How far is the fit from what is possible?

Two floors, per q_proj:

| Floor | rel err | Fair? |
|---|---|---|
| Rank-542 SVD (real-valued factors) | 0.1629 | No — real factors are a ~16–32 bpw budget |
| Rate–distortion bound for **any** 1-bpw code, Gaussian-row model with q_proj's empirical spectrum (reverse water-filling) | 0.2193 | Yes, for any code; generous because it assumes ideal vector quantization + entropy coding |
| iid-Gaussian Shannon bound at 1 bpw (no structure) | 0.5000 | Reference point |

The achieved 0.4174 already beats the unstructured 0.50 bound — the factorization does exploit spectral structure — but roughly half the theoretically available fidelity at this bit rate is left on the table **by the format**. No fitter for the current format can close that; reaching toward 0.22 requires codes that exploit correlations across many weights (vector quantization / trellis / entropy-coded schemes à la QuIP#, AQLM, QTIP).

## 5. Same-budget format variants that do NOT help

All tested on q_proj at the identical `r(m+n)` bit budget, seed 0:

| Variant | rel err | vs 0.4174 |
|---|---|---|
| 2-stage residual (r=271 each; stage 2 fit on residual of stage 1) | 0.4233 | worse |
| 4-stage residual (r=135 each) | 0.4326 | worse |
| Round-robin refits of the stages (block coordinate descent, 3 sweeps) | 0.4442–0.4756 | degrades further* |
| 2 column blocks, independent fits (r=369/block) | 0.4398 | worse |
| 4 column blocks (r=225/block) | 0.4656 | worse |
| Sparse fp16 patch of top 0.5% residual entries (+0.18 bpw) | 0.4065 | small gain, poor bits-for-error trade |

*The refit degradation is itself informative: a fresh ADMM solve on a residual-structured target (one whose dominant component is another binary reconstruction) does markedly worse than on a raw weight matrix — the only setting where we observed genuine optimizer fragility. Multi-stage schemes would need warm-started refits, and even then start from behind.

The residual after fitting has kurtosis 3.16 (Gaussian = 3): the fit leaves behind white, structureless noise. There are no outliers to patch and no low-rank structure left to mine — which is exactly what a near-optimal fit of this format should leave.

Conclusion from the block/stage sweep: **shared pre/post diagonals across all ranks are not the binding constraint; rank breadth is worth more than extra scale freedom.** Any proposal that trades rank for finer-grained scales starts at a disadvantage at 1 bpw.

## 6. What would actually improve reconstruction

Ranked by expected payoff per unit effort:

1. **Spend bits non-uniformly across layers (allocation), not better fits within layers.** Per-layer relative error at fixed 1 bpw is essentially determined by the layer's spectral decay (q_proj 0.417 vs down_proj 0.556 at the same bpw). Moving rank from spectrally-easy to spectrally-hard layers — or allocating by end-loss sensitivity — changes end quality far more than the 0.3% per-layer fitting slack. This aligns with the experiment-003 finding that retry is a no-op under uniform allocation. **Measured: 8.2% lower global error at equal bits — see §8.**
2. **Optimize the objective that matters instead of the Frobenius one.** The dormant Hessian/importance-weighted objectives and the existing network-level STE distillation change *which* reconstruction is sought. The per-layer machinery (`fit_scales`, and the flip refinement below) extends to the weighted objective with the same closed forms.
3. **Format change toward correlation-exploiting codes** if reconstruction at fixed bpw is the true goal: incoherence preprocessing (random rotations) plus codebook/lattice quantization of factor groups, or entropy-coded residuals. This is the only route toward the ≈0.22 floor; it is a redesign, not a tweak, and interacts with the llama.cpp/packed-runtime export formats.
4. **Greedy sign-flip refinement stage** (cheap, bounded, already prototyped): after `fit-scales`, alternate one-bit coordinate descent on B_L/B_R with scale ALS. Closed-form flip gains via the Gram matrix: for row `i` with prediction scales `p`, `ΔE(flip k) = 4[p·b_k·c_k − p²·b_k·(Gb)_k + p²·G_kk]` where `G = MMᵀ`, `c = M·wᵢ`, `M` the scaled right factor; symmetric transpose form for B_R. Converges in a handful of sweeps, ~1 s/layer on a 12 GiB GPU, worth ~0.2–0.3% per layer. Only worth adding because it is nearly free; do not expect visible perplexity movement from it alone.
5. **Not worth pursuing** (measured dead ends): per-layer STE against W, multi-stage residual fitting at equal bits, column-block splits at equal bits, sparse outlier patching of the residual, more ADMM seeds/restarts, more ADMM iterations at this scale.

## 7. Reproduction notes

Scripts lived in the session scratchpad (`fit_gap_experiment.py`, `ste_experiment.py`, `basin_hop.py`, `residual_experiment.py`, `residual_tail.py`, `rd_bound.py`, `rank_allocation_experiment.py`, `rank_allocation_v2.py`); the methodology above is sufficient to recreate them. Key ingredients: `factorize_admm` with unit importances; `fit_scales` with `torch.ones` importances for the unweighted objective; flip descent per §6.4; RD floor by reverse water-filling over `λᵢ = σᵢ²/m` at `n` bits/row.

## 8. Measured: cross-layer rank allocation gains 8.2% at equal bits (2026-07-18)

Lever §6.1 tested end-to-end on all 182 linear matrices of Gemma-3-1B (26 blocks × 7 types), unweighted Frobenius objective, total binary bit budget held exactly at the uniform-1-bpw level (83.36 MiB, costing each rank `m+n+16` bits including an fp16 mid scale).

**Error model.** Binary-factorization error is log-linear in rank over at least 0.6–1.4× the uniform rank: `E(r) = E_u · exp(−β(r − r_u))`. This was validated on q_proj and down_proj at 5 rank points each (and holds to full rank on down_proj: predicted 0.503 vs measured 0.500 at r=1151). The SVD tail is **not** a valid proxy — `E_admm/E_svd` drifts from 1.4 to 62 across ranks, so water-filling on singular values alone mis-allocates catastrophically (it cut one down_proj to r=127; a real run at r=392 measured 0.79 error). Calibration used one ADMM run per matrix at uniform rank (`E_u`) plus one probe per matrix type for `β`:

| type | β (per rank) | type | β (per rank) |
|---|---|---|---|
| mlp.down_proj | 6.22e-4 | self_attn.o_proj | 1.09e-3 |
| mlp.gate_proj | 6.32e-4 | self_attn.q_proj | 1.14e-3 |
| mlp.up_proj | 6.29e-4 | self_attn.k_proj | 3.18e-3 |
| | | self_attn.v_proj | 2.87e-3 |

**Allocation.** Lagrangian water-filling on the exponential model (marginal squared-error per bit equalized), ranks clamped to [0.6, 1.4]·r_u and below full rank. The optimizer cuts down_proj (mean multiplier 0.61×) and boosts everything else (k/v/o ≈ 1.22–1.25×, gate/up 1.17×, q 1.05×).

**Result (all 182 matrices re-run with real ADMM at the chosen ranks):**

| | global rel err | vs uniform |
|---|---|---|
| Uniform 1-bpw ranks | 0.5433 | — |
| Optimized allocation (predicted by model) | 0.5001 | −7.96% |
| Optimized allocation (**measured**) | **0.4988** | **−8.20%** |

The model's prediction was accurate to 0.24 pp, so the cheap calibrate-then-water-fill procedure can be trusted for planning without full validation sweeps. 119 of 182 matrices sat at the +40% clamp, so **more gain is available with wider bounds** (diminishing as matrices approach full rank, where binary error saturates around 0.50 for the MLP matrices); re-deriving β locally when moving far outside the calibrated range is advised.

**Caveats.** (1) The objective is equal-weight Frobenius across matrices; end-loss sensitivity may penalize the down_proj cuts specifically — an importance/Hessian-weighted rerun should precede adoption, and the same machinery applies (weight each matrix's squared error by its sensitivity). (2) β was calibrated per type from block 12 only; per-matrix β (one extra probe each) would sharpen the allocation. (3) For comparison, naive SVD-proxy allocation measured 8.5% on a 3-block sample, but only because large gains at capped matrices masked pathological cuts (down_proj 987→392 went 0.556→0.793); the calibrated model achieves its gain without such regressions.

## 9. Two more strategies tested and rejected (2026-07-18)

Both on block-12 q_proj and down_proj, same setup as §2, full pipeline (ADMM + scale ALS + flip descent) on each variant.

**9.1 Incoherence rotation (QuIP#-style) hurts this format.** Factorizing `Qₘᵀ·W·Qₙ` with random orthogonal rotations (spectrum-preserving; fast-Hadamard would be the production analog):

| variant | q_proj | down_proj |
|---|---|---|
| original | 0.4164 | 0.5545 |
| rotate both sides | 0.4344 | 0.5579 |
| rotate right (inputs) only | 0.4170 | 0.5569 |
| rotate left (outputs) only | 0.4340 | 0.5554 |

The q_proj damage comes almost entirely from the output-side rotation: the head-structured row magnitudes are precisely what `diag(post)` exploits, and incoherence flattens them into noise the binary factors can't represent. Conclusion: **do not port incoherence preprocessing into the current format** — rotations only pay when paired with codebook/lattice quantizers (where they are essential). This sharpens §6.3: rotation is part of the format *redesign*, not an add-on.

**9.2 Rank-group pre/post scales lose to rank at equal bits.** Generalizing the format to `Σ_g diag(post_g)·B_L,g·diag(mid)·B_R,g·diag(pre_g)` (ranks split into G groups sorted by |mid|, scales refit by batched closed-form ALS on the same binaries; G=1 reproduces `fit_scales` exactly). The scales do help — but each extra group pair costs `(G−1)(m+n)` fp16 values ≈ `16(G−1)` ranks, and rank is worth far more:

| G | q_proj E | same bits as rank instead | down_proj E | same bits as rank |
|---|---|---|---|---|
| 2 | 0.4165 | 0.4094 | 0.5547 | 0.5495 |
| 4 | 0.4157 | 0.3947 | 0.5543 | 0.5386 |
| 8 | 0.4141 | 0.3670 | 0.5533 | 0.5176 |

Group scales recover 0.1–0.7%; the equivalent rank buys 2–12%. Together with §5's block/stage results this closes the question: **within the binary-factorization family, every marginal bit belongs in rank**, and the current three-diagonal scale structure is already the right amount of scale freedom. The remaining big jumps require leaving the family (multi-bit factors, codebooks — §6.3).

## 10. Ternary loses; shared-input stacked factorization WINS (2026-07-18)

**10.1 Ternary factors {−1, 0, +1} at equal bits — rejected.** At 1.6 bits/trit (5-trits-per-byte packing), ternary rank = binary rank / 1.6. Fitter: ADMM binary init at the reduced rank, then coordinate descent where every entry may move to any of {−1, 0, +1} (closed-form ΔE via the Gram matrix), interleaved with scale ALS. The descent works hard — it zeroes 18–24% of q_proj entries and moves >1M entries on down_proj — but the rank deficit is unrecoverable:

| | ternary @ r/1.6 | binary @ r |
|---|---|---|
| q_proj | 0.4499 | **0.4164** |
| down_proj | 0.6186 | **0.5545** |

Caveat: coordinate descent from a binary init is a weak ternary fitter, so these are upper bounds on ternary error — but the 8–11% gap is far larger than any plausible fitter improvement, and assuming ideal entropy coding (~1.45 bits at the observed zero rates) does not flip the verdict either. Same conclusion as §9.2 from the other direction: per-entry expressiveness is a worse use of bits than rank.

**10.2 Stacked shared-input factorization — the first clear format win.** Matrices that consume the same input vector (q/k/v; gate/up) can be stacked row-wise and factorized jointly, sharing `B_R` and `pre` across the group; the shared bits buy extra rank at equal total budget (`r_stack = Σᵢ rᵢ(mᵢ+n) / (Σᵢ mᵢ + n)`). Deployment is the standard fused-QKV / fused-gate-up matmul. Results at equal bits, with a re-allocation control arm (k/v boosted to their rank cap, remainder to q) to separate basis-sharing from implicit rank re-allocation:

| q+k+v | separate, uniform ranks | separate, re-allocated | stacked (r=658) |
|---|---|---|---|
| block 0 | 0.5220 | 0.4854 | **0.4643** |
| block 12 | 0.5174 | 0.4612 | **0.3957** |
| block 24 | 0.5237 | 0.4833 | **0.4490** |

Stacking beats uniform separate fits by **11–24%** and still beats optimally re-allocated separate fits by **4–14%** — the excess is genuine basis sharing, not just allocation. A structural bonus: the 256-row k/v matrices are freed from their `min(m,n)` rank cap (the stack's min dim is the 1152 input dim), which is exactly where separate allocation saturates (§8 hit the same cap).

**gate+up stacking loses** (0.5487 stacked vs 0.5399 separate at block 12): the stacked rank (1063) approaches the shared input dimension (1152), where the shared right basis is nearly complete and stacking only removes the second pre/post diagonal pair. Rule of thumb: **stack when `r_stack` stays well below the shared input dim** (true for attention q/k/v; false for the wide MLP pair at 1 bpw).

Practical notes for adoption: the weighted objective extends trivially (concatenate output importances along the stacked rows; input importances are shared); export needs a fused-QKV layout in the packed/GGUF formats; and the allocation machinery of §8 applies unchanged with the stack treated as one matrix.

**Follow-up results — model-wide sweep, composite with allocation, 0.5 bpw probes, and a Gemma-3-4B generality check — live in [StackedFactorization.md](StackedFactorization.md)** (attention-side −14.9% model-wide; composite ≈ −9% global; q/k/v wins grow at tighter budgets; gate+up loses at every budget tested).
