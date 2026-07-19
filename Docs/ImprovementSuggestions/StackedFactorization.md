# Stacked Shared-Input Factorization (Fused q/k/v)

**Status:** Findings + proposal (experiments run 2026-07-18, including model-wide sweep, composite with allocation, 0.5 bpw probes, and a Gemma-3-4B generality probe)

**Audience:** Algorithm researchers and pipeline maintainers

**Related:** [ReconstructionHeadroom.md](ReconstructionHeadroom.md) (§8 rank allocation, §10 first stacked results), [21-weight-reconstruction-overview.md](../21-weight-reconstruction-overview.md), `src/nanoquant/domain/factorization.py`

## 1. Summary

Matrices that consume the same input vector — q/k/v within a block, gate/up within a block — can be stacked row-wise and factorized **jointly**, sharing the right binary factor `B_R` and the `pre` diagonal across the group. The bits saved by sharing buy extra rank at identical total budget. For attention q/k/v this is the largest same-budget reconstruction win found to date: **11–24% lower combined error vs uniform separate fits, and 4–14% beyond what optimal rank re-allocation between the separate fits can achieve.** Deployment is the standard fused-QKV matmul; no new kernel math is required, only a fused export layout.

Model-wide (all 26 blocks): stacking cuts the **attention-side** error 14.9% (0.5064 → 0.4307); the **global** effect alone is 1.2% because MLP matrices dominate Frobenius mass. Combined with cross-layer rank allocation the two compose: ~9% global at equal bits (validated; wider bounds add only ~0.2% more — allocation is saturated). The q/k/v win holds at 0.5 bpw (16% lower) and on Gemma-3-4B (mean +5.4% over nine blocks).

On real activations the case is stronger still (§3.8): stacking survives activation-space evaluation (−7 to −18%), **input-importance weighting is the largest single lever measured to date** (−20 to −30% functional error, obtained for free via a column-scaling identity), and stacked+weighted together cut functional error **36–44%** vs today's unweighted separate fits at identical bits.

Stacking is not universally good: **gate+up loses at both 1 bpw and 0.5 bpw.** Staying well below the input-dim rank cap is necessary but *not sufficient* — the group members must actually share right-basis structure. Empirically attention q/k/v do; the SwiGLU gate/up pair does not.

## 2. The construction

For matrices `W₁ (m₁×n), …, W_g (m_g×n)` sharing an input space, factorize the row-stack `W = [W₁; …; W_g]` as usual:

```
[W₁; …; W_g] ≈ diag(post) · B_L · diag(mid) · B_R · diag(pre)
```

`B_R (r×n)` and `pre (n)` are shared; `post` and the rows of `B_L` partition among the group members. Equal-bit rank:

```
r_stack = Σᵢ rᵢ(mᵢ + n) / (Σᵢ mᵢ + n)
```

For Gemma-3-1B attention (q: 1024×1152, k/v: 256×1152, uniform ranks 542/209/209) this gives r_stack = 658 — more effective rank for every member than uniform allocation gave the largest of them, because the k/v rows ride along nearly free.

Two structural effects beyond raw bit savings:

- **Rank-cap liberation.** A separate 256×1152 k_proj can never exceed rank 256; §8 of ReconstructionHeadroom showed the allocator slamming into exactly this cap. The stack's rank cap is the input dim (1152).
- **Basis sharing.** q/k/v read the same residual-stream directions; a joint `B_R` spends its rows on directions all three consume, instead of three fits rediscovering overlapping bases.

At inference the stacked reconstruction is one matmul producing the concatenated q|k|v outputs — the fused-QKV layout most runtimes already prefer.

## 3. Verified findings

All numbers: real Gemma-3-1B weights, unweighted Frobenius objective (`‖W−Ŵ‖/‖W‖`), equal total bits vs the uniform ~1 bpw baseline, production `factorize_admm` defaults. "Combined" separate errors are Frobenius-weighted over the group.

**3.1 q+k+v stacking wins on every block tested, including against an allocation control.** The control arm re-allocates the separate fits' ranks optimally at the same bits (k/v to their 255 cap, remainder to q), isolating basis-sharing from implicit allocation (blocks 0/12/24, full pipeline: ADMM + scale ALS + sign-flip descent on all arms):

| block | separate, uniform ranks | separate, re-allocated | stacked (r=658) | stacked vs realloc |
|---|---|---|---|---|
| 0 | 0.5220 | 0.4854 | **0.4643** | −4.3% |
| 12 | 0.5174 | 0.4612 | **0.3957** | −14.2% |
| 24 | 0.5237 | 0.4833 | **0.4490** | −7.1% |

Roughly half to two-thirds of the raw win is rank re-allocation (which stacking performs implicitly and optimally); the remainder is genuine basis sharing.

**3.2 gate+up stacking loses at 1 bpw.** Block 12: stacked (13824×1152, r=1063) E=0.5487 vs separate combined 0.5399. The stacked rank sits at 92% of the shared input dim, where a shared basis is nearly complete anyway and stacking only removes one pre/post diagonal pair while forcing a single shared basis. This is the boundary condition, not a contradiction.

**3.3 The win is robust to fit quality.** The 3.1 numbers include scale ALS + flip descent on both arms; raw-ADMM comparisons (§10.2 of ReconstructionHeadroom, and the model-wide sweep below) show the same margins, so the advantage is not an artifact of refinement interacting with shape.

**3.4 Model-wide sweep (all 26 blocks, raw ADMM both arms).** Stacked at r=660 per block vs the calibrated separate-uniform baselines: attention-side combined error 0.5064 → 0.4307 (**14.9% lower**); per-block stacked errors range 0.3957–0.4646. Global error including the MLP side moves 0.5433 → 0.5370 (1.2%) — attention is a small share of total Frobenius mass, so the global Frobenius number understates the attention-side improvement (and says nothing about perplexity weighting).

**3.5 Composite: stacking + cross-layer rank allocation.** Treating each block's stack as one allocation unit (measured β_stack = 1.105e-3, essentially q_proj's β) and re-running the §8 water-filling over {stack, o, gate, up, down}×26 at equal total bits: **predicted global 0.4941 vs 0.5433 uniform (−9.1%)**, beating allocation-alone (0.4988 measured) and stacking-alone (0.5370 measured). The allocator exploits the cap liberation directly — every stack is pushed to 822–923 ranks (+25–40%, at the 1.4× clamp), draining bits from spectrally-flat MLP matrices. Spot validation at these extrapolated ranks (blocks 0/12/24): measured 0.3700/0.3203/0.3633 vs predicted 0.3474/0.3101/0.3447 — the log-linear model is ~3–6% (relative) optimistic when extrapolating 25–40% beyond the calibration point, so the honest composite estimate is **global ≈ 0.495–0.50, i.e. ~8.5–9% below uniform**, with attention-side errors around 0.32–0.37 (vs 0.51 uniform). Re-calibrating β locally before committing an allocation removes most of this drift.

**3.6 Budget sensitivity: the q/k/v win grows at tighter budgets; gate+up loses everywhere.** At 0.5 bpw (block 12): q+k+v stacked 0.5895 vs separate 0.7025 (**16.1% lower**); gate+up stacked 0.7457 vs separate 0.7329 (still a loss, despite r_stack=532 being far below the 1152 cap). This falsifies the earlier hypothesis that gate+up only lost to the rank cap: gate and up simply do not share enough right-basis structure to pay for merging, at any budget tested.

**3.7 Scale/GQA generality: wins on Gemma-3-4B on average, block-dependent.** Nine blocks of google/gemma-3-4b-it probed (q: 2048×2560, k/v: 1024×2560 — 2:1:1 GQA vs 1B's 4:1:1; stacked 4096×2560 at r=1578): gains of 8.1% (block 12), 8.3% (2), 10.1% (6), 0.2% (10), 6.8% (16), 11.9% (20), 3.5% (24), 0.6% (28), and one small loss, −1.3% (32). Mean ≈ +5.4%. Weaker and more variable than 1B — consistent with the mechanism (4B's k/v are less cap-starved, so more of the win must come from basis overlap, which varies by depth). Practical consequence: **the stack-vs-separate decision is measurable per block at calibration time** (a few seconds of ADMM per block) and should be made per block rather than globally.

**3.9 Wider allocation bounds are nearly exhausted; the model is accurate once β is recalibrated.** Re-running the composite with bounds [0.5, 2.0]× and a piecewise β for stacks (measured 1.105e-3 below r=660, 9.03e-4 above, from the §3.5 validation points): predicted global 0.4930 vs 0.4941 for the 1.4×-clamped composite — widening the bounds buys only ~0.2% more. Stacks land at 966–1150 (some at the 1151 cap); down_proj slams into the 0.5× *lower* bound at E≈0.72–0.77. Validation on all 15 units of blocks 0/12/24 at the chosen ranks: predicted 0.4970 vs measured 0.4958 — the piecewise model is now slightly conservative rather than optimistic, so allocation planning can trust it. **Caution:** the equal-weight Frobenius objective is what drains down_proj so aggressively; given §3.8's finding that functional and Frobenius rankings diverge, the composite should be re-derived under importance weighting before adoption — down_proj's true sensitivity may not tolerate E≈0.75.

**3.8 The win survives real activations and the weighted objective — and the levers compose.** Using q/k/v-input activations captured from google/gemma-3-1b-it on wikitext-2 (24×512 tokens), three metrics per arm: unweighted Frobenius, input-importance-weighted Frobenius (imp = E[x²] per channel; dynamic range 500–8000× across channels), and activation-space error `‖X(Ŵ−W)ᵀ‖/‖XWᵀ‖` — the layer's true functional error and the best cheap proxy for end quality. Weighted fitting is exact and free: fit `W·diag(√imp)` with the standard unweighted pipeline and divide `pre` by `√imp` afterward (the format's free per-column diagonal absorbs the weighting).

Activation-space error, equal bits:

| block | separate unweighted | stacked unweighted | separate weighted | **stacked weighted** |
|---|---|---|---|---|
| 0 | 0.3255 | 0.2844 | 0.2526 | **0.2083** |
| 12 | 0.3951 | 0.3235 | 0.3104 | **0.2265** |
| 24 | 0.3916 | 0.3657 | 0.2734 | **0.2208** |

Three conclusions: (1) stacking's advantage is not a Frobenius artifact — it persists on real activations (−7 to −18% unweighted); (2) **input-importance weighting alone is the largest single lever measured so far** (−20 to −30% functional error vs unweighted fits — this is the first quantification of the dormant importance objectives); (3) the levers compose — stacked+weighted cuts functional error **36–44%** vs today's unweighted separate fits at identical bits. Weighted fits show *worse* unweighted Frobenius (0.42–0.50 vs 0.39–0.46 stacked) while being far better functionally — Frobenius rankings and functional rankings genuinely diverge, so future format comparisons should report activation-space error.

## 4. Unverified — pending testing

- **End-model quality (perplexity).** Activation-space error (§3.8) is the strongest proxy measured so far and confirms the direction, but the end-to-end pipeline + eval run remains the gating question for adoption (attention errors interact through softmax and later blocks nonlinearly).
- **Output-importance weighting.** §3.8 verified the input-importance side (the dominant term). Row-side (output) importances — including whether q vs k/v sensitivity differences shift the stacked fit — remain untested; the extension is mechanical (concatenate output importances along stacked rows).
- **Weighted objective for MLP and o_proj.** The −20 to −30% functional gain from input-importance weighting was measured on the attention input only; the same identity applies to every layer and should be validated there too — it is currently the largest single unadopted lever.
- **Interaction with STE distillation tuning.** `left_latent`/`right_latent` carry tuning state; a stacked layer tunes as one unit. Whether network-level STE preserves or amplifies the stacked advantage is unmeasured.
- **The pipeline's actual planner.** The composite in §3.5 uses the idealized water-filling model; `domain/planning.py` has integer/format/retry constraints not modeled. Also, stacks at extrapolated ranks drift ~3–6% above the log-linear prediction — re-calibrate β near the target rank before committing an allocation.
- **Other model families.** Verified on Gemma-3-1B (4:1:1, all blocks) and nine blocks of Gemma-3-4B (2:1:1, mean +5.4%, one slight loss). Non-Gemma architectures and 12B remain unprobed; per-block adoption decisions (cheap to measure) are recommended regardless.
- **Importance-weighted composite allocation.** §3.9 shows equal-weight Frobenius drives down_proj to E≈0.75 at the allocation optimum; §3.8 shows functional and Frobenius rankings diverge. The composite must be re-derived with per-layer importance weights (activation-space calibrated) before its ~9% global figure can be trusted as an end-quality proxy.
- **Full-pipeline parity concerns.** ADMM's `transpose_wide` orientation, run-local calibration tokens, and checkpoint formats all assume per-layer tensors; a stacked unit touches each of these contracts.
- **Export and runtime.** Packed layout, GGUF conversion, and the CUDA/llama.cpp runtimes need a fused-QKV (and per-member slicing) story. The math is one matmul; the plumbing is real work: `runtime/packed.py`, `infrastructure/gguf_export.py`, `tools/llamacpp/convert_nanoquant_to_gguf.py`.
- **Calibration statistics.** Per-member output importances are collected today per layer path; the stacked unit needs concatenated accumulators (`application/calibration.py`).
- **o_proj and down_proj have no stacking partner** (unique input spaces). Their gains must come from allocation alone. Cross-block stacking is ruled out by construction — different blocks read different activations, so a stacked matmul cannot be fused at runtime.

Falsified along the way (do not re-test): gate+up stacking at lower budgets — it loses at 0.5 bpw too (§3.6), so the failure is basis mismatch, not the rank cap.

## 5. Adoption sketch

1. Stack q/k/v per block ahead of the factorization stage (one new stage or a transform in layer enumeration); keep gate/up separate at ≥1 bpw budgets.
2. Concatenate output-importance vectors in the same order; share the input-importance vector.
3. Treat each stack as a single unit in rank allocation (β calibrated per §8 of ReconstructionHeadroom).
4. Export either as a fused unit (preferred; runtimes already like fused QKV) or unstack `B_L`/`post` rows into three logical layers sharing `B_R`/`pre` storage.
