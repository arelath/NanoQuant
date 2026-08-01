# Format Capacity: Why Allocation Has Stalled, and What Is Left

**Date:** 2026-07-29
**Status:** Idea catalog. The §1 capacity measurements are computed from the repository's own
`factor_bit_cost` and reproduce [Doc 38](../38-full-rank-bpw-estimates.md) exactly. Everything
proposed in §3–§8 is UNTESTED unless it cites a measurement.
**Prerequisites:** [ErrorAnatomy.md](ErrorAnatomy.md) (where the error lives),
[ReconstructionHeadroom.md](ReconstructionHeadroom.md) (the falsification table),
[NextQualityLevers.md](NextQualityLevers.md) (the prior catalog),
[Doc 33](../33-error-budget-driven-quality-improvements.md) (D1–D5 outcomes).

---

## 0. Where the project actually stands

The pinned Gemma-3-1B baseline is Experiment 022: WikiText-2 perplexity **96.460 → 228.551**,
mean task accuracy **0.6233 → 0.4692**, effective **1.024495 BPW**. Everything since has failed to
beat it:

| Attempt | Result vs Exp 022 | Source |
|---|---|---|
| Experiment 023 — interaction-corrected D2 | ppl 237.234 (worse) | [Exp 023](../Experiments/023-gemma-3-1b-d2-interaction.md) |
| Experiment 024 — combined "best methods" | ppl 235.283 (worse) | [Exp 024](../Experiments/024-gemma-3-1b-best-methods.md) |
| Experiment 032 — raw Fisher (won static KL by 13.13%) | ppl 7.22% worse | [Exp 032](../Experiments/032-gemma-3-1b-raw-fisher-d2.md) |
| D3 closed-form bias correction | `o_proj` splice KL +21.66% | [Doc 33](../33-error-budget-driven-quality-improvements.md) |
| D5 `o_proj` low-rank patch, equal budget | `o_proj` KL +14.25% | [Doc 33](../33-error-budget-driven-quality-improvements.md) |
| Fixed QV/KO attention topology | attention KL +27.29% | [Doc 41](../41-attention-partition-functional-gate.md) |
| Input-only structured Hadamard | joint KL +46.6% to +57.4% | [Doc 45](../45-input-hadamard-covariance-screen.md) |
| Sparse / column residual patches | every arm regresses | [Doc 39](../39-factor-grouping-and-sparse-outlier-probe.md) |

That is eight consecutive negative results against one baseline. The pattern is consistent enough to
be diagnostic rather than unlucky, and §1 proposes what the common cause is.

Two rules follow from the table and should govern every proposal below:

1. **Name the funding source.** D5 is the cleanest lesson in the repository: the activation-space
   patch ceiling was real (−33.83% splice KL when unfunded) and the *same patch* regressed +14.25%
   once the allocator shaved binary rank to pay for it. An idea that costs bits is not a win until
   it beats what those bits were already buying.
2. **Local wins do not add.** Experiment 024 combined the individually-plausible levers and lost to
   the simpler recipe at the same budget.

---

## 1. The finding: the format's entire capacity range is 1.00 → 1.26 BPW

`ReconstructionAllocationUnit.maximum_rank` is `min(in_features, out_features)`
([application/planning.py:391](../../src/nanoquant/application/planning.py:391)), and
`factorize_admm` rejects anything larger
([domain/factorization.py:166](../../src/nanoquant/domain/factorization.py:166)). Applying the
repository's own `factor_bit_cost` at that ceiling, per Gemma-3-1B decoder block, using the
production fused-QKV topology:

| Physical unit | rank at 1.0 BPW | maximum rank | headroom | BPW at maximum rank | share of type-summed KL |
|---|---:|---:|---:|---:|---:|
| QKV (fused) | 638 | 1,152 | 1.80× | 1.785 | 13.1% |
| `o_proj` | 522 | 1,024 | 1.96× | 1.932 | 14.9% |
| `gate_proj` | 970 | 1,152 | **1.19×** | **1.185** | 23.1% |
| `up_proj` | 970 | 1,152 | **1.19×** | **1.185** | 28.5% |
| `down_proj` | 970 | 1,152 | **1.19×** | **1.185** | 20.3% |

KL shares are ErrorAnatomy Finding F. The rank-638 QKV figure independently reproduces the value
Doc 39 reports for the production topology, and the whole-block maximum-rank total of **1.2576 core
BPW** reproduces [Doc 38](../38-full-rank-bpw-estimates.md)'s independently-derived 1.2537 for the
separate-projection layout. The arithmetic is the repository's, not new modelling.

Three consequences, in increasing order of importance.

### 1.1 The allocator is nearly out of room, and its clamp is not what binds

The reconstruction-aware/D2 path that Experiment 022 uses clamps rank to [0.6, 1.4]× of uniform
(`RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE`,
[experiments/recipes/base_compression.py:195](../../experiments/recipes/base_compression.py:195));
the plain base template is tighter still at [0.9, 1.1]. For the three MLP projections the
**physical** ceiling arrives at 1.19× — inside even the wider clamp. So for 89.0% of the quantized
weights, the allocator's upper bound is unreachable: it cannot give `up_proj` 1.4× rank even when
its own measured KL profile says it should.

This retro-explains three previously puzzling results:

- ReconstructionHeadroom §8 found 119 of 182 matrices sitting at the +40% clamp. For the MLP
  matrices that were "at the clamp," many were in fact at the physical cap.
- NextQualityLevers records "wider allocation bounds (>1.4×) → +0.2% only — saturated." That is not
  a diminishing-returns curve; it is a wall.
- D2/KL-calibrated allocation delivered small and inconsistent gains (Exp 021 helped, Exp 022 helped
  over Exp 017, Exp 023 and Exp 024 regressed) despite ErrorAnatomy measuring an enormous and
  correctly-signed KL concentration. The signal was right; the actuator was saturated.

**Allocation is close to exhausted as a lever.** It should stop being treated as the frontier.

### 1.2 Where the damage is, the format is capped tightest

The inverse correlation between headroom and damage is near-perfect. The two units with real
headroom (`o_proj` 1.96×, fused QKV 1.80×) carry 28.0% of type-summed KL. The three units pinned at
1.19× carry 71.9% — and ErrorAnatomy separately confirms `up_proj` as the single worst type
end-to-end.

### 1.3 At maximum rank the format still has ~50% weight error

ReconstructionHeadroom §8 measured `down_proj` at essentially full rank (r=1151): relative Frobenius
error **0.500**, against 0.5545 at the 1-BPW rank of 987. So spending every remaining representable
bit on the worst-affected 89% of the model buys a **9.8% error reduction for 18.5% more bits**, and
lands at 0.50 — which is the iid-Gaussian Shannon reference point at 1 bpw (ReconstructionHeadroom
§4), i.e. no better than an unstructured code.

Putting §1.1–§1.3 together:

> **NanoQuant as currently implemented cannot produce a model above ~1.26 core BPW at all, and the
> MLP side saturates at 1.185 BPW with half its weight energy unrepresented. The distance from the
> shipped 1.0245 BPW operating point to the format's absolute ceiling is 22% of bits and roughly 10%
> of error. No allocation policy, objective, calibration, or fitter can cross that.**

This is why nine consecutive local improvements have failed to move end-to-end quality. They were
competing for a budget that has nowhere left to go.

---

## 2. How to read the rest of this document

§3–§5 cost **zero bits** and are compatible with the shipped artifact format and runtime. §6–§7
cost bits and must state what funds them. §8 lists things deliberately not proposed.

---

## 3. Lift the `rank ≤ min(m, n)` cap (the headline proposal)

### 3.1 The cap is a policy, not a property of the format

For a real-valued factorization, rank `min(m,n)` is a genuine ceiling — beyond it the product cannot
improve. **For sign factors it is not.** The reconstruction identity is

```
Ŵ = diag(post) · U · diag(mid) · V · diag(pre),    U ∈ {±1}^(m×r),  V ∈ {±1}^(r×n)
```

The set reachable with `r = min(m,n)` is a strict, small subset of all matrices — which is exactly
why `down_proj` still measures 0.500 relative error at r=1151 where a real-valued factorization of
the same rank would measure 0. It is the binary constraint, not the rank, that is unsatisfied at the
cap. Adding rank components beyond `min(m,n)` is well-defined, strictly enlarges the reachable set,
and is the only remaining way to buy capacity **without leaving the shipped format**.

That is a capability claim, and it is as far as this section goes. **The rate at which error falls
past the cap is unmeasured.** The three scale diagonals are shared across all rank components, so
components added beyond `min(m,n)` are increasingly redundant against `diag(mid)`'s single value per
component, and the log-linear response model was fitted entirely inside the cap. The extrapolation in
§3.2 is the screen's null hypothesis, not a forecast.

The constraint exists in exactly two places, both policy:

- [application/planning.py:391](../../src/nanoquant/application/planning.py:391) — `maximum_rank`
- [domain/factorization.py:166](../../src/nanoquant/domain/factorization.py:166) — the ADMM guard

Nothing downstream requires it. The packed layout stores `U` as `[out, ceil(rank/32)]` I32 words and
`V` as `[rank, ceil(in/32)]` (Doc 19); the two-stage kernel (Doc 21) reduces inputs into a
rank-sized latent and then into outputs. Both are rank-agnostic. Cost stays exactly
`r(m+n) + 16(m+n+r)`, so `factor_bit_cost` needs no change either — only its caller's ceiling.

### 3.2 The decisive screen is one matrix and a few GPU-minutes

Before any planner or format work, answer one question: **does `E(r)` keep falling past
`r = min(m,n)`?**

Fit one Gemma-3-1B `down_proj` (1152×6912) with production ADMM at r ∈ {970, 1152, 1440, 1728,
2304}, unweighted and Fisher-weighted, and plot relative error. ReconstructionHeadroom's log-linear
model `E(r) = E_u·exp(−β(r−r_u))` with β = 6.22e-4 held to full rank on this exact matrix (predicted
0.503, measured 0.500), so it supplies a falsifiable null: r=1728 should reach ≈0.35.

**Use two matrices, not one.** `down_proj` has ReconstructionHeadroom's full-rank datapoint, which is
why it is the natural anchor — but its `min(m,n) = 1152` is the *output* dimension, so at r=1728 the
left factor becomes 1152×1728, i.e. more rank components than output rows. That is the most redundant
possible configuration for the shared `diag(post)`, and therefore the case least likely to show a
gain. Add one near-square matrix with the opposite aspect ratio — `o_proj` (1152×1024) or the fused
QKV group (1536×1152) — to the same session. Without it, a flat curve cannot distinguish "the binary
family is exhausted at its cap" from "wide matrices saturate and square ones do not," and §11 puts
this screen first precisely because everything downstream depends on telling those apart.

- If measured error tracks the extrapolation, over-complete rank is a **continuous capacity dial**
  and the 1.26 BPW ceiling is an artifact of two lines of code.
- If error flattens at the cap, the binary family is genuinely exhausted there and §3 closes —
  which is itself a decisive, publishable result that redirects effort to §7.

### 3.3 Honest risks

- **Initialization.** ADMM's SVID projection derives signs from a continuous relaxation whose natural
  rank is capped. An over-complete solve needs a new init (random, or warm-started from the rank-cap
  solution plus fresh components). This is real implementation work, not a config flag.
- **The closest negative evidence points at the naive implementation.** Multi-stage residual binary
  fitting was falsified at equal bits (0.4233 for two r=271 stages vs 0.4164 for one r=542), and
  ReconstructionHeadroom explicitly notes that a fresh ADMM solve on a residual-structured target is
  "the only setting where we observed genuine optimizer fragility." So the experiment must be a
  **joint** over-complete solve. A sequential residual construction is the thing already known to
  lose.
- **This raises BPW rather than redistributing it.** That makes it a new operating point, not an
  equal-bit win — see §6 for why that is defensible here.
- **Compute.** Decode cost is `r(m+n)` instead of `mn`; at r ≈ 1.5·min(m,n) the MLP projections stop
  being cheaper than a dense matmul. The screen must be paired with a runtime measurement before
  adoption, and the result may be "worth it for quality, not for speed."

### 3.4 The k/v special case

`k_proj` and `v_proj` are 256×1152, so their standalone maximum rank is 256 and their cost *at* that
rank is **1.3125 BPW** — already above the 1.0 target, meaning the base recipe's promotion of every
k/v to physical maximum ([Doc 22](../22-base-compression-recipe.md)) is subsidised by other layers.
Fused QKV already relieves this by lifting the group cap to 1152 (ReconstructionHeadroom §10.2
notes the same effect). The point worth recording is that the subsidy buys a saturated format:
the correct question is not "should k/v get more bits" but "does the QKV group want over-complete
rank," which §3.2's screen answers on the same footing as the MLP.

---

## 4. Wire the low-rank-plus-diagonal covariance objective (zero bits)

[Doc 44](../44-covariance-headroom-probe.md) measured a **41.53% held-out same-rank real-valued
error-energy reduction** for a full input covariance over its diagonal. That is a decision bound
under an unconstrained real-valued rank-r reconstruction — explicitly *not* an expected end-to-end
gain — but it is broad (11 of 12 groups clear 20%) and it is held out.

Doc 44 excluded `down_proj` because its 6,912-wide dense covariance is ~182 MiB per matrix. That
exclusion removes the projection carrying 20.3% of type-summed KL, and the promoted dense
covariance-weighted screen (`tools/probe_covariance_binary.py`) inherits it.

**`LowRankDiagonalObjective` already exists** in `domain/objectives.py`, alongside `whiten`/
`unwhiten` and covariance materialization in `application/covariance.py` — and per Fable.md,
`load_covariance_objective` and `transform_for_factorizer` have **no callers**. A diagonal-plus-top-k
eigenvector metric captures most correlation structure at a small fraction of the memory and is the
only covariance form that reaches the wide MLP inputs.

**Why Doc 45's failure does not apply.** The Hadamard screen failed because rotating into a
decorrelated basis destroyed coordinate structure the binary factors exploit — the transform changed
*what is factorized*. A low-rank covariance metric rotates nothing and changes only *how error is
weighted*. That is the same class of change as the already-adopted diagonal importance weighting.
Anyone reading Doc 45 first will pattern-match "covariance work → already failed"; it did not.

**Test.** Same protocol as Doc 44 — blocks 0/12/24, identical ranks, seeds, and physical bits —
adding `down_proj` with a diag + top-{16, 64, 256} metric. Promote on paired held-out splice KL, not
on reconstruction RMSE (§9). **Effort:** low; the objective code exists and is untested only because
it is unwired. **Composes with:** the in-flight dense screen, additively — this is the cheap arm that
covers the projection the expensive arm cannot.

---

## 5. Fix the tail-blind distillation objective (zero bits)

`topk_distillation_loss` computes `log_softmax` over **only the teacher's top-k selected logits**
([application/distillation.py:220](../../src/nanoquant/application/distillation.py:220)). Student
probability mass placed outside those indices is invisible to the loss. `FULL_KL` exists as a config
enum ([config/schema.py:118](../../src/nanoquant/config/schema.py:118)) with no implementation.

This was raised in [GPT5_6.md](GPT5_6.md); only that document's BF16-scales item was subsequently
implemented, so the objective gap is still live in current code — verified directly, not inferred.

The gap matters because a ~1-bpw student's characteristic failure is diffuse mass on wrong tokens,
which is precisely what this loss cannot see. GPT5_6 supplies the corroborating anomaly: in the run
it examined, global KD *worsened* perplexity 415.16 → 453.57 while the cached KD objective decreased
2.3988 → 2.1484 — an objective improving while the metric it proxies degrades, exactly the signature
of an unnormalized surrogate.

**Proposal.** Cache the teacher's full log-normalizer and tail mass alongside the top-k values, and
compute the student's full-vocabulary denominator chunkwise. Cost is one extra scalar pair per token
in the cache and one chunked reduction per step. Optionally blend hard-label CE.

**Test.** Re-run global distillation on the Experiment 022 artifact with the corrected objective,
selected on held-out sequences disjoint from the KD cache, with pre-KD rollback as checkpoint zero.
**Success criterion:** held-out perplexity improves over both the current KD result and the pre-KD
artifact. **Effort:** low-medium. **Risk:** this is a re-raise, not a new theory — its value is that
it is cheap, verified-present, and gates every recovery pass downstream.

---

## 6. Target a higher core BPW deliberately (bit cost — with an unusual funding story)

Every experiment in the repository is pinned near 1.02 effective BPW, and §1 shows that band is
within 22% of the format's absolute ceiling. Meanwhile the shipped model is at 2.37× baseline
perplexity and has lost 15 points of task accuracy — a long way from usable. It is worth asking
whether 1.0 BPW is the right target rather than an inherited one.

The funding story here is different from D5's, and it is the reason this is defensible rather than
just "spend more bits":

[Doc 38](../38-full-rank-bpw-estimates.md) separates **core BPW** (the seven decoder matrices) from
**whole-text-model BPW**. For Gemma-3-1B those are 1.2537 and 3.4442 respectively at maximum rank —
because Q8_0 token embeddings dominate a small model's payload. Moving core BPW from 1.02 to 1.5
therefore changes the shipped artifact by far less than it appears to: the decoder is a minority of
the bytes at this scale. The comparison that matters to a user is total file size at equal quality,
and that comparison currently is not being run.

**This argument is scale-dependent and weakens exactly where the roadmap is headed.** Doc 38's two
columns converge as models grow — 1.2537 vs 3.4442 at Gemma-3-1B, but 1.3984 vs 1.6102 at Llama-3.3-70B
— because the embedding stops dominating. At the 8B sizes named in [TODO.md](TODO.md) for paper
comparison, core BPW is most of the payload and a core increase is nearly a payload increase. The
sweep must therefore report both columns rather than leaning on the small-model framing.

**Proposal.** Run the existing pipeline at core targets {1.0, 1.25, 1.5, 2.0} — the last two
requiring §3 — and report quality against *both* core BPW and total payload bytes. This reframes the
project's operating point as a measured choice rather than a default, and it is the natural consumer
of §3's capacity if the screen passes.

---

## 7. Global QAT on preserved binary latents (zero deployed bits)

Global distillation freezes the sign factors and trains only scales, outlier values, biases, and
norms. ReconstructionHeadroom falsified **per-layer STE against the Frobenius objective** — a
different objective and a different scope.

The distinction is the same one that promoted scale-only distillation in NextQualityLevers §15:
a *global* objective over parameters that a *local* objective could not improve. Extending it to the
signs requires preserving the pre-sign latent margins as training-only artifacts (the SVID export
already produces them, per Doc 21 §5), a much smaller LR for the binary latents than for scales, a
trust region or gradient clip, and held-out rollback. This changes training artifacts only —
deployment size is unchanged. BitDistiller and OneBit both report this class of pass mattering at
ultra-low bit widths.

**Gate this behind §5.** Training signs against a tail-blind objective is the worst possible
combination: maximum parameter freedom pointed at a surrogate that cannot see the dominant failure
mode. This is not merely prudent ordering — Experiment 032 is the repository's measured instance of
exactly that failure: a change that won its proxy gate by 13.13% with a clean interval and lost the
complete run by 7.22% perplexity. Unfreezing the signs is the highest-variance change in this
document, so it needs the objective fixed first and a held-out rollback throughout. **Effort:**
medium. **Label:** re-raise of GPT5_6 item 5, still unimplemented.

---

## 8. One cheap measurement that reopens or closes a demoted lever

ErrorAnatomy Finding H demoted error-feedback (propagated) calibration because per-block KLs were
**sub-additive** (sum 5.170 vs whole-model 4.675, ratio 0.90) — no compounding for it to correct.
The document set the revisit condition as "after total KL drops an order of magnitude."

That ratio was measured at an operating point that no longer exists: uniform ranks, no allocation,
no tuning, perplexity 7262. The shipped baseline is at 228.6. **The premise deserves re-measurement,
not assumption in either direction** — and note that the shipped ppl gap is an NLL quantity, not KL
against the teacher, so it does not by itself establish that the trigger has fired (Doc 41 shows the
two can diverge markedly).

**Test.** Re-run the existing `application/kl_budget.py` harness on the Experiment 022 artifact for
`block:*` arms plus `full`, and recompute the additivity ratio. This is minutes of GPU time on
already-built machinery. If the ratio has risen toward or above 1.0, NextQualityLevers §3 reopens
with a measured premise. If it is still ≈0.9, the demotion is confirmed with current evidence
instead of inherited evidence. Either outcome is worth the cost.

---

## 9. Measurement discipline (reinforced by the newest failures)

The repository's own results have now falsified three proxy metrics in a row. Recording them
together, because each was individually persuasive:

- **Plain Frobenius RMSE ranks arms backwards.** Shrinkage 0.6 wins original-space RMSE and loses
  held-out KL by 13.13% (Doc 42); `alpha=0.5` monotonically improves unweighted RMSE while being
  decisively harmful functionally (Doc 43).
- **Weighted matrix RMSE is also insufficient.** QV/KO improved corrected-Fisher RMSE by 1.035% and
  raised attention KL by 27.29% (Doc 41). Block 17 improved its matrix RMSE by 13.97% while its
  actual attention output got 19.4% worse. A scalar sum of weighted matrix errors can reward trades
  that are destructive inside a nonlinear operator.
- **Static held-out KL does not survive the full lifecycle.** Raw Fisher won the static gate by
  13.13% with a clean confidence interval and lost the complete run by 7.22% perplexity (Exp 032).
  Allocation, outlier selection, tuning, and distillation interact with the change being tested.

**Practical rule:** for any candidate touching the objective or the weighting, the promotion gate is
paired held-out KL *plus* a complete run. For any candidate touching topology or an operator's
internals, add operator-level output error. Reconstruction error of any flavour is a screen for
generating candidates, never for selecting them.

---

## 10. Deliberately not proposed

| Idea | Why not |
|---|---|
| Entropy-coding the sign matrices | ADMM signs are near-maximum-entropy by construction, and any variable-length code breaks the I32 load-width kernel contract (Doc 19). A positive measurement would not be shippable. |
| Further attention topology enumeration | Doc 40 enumerated all 15 partitions; Doc 41 rejected the only winner functionally. Attention is 13.1% of the KL budget. The selection objective, not the candidate set, was the problem. |
| Rotations / incoherence preprocessing | Falsified twice, in both configurations (ReconstructionHeadroom §9.1 both-sides and output-side; Doc 45 input-only). |
| Sparse or column residual patches | Doc 39 tested both against the post-factorization residual at equal bits; sparse dominates columns as a representation but both lose to binary rank. |
| Ternary factors, column blocks, rank-group scales | All falsified at equal bits — but note every one of these was measured in the rank-*unsaturated* regime where the counterfactual was "spend it on rank." §3 is the one direction that argument does not cover. |
| Per-head attention factorization | NextQualityLevers §14 — the semantic version of column blocks, which lost decisively. |

Two items reduced to footnotes rather than sections:

- **Embedding/head compression.** Real, but GPT5_6 already made the argument, the base recipe
  already ships Q8_0 embeddings, and it is a total-file question rather than the decoder-BPW question
  the pipeline optimizes. Doc 38's core-vs-whole-model columns quantify it; §6 is where it belongs.
- **Domain-matched calibration data.** Doc 36 covers this for Qwen3 thinking traces and
  `teacher_dataset.py` plus Doc 38-reusable-teacher-dataset-builder exist. Fold into
  NextQualityLevers §4's robustness work rather than treating as new.

One accounting inconsistency worth fixing but not worth a campaign: `outlier_bit_cost`
([domain/planning.py:339](../../src/nanoquant/domain/planning.py:339)) derives index width from
`out_features` when no `index_bits` is passed — and none of its four callers pass one — while indices
identify *input* columns and the artifact stores them as I32. Doc 21 puts total outlier indices at
~2 KB, so this is a correctness footnote in the plan's bit accounting, not a lever.

---

## 11. Suggested order

1. **§3.2 over-complete rank screen** — one matrix, few GPU-minutes, and it decides whether the
   project has a capacity dial at all. Everything about the long-term direction hinges on the answer,
   and a negative result is as valuable as a positive one.
2. **§8 additivity re-measurement** — minutes on existing machinery, reopens or closes a demoted
   lever with current evidence.
3. **§5 KD tail fix** — verified-present gap, cheap, and gates §7.
4. **§4 low-rank covariance objective** — the code exists unwired, and it is the only covariance form
   that reaches `down_proj`.
5. **§6 BPW sweep** — once §3 is answered, report quality against total payload bytes.
6. **§7 global binary QAT** — only after §5 lands.

The through-line: stop optimizing the allocation of a budget that §1 shows is nearly spent, and
establish first whether the format has any capacity left to allocate.
