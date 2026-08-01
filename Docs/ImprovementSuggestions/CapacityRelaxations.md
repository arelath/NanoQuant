# Capacity Relaxations and Operator-Scope Objectives: New Levers Past the Wall

**Date:** 2026-07-29
**Status:** Idea catalog. Everything here is UNTESTED unless it cites a measurement. Every proposal
names its funding source and its promotion gate per [FormatCapacity.md](FormatCapacity.md) §9.
**Prerequisites:** [FormatCapacity.md](FormatCapacity.md) (the 1.26-BPW wall),
[ErrorAnatomy.md](ErrorAnatomy.md) (where the error lives),
[ReconstructionHeadroom.md](ReconstructionHeadroom.md) (falsification table),
[Doc 41](../41-attention-partition-functional-gate.md) and
[Doc 46](../46-covariance-binary-refinement-screen.md) (operator-level objective lessons; the Doc 46
GPU run is still pending and nothing below depends on its outcome).

---

## 0. Positioning — what this document adds

FormatCapacity establishes that the format is capacity-limited (§1: hard ceiling ~1.26 core BPW,
MLP pinned at 1.185 with ~50% weight error) and proposes one capacity relaxation: the over-complete
joint solve (§3). This document contributes ideas that are in none of the existing catalogs:

1. **§2 — the full relaxation family.** Over-complete rank is one of three ways to keep buying
   binary degrees of freedom past the cap. The other two (input-partitioned and row-partitioned
   fits) need *zero new solver work* and de-risk exactly the initialization problem FormatCapacity
   §3.3 flags as the over-complete solve's biggest risk. They should be arms of the same screen.
2. **§3 — sign-word codebooks**: a way to cut the per-component bit price so the freed bits fund
   §2's components, inside the existing I32 kernel family (LUT decode, llama.cpp IQ-style).
3. **§4 — operator-scope joint refit of the SwiGLU pair**: the "new operator-level selection
   objective" that Doc 41's verdict explicitly asked for, pointed at the operator that carries 72%
   of the KL budget instead of at attention.
4. **§5 — quantization-aware target reshaping**: stop treating the dense weights as immovable.
   At the cap, *all* remaining error is the binary constraint (§1 below); the weights are one point
   in a wide low-loss basin, and nothing in the pipeline currently tries to move that point toward
   the reachable set.
5. **§6 — gauge optimization over exact attention symmetries** (minor, zero bits).
6. **§7 — two recovery-pass corrections** downstream of FormatCapacity §5's KD tail fix.
7. **§8 — binary-factorized tied embedding**: the only lever in either catalog that can change
   total file size by integer factors at the 1B scale.

§9 records the deliberate non-proposals and one warning about an existing catalog item.

---

## 1. Restating the binding constraint: at the cap, 100% of the error is binarity

A fact worth making explicit because three proposals below lean on it. Every production unit's cap
is `min(m, n)`, and at that rank a *real-valued* factorization is exact (take `U = W, V = I` or the
transpose). FormatCapacity §3.1 already observes this in passing; the sharp form is:

> At `r = min(m, n)` the rank constraint is vacuous. The measured 0.500 relative error on
> `down_proj` at r=1151 (ReconstructionHeadroom §8) is contributed entirely by the requirement that
> factor entries be ±1 under three shared scale diagonals. Capacity levers are therefore exactly
> the set of ways to buy more, or better-placed, *binary* degrees of freedom per stored bit.

Two corollaries:

- Any relaxation that strictly enlarges the reachable set at bounded extra bits is a candidate —
  it does not have to look like "more rank."
- **The stacking inversion.** Stacking trades per-member rank ceilings for cross-member sharing.
  For unsaturated units (fused QKV: 1.80× headroom) that trade won and stays won. For saturated
  units it *removes* capacity: stacking two 6912×1152 MLP matrices caps the pair at 1,152 shared
  components where separate fits reach 2×1,152. NextQualityLevers §11 (cross-block stacking for
  partner-less matrices) should be re-scoped to unsaturated units only — as written it would lower
  the ceiling on exactly the matrices FormatCapacity §1.2 shows are pinned against it.

---

## 2. The relaxation family screen (completes FormatCapacity §3.2)

### 2.1 Three moves, one axis

Written for `down_proj` (m=1152 out, n=6912 in) and `gate/up` (6912×1152); the cap fit for each is
r=1152 at 1.185 BPW. All three moves strictly generalize the cap fit — each contains it as a
special case, so fitted-not-forced versions cannot be worse except through optimizer error:

| Move | Construction | What the extra bits buy | BPW (k=2 / first step) |
|---|---|---|---|
| **Over-complete** (FormatCapacity §3) | same U/V grown to r > min(m,n) | more components, all scales still shared | 1.48 at r=1440 |
| **Input partition** (`down_proj`) | split the 6912 input into k slices; independent production fit per slice; runtime sums k factorized products | a *fresh left factor and fresh scale diagonals* per slice (k=2 costs exactly one extra 1152² sign block plus scales) | 1.357 |
| **Row partition** (`gate/up`) | split the 6912 output rows into k slices; independent fit per slice; runtime concatenates | a *fresh right factor* per slice — each output half stops sharing V with the other | 1.357 |

Partition arithmetic for `down_proj`: k = 1/2/3/6 at full per-slice rank gives 1.185 / 1.357 /
1.528 / 2.042 BPW — a smooth capacity dial to 2 BPW using only the existing ADMM, existing scale
fit, and existing flip descent, run k times on smaller matrices (embarrassingly parallel, and each
slice is better-conditioned VRAM-wise than the full matrix).

### 2.2 The falsification record, addressed head-on

Column blocks and per-head splits were falsified at equal bits (0.4398/0.4656 vs 0.4164), and this
construction is the same family. The exemption is not mine: FormatCapacity §10 states that every
one of those falsifications "was measured in the rank-*unsaturated* regime where the counterfactual
was 'spend it on rank'." Past the cap that counterfactual **does not exist** — partitioning (or
over-complete solves) are the only ways to keep buying components. The regime argument is exactly
the new-mechanism argument the falsification table demands before a retest.

Equally important: the partition arms are *not* the falsified sequential-residual construction
(two-stage fitting, 0.4233 vs 0.4164) — each slice fits a disjoint sub-problem of the original
weight, jointly summed only at runtime. There is no residual-structured target, which
ReconstructionHeadroom identifies as the one setting with genuine optimizer fragility. That is
precisely FormatCapacity §3.3's stated risk for the over-complete arm — and why the partition arms
belong in the screen as the de-risked control: if over-complete initialization proves fragile, the
partition curve still answers whether binary error keeps falling past the wall.

### 2.3 Protocol

Extend the FormatCapacity §3.2 screen from one arm to the family, same session, same discipline:

- Matrices: `down_proj` block 12 (anchor with the full-rank datapoint) plus `o_proj` or fused QKV
  (opposite aspect ratio, per §3.2's own caveat).
- Arms at matched total stored bits (±rounding): cap baseline; over-complete r ∈ {1440, 1728};
  input-partition k ∈ {2, 3}; row-partition k ∈ {2, 3} on `gate_proj`; production Fisher-weighted
  objective throughout, production seeds.
- Null: ReconstructionHeadroom's log-linear model E(r) = E_u·exp(−β(r−r_u)), β = 6.22e-4, which
  held to full rank (predicted 0.503, measured 0.500). Plot every arm against it on an
  error-vs-bits axis.
- Screen metric: held-out weighted reconstruction error (screen only — candidate generation).
  Promotion of any arm requires paired held-out splice KL via `application/kl_budget.py`, then a
  complete run, per FormatCapacity §9.

**Decision value.** Four outcomes, all informative: (a) all arms track the null → the format has a
real capacity dial and §6 of FormatCapacity (BPW sweep) gets its mechanism, with the partition arms
shippable first; (b) partitions fall but over-complete flattens → the shared-scale redundancy is
the binding sub-constraint, which redirects solver work; (c) over-complete falls but partitions
flatten → sharing is cheap and the §3.3 init work is worth funding; (d) everything flattens → the
binary family is exhausted at the wall, and effort moves decisively to §3/§5/§7 of this document
and §7 of FormatCapacity. **Effort:** low for partition arms (scripting around production fits);
the over-complete arm carries its known init cost. **Runtime note:** row partition is plan-level
only (two units whose outputs concatenate — the stacked machinery's inverse); input partition needs
a k-way summed dispatch in the runtimes, which is deferred until the screen says the capacity is
real.

---

## 3. Sign-word codebooks: cut the per-component price, spend it on components

### 3.1 Mechanism

Signs are 95.2% of stored bytes, at exactly 1 bit per binary DoF. Constrain every aligned 32-bit
word of packed U and V to come from a per-tensor-type codebook C ⊂ {±1}³² with |C| = 2^k, store
k-bit fixed-width indices, and decode index → I32 word through a table that fits in L1/L2. Per-word
cost falls from 32 to k bits; at k=12 the same sign budget funds ~2.67× the components. This is the
one lever that makes §2's extra components *cheaper* instead of more expensive — the two compose:
codebooked partitioned fits reach k=2 partition capacity near the current 1.19-BPW price.

The repository's own repeated finding — rank breadth beats per-component freedom at this bit rate
(column blocks, rank-group scales, ternary all lost to "spend it on rank") — is an argument that
trading word-level freedom for component count is on the right side of the exchange rate. It is not
a proof; it prices the screen.

### 3.2 Why the two standing objections do not apply

- **"ADMM signs are near-maximum-entropy"** (FormatCapacity §10) is a statement about the
  *unconstrained* solution, and rules out compressing that solution post-hoc — projecting fitted
  words onto a random 4096-word codebook would flip ~8 of 32 signs per word. It does not preclude a
  good *constrained* solution: flip-descent plateaus and multistart indifference show the solution
  set is massively degenerate, and degeneracy is exactly what a fitted codebook exploits. The fit
  must be constrained from the start (alternate word-assignment / codeword-update inside the ADMM
  outer loop — assignment is a word-level generalization of flip descent; codeword update is a
  sign-of-centroid step, k-means-style).
- **"Variable-length codes break the I32 kernel contract"** — fixed-width indices are not
  variable-length. Decode is one gather per word in front of the unchanged two-stage reduction, the
  same pattern mainline llama.cpp already ships in the IQ-family quants. A per-tensor-type codebook
  shared across all 26 blocks makes table storage negligible (k=12 → 16 KiB per type).

### 3.3 Screen

One matrix (`down_proj` block 12), equal total stored bits including codebook and the 16·r mid-scale
growth: free words at r=970 (production 1.0-BPW point) vs codebooked k ∈ {8, 12} at r ≈ {3500,
2500}. Constrained fit, not post-hoc projection. Held-out weighted error as screen; splice KL as
gate. **Effort:** medium-high — this is the largest solver change in this document, and it should
wait until §2's screen shows components past the wall are worth buying. **Risk:** LUT gather cost
on the hot path; measure decode throughput before promotion (FormatCapacity §3.3's runtime rule).

---

## 4. Operator-scope joint refit of the SwiGLU pair (zero bits)

### 4.1 Mechanism

Doc 41's verdict asked for an operator-level selection objective and attention topology was the
wrong place to spend it (13.1% of KL). The MLP operator carries 71.9%, and its structure is a
*product*: h = silu(W_g x) ⊙ (W_u x), y = W_d h. Every current fit treats W_g and W_u as
independent matrix problems; the error of the operator is

```
δh ≈ silu'(g)·u·δg + silu(g)·δu + silu'(g)·δg·δu
```

At ~0.5 relative per-matrix error the fits are far outside the regime where the cross term is
negligible — and, more usefully, the first two terms can be made to *cancel*: refit W_u's factors
against the pair target with the gate error baked in, then refit W_g under the silu linearization,
and alternate. Concretely, the up-pass minimizes `Σ_t ‖silu(ĝ_t) ⊙ (Ŵ_u x_t) − h*_t‖²` — a
sample-space quadratic objective in exactly the form the Doc 46 covariance flip solver already
handles; the coupling enters through per-token-channel weights `silu(ĝ)` and a modified target,
not through new solver mathematics. Gate-pass weights are `silu'(ĝ)·û`.

This is not NextQualityLevers §3 (cross-block error feedback, demoted by Finding H's sub-additivity
— a *depth* claim). Within one operator there is no measured additivity result, the interaction is
architecturally multiplicative rather than residual-additive, and cancellation is engineered rather
than hoped for. An optional third arm may refit W_d on the student's ĥ inputs; that arm *is*
propagation-flavored and should be labeled as such and judged separately.

### 4.2 Screen

Blocks 0/12/24, quantize only the MLP triple, production recipe vs production + 2–3 pair-refit
alternations (bounded flip steps as in Doc 46's 32-step setting). Equal everything — this changes
no stored bits, only sign values and scales. Gate: paired held-out splice KL; add isolated MLP
block-output error per Doc 41's operator rule. **Effort:** low-medium — it is a driver loop around
the existing weighted fitters. **Expected value:** attacks the largest single share of the budget
with the exact class of objective (operator-scope, functional) that the last three probes concluded
is the only trustworthy kind.

---

## 5. Quantization-aware target reshaping (zero deployed bits)

### 5.1 Mechanism

Every fit in the repository treats the pinned dense weights as the fixed target. But §1 says the
residual error at the cap is entirely "W is far from the binary-reachable set" — and W itself is
not sacred: it is one point in a wide basin of dense weights with equivalent function. Nothing
prevents moving W *within that basin* toward the reachable set before (or interleaved with)
factorization:

```
minimize over dense W:   L_preserve(W) + λ · ‖W − Π(W)‖²_weighted
```

where Π is the production factorization and L_preserve anchors function (block-output match on
calibration data for the local version; KD for the global version). Alternate a few descent steps
with re-projection — alternating projection between the low-loss manifold and the reachable set.
The final artifact is Π(W_reshaped): format, bits, runtime all unchanged.

This is distinct from everything tried or cataloged: per-layer STE (falsified) moves the *latents*
against a local Frobenius objective with W fixed; FormatCapacity §7's global QAT trains latents
*after* fitting, still against the original W's function; this moves the *target* so that every
downstream stage — allocation, outliers, tuning, distillation — inherits an easier problem. It is
the standard trick of quantization-aware fine-tuning re-aimed at a factorized format, and it is the
only proposal in either catalog that can reduce the 0.500-at-full-rank number itself.

### 5.2 Screen first, training run later

The cheap falsifiable version needs no training infrastructure: one block's MLP triple, a few
hundred proximal descent steps against that block's output on calibration activations (held-out
gated), re-projection every N steps, then a fresh production fit and paired splice KL vs the
untouched baseline. If block-local reshaping cannot beat the baseline under its own splice gate,
the global version is dead at afternoon cost. If it wins, the global KD-anchored version becomes a
planned training run — **gated behind FormatCapacity §5's KD tail fix** for precisely the Exp 032
reason: maximum parameter freedom must not be pointed at a tail-blind surrogate. **Risk:** teacher
drift — L_preserve must be evaluated against the *original* teacher on held-out data, with
rollback. Compute is the real price of the global version; the screen is cheap.

---

## 6. Gauge optimization over exact attention symmetries (zero bits, minor)

The v/o pair carries an exact continuous symmetry: for any invertible R (256×256, one per KV
head), replacing W_v ← R·W_v and each per-head o-block W_o,h ← W_o,h·R⁻¹ leaves the operator
*exactly* unchanged — attention mixes v rows with scalar weights, so R commutes through. The
current format fits whatever gauge the checkpoint happened to ship. Optimizing R (identity-init,
orthogonal × diagonal parameterization for conditioning) to minimize the pair's weighted binary-fit
error is free capacity in the only mathematically exact sense available.

Why the rotation falsifications do not apply: ReconstructionHeadroom §9.1 and Doc 45 imposed
*random/incoherence* rotations chosen to flatten structure, forced onto the format, and they
destroyed the row-magnitude structure `diag(post)` exploits. Here R is *fitted to the format's own
objective* with identity as a reachable point — it can discover "do nothing" and can only improve
modulo optimizer noise. The honest ceiling is small: v/o influences ≤ 14.9% of type-summed KL
(o) plus v's share of QKV, and ErrorAnatomy names v as worst-fit-and-most-exposed, which is the
one reason this might punch above its share. The q/k analog is nearly blocked by Gemma-3's QK-norm
(only per-head orthogonal transforms survive, modulo re-fitted norm gains) — scope it out of v1.
**Screen:** blocks 0/12/24, fit R per block by differentiating through a relaxed fit or by
coordinate search, refit production factors, paired splice KL. **Effort:** low. Priority: last
among the zero-bit items; run it when a GPU session has slack.

---

## 7. Recovery-pass corrections beyond the KD tail fix

Both items assume FormatCapacity §5 (full-normalizer KD) lands first; both are cheap arms on top of
that re-run rather than new campaigns.

- **7.1 Trajectory-anchored global KD.** The measured KD regression (pre-KD ppl 415.16 → post-KD
  453.57 while the cached surrogate improved, per [GPT5_6.md](GPT5_6.md)) is consistent with
  tail-blindness, but a 26-block ~1-BPW student also gives end-only losses a long, high-variance
  credit path. Cache teacher residual-stream states per block (the splice harness already produces
  them) and add an annealed per-block anchor `Σ_b w_b·‖s_b − t_b‖²` with w_b from ErrorAnatomy
  Finding I sensitivities. Run as a paired arm against tail-fixed KD alone; keep whichever wins
  held-out. Zero deployed bits, small implementation.
- **7.2 On-policy blend, honestly scoped.** Mixing student-sampled continuations scored by the
  teacher (GKD-style) targets generation drift — a failure mode the pinned gates (teacher-forced
  ppl, loglikelihood-ranked tasks) barely observe. It therefore cannot be promoted through the
  standard gates and should be framed as a deployment-quality experiment for the interactive/chat
  path (the Doc 36 Qwen3 thinking-mode work is its natural home), not as a baseline-beating
  candidate. Listed to pre-empt it being proposed later as one.

---

## 8. Binary-factorized tied embedding (the total-file lever)

At the 1B scale the decoder is a minority of the payload: Doc 38 puts Gemma-3-1B at 1.25 core vs
3.44 whole-model BPW, and the shipped artifact carries a Q8_0 embedding of roughly 300 MiB against
~90 MiB of NanoQuant decoder tensors. GPT5_6 proposed Q8 (done) and reallocation; FormatCapacity
footnotes the topic into its §6 sweep. Neither proposes the direct step: **run the format on the
embedding itself.**

The embedding is a 262,144×1152 matrix; the format applies as-is with rank capped at 1152. The
numbers are dramatic because the matrix is so tall: at r=1152, signs cost ≈38 MiB against ≈320 MiB
for Q8_0 — and the same §2 relaxation family applies along the 1152 axis if full-rank binary error
(~0.5, unknown tolerability for embeddings) proves too coarse. The natural design has three tiers:

- top-N frequent tokens (N ≈ 8–16k from calibration statistics) kept exact in Q8 rows — the row
  analog of the existing exact outlier columns, ~10–20 MiB;
- the remaining rows factorized under a **token-frequency-weighted objective** (frequency is the
  correct row importance for both the lookup and the tied-head logit direction — a rare token's
  row error perturbs logits mainly through the softmax normalizer);
- the tied head evaluated with the existing two-stage kernel (for a matrix this tall, decode cost
  r(m+n) ≈ r·m is essentially the dense cost — no penalty).

A plausible landing zone is a 400 MiB artifact becoming ~170–200 MiB at matched decoder quality —
integer-factor compression no decoder-side idea can reach at this scale. Risks are real and
measurable: rare-token perplexity, the logit-scale sensitivity of the tied head, multilingual/tail
degradation; gate on wikitext ppl plus a rare-token-stratified eval. Per GPT5_6, run as a
**separate fixed-total-file experiment**, never mixed into 1-BPW decoder comparisons. This is also
the cleanest place to *practice* §2's relaxations: the embedding tolerates experimentation without
touching the decoder contract. **Effort:** medium (planner + GGUF surface for one new tensor
family). Scale note: this lever shrinks as models grow (Doc 38's columns converge at 70B) — it is
specifically the small-model win, which is the project's current arena.

---

## 9. Deliberately not proposed, and one warning

| Idea | Why not |
|---|---|
| Anything in FormatCapacity §3–§8 | Already cataloged there; §2 here extends §3.2 rather than repeating it. |
| Entropy coding, variable-length sign codes | Still rejected for the Doc 19 reasons; §3 here is fixed-width and kernel-shaped precisely to stay inside that contract. |
| More attention topology enumeration | Doc 40/41 closed it; §6 here changes basis within a fixed topology, not the partition. |
| Sparse/column residual patches, ternary, rank-group scales, per-head splits at fixed bits in the unsaturated regime | Falsified; §2 retests the partition family only under the saturation exemption FormatCapacity itself records. |
| Incoherence/random rotations | Falsified three times; §6 is fitted-not-forced and identity-reachable, which is the distinction that matters. |
| Bias correction / low-rank patches as add-ons | D3/D5 measured regressions; §4 and §5 move the same error mass by changing the fit, not by appending correctors. |

**Warning (actionable):** NextQualityLevers §11 — cross-block stacking for partner-less matrices —
predates the capacity finding. Per §1's stacking inversion it would *lower* the physical ceiling of
already-pinned MLP units. It should be struck or explicitly re-scoped to unsaturated units before
anyone spends a probe on it.

---

## 10. Suggested order

1. **§2 relaxation-family screen** — subsumes FormatCapacity §11's item 1 and adds the de-risked
   partition arms; the cheapest decisive capacity measurement available.
2. **§4 SwiGLU pair refit screen** — zero bits, largest KL share, reuses the Doc 46 solver; can run
   while §2 waits for GPU time.
3. **§5 block-local target reshaping screen** — one afternoon; its global version waits for the KD
   tail fix regardless of outcome.
4. **§6 v/o gauge probe** — slack-time item.
5. **§7.1 anchored-KD arm** — attach to the FormatCapacity §5 re-run when it happens.
6. **§3 sign-word codebooks** — only after §2 shows components past the wall are worth buying;
   largest solver investment here.
7. **§8 embedding factorization** — independent track; start whenever total-file size becomes the
   headline metric (TODO.md's 8B comparisons will need the honest-both-columns reporting anyway).

The through-line continues FormatCapacity's: the wall is real, so stop bidding up the price of the
last bits inside it — buy degrees of freedom past it (§2, §3), make the ones already bought
cooperate (§4, §6), move the target closer (§5), and spend the recovery passes on objectives that
can see the failure they are supposed to fix (§7).
