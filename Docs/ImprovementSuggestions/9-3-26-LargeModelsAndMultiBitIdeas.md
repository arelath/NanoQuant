# Large Models at ≤1 BPW and 2+ BPW Recovery: Review and New Ideas

**Date:** 2026-09-03
**Status:** Review plus idea catalog. Part I is a reading of retained evidence; every number cites a
retained result. Everything in Part II is UNTESTED unless it cites a measurement. Each proposal names
its funding source and its promotion gate per [FormatCapacity.md](FormatCapacity.md) §9 and
[Doc 81](../81-experiment-methodology-guardrails.md).
**Prerequisites:** [8-12-26-Theory-Test-Ledger.md](8-12-26-Theory-Test-Ledger.md) (what was just
falsified), [FormatCapacity.md](FormatCapacity.md) (the 1.26-BPW wall), [ErrorAnatomy.md](ErrorAnatomy.md)
(where the error lives), [Doc 38](../38-full-rank-bpw-estimates.md) (full-rank BPW per model),
[Doc 79](../79-experiment-042-transfer-failure-analysis.md) (the KD-horizon defect).

**The two goals this document serves**

1. Very large models (8B–70B class) loading at **≤1 BPW**. At 1.0 core BPW a 70B decoder is ~8.0 GiB,
   which is the first point at which such a model fits a 12 GiB consumer GPU at all.
2. A **2+ BPW** operating point that lands materially closer to the source model than the current
   ~1-BPW recipes do.

The short version of the review: the 1-BPW recipe has been tuned exhaustively on the one model size
where 1-bit compression is hardest (≤1B parameters), the best 1-BPW result comes from the
functional-objective line (tail-aware KD plus adaptive correction, Experiment 048), the 2-BPW runs
did not use that line, and the project has never measured itself against the 1–2 bit formats that
already ship in llama.cpp. The most promising moves are therefore not new factorizers: they are
(a) fixing the 2-BPW recipe's distillation, (b) getting one ≥8B model through the pipeline so quality
levers are tuned in the regime the goal actually names, (c) spending the "second bit" on a
representation chosen for rate–distortion efficiency rather than convenience, and (d) making the
end-to-end stage the capacity lever it already proved it can be.

---

## Part I — Review of current attempts

### 1. Scoreboard

Gemma 3 1B, protocol-matched 64×128 WikiText-2 and six tasks. Task means are recomputed from the six
rows of each retained quality report (1,000 rows unless noted). BF16: perplexity **96.46**, six-task mean
**0.6057**.

| Experiment | Effective BPW | WikiText ppl | Ratio | Six-task mean | Global KD objective | What it changed |
|---|---:|---:|---:|---:|---|---|
| 022 D2 exact-unit allocation | 1.0245 | 228.55 | 2.37× | 0.4692 (200-row) | conditional top-64, 2,048 steps | KL-calibrated rank allocation |
| 054 functional binary LR | 1.0244 | 203.84 | 2.11× | 0.4618 | conditional, 2,048 | binary latent LR 3e-5 |
| 055 over-complete rank + tabu | 1.0244 | 285.50 | 2.96× | — | conditional, 2,048 | **rejected** |
| 056 physical cap + 1.5× + tabu | 1.0245 | 253.15 | 2.62× | 0.4570 | conditional, 2,048 | worse than 054 |
| 057 + product-k16 codebook | 1.0245 | 248.37 | 2.57× | 0.4625 | conditional, 2,048 | tiny gain over 056 |
| 058 + 13 INT8 down columns | 1.0373 | 223.21 | 2.31× | 0.4603 | conditional, 2,048 | +0.013 BPW |
| 042 mass-floor correction | 1.0245 | 171.87 | 1.78× | ≈0.466 | conditional, 2,048 + correction | correction line |
| 044 tail-aware KD | 1.0245 | 180.36 | 1.87× | 0.4567 | **top-64 + tail (0.5), 256** | objective fix |
| **048 tail-aware + adaptive correction** | **1.0245** | **153.89** | **1.60×** | **0.462** | **top-64 + tail, 256 + 64 corr.** | **best 1-BPW result** |
| **060 best-methods 2 BPW** | **1.9634** | **115.67** | **1.20×** | **0.5022** | conditional top-64, 2,048 | physical cap + 8.8% INT8 columns |

Other models and scales:

| Model | Experiment | BPW | ppl BF16 → candidate | Six-task mean BF16 → candidate |
|---|---|---:|---|---|
| Gemma 3 270M | 021 (1 BPW) | 1.02 | 194.05 → 1,141.16 (5.9×) | 0.4993 → 0.3983 (200-row) |
| Gemma 3 270M | 059 (2 BPW) | 1.975 | 194.05 → 218.33 (1.13×) | 0.4993 → 0.4237 |
| Llama 3.2 1B | 025 | 1.022 | 36.86 → 116.98 (3.17×) | 0.6092 → 0.5067 |
| Qwen3 0.6B | 028 | 1.028 | 55.17 → 292.26 (5.30×) | 0.5608 → 0.4483 |
| Gemma 3 4B | 003 | ~1.0 | "materially degraded" | — |
| Gemma 3 12B | 008 | — | stopped at 23/48 blocks | — |
| Llama 3 8B, Qwen3 8B | 027/029 | — | never run | — |

Two readings of the table matter for the goals:

- **The 1-BPW task metric is flat.** Across a perplexity range of 154–253, the six-task mean sits in
  0.457–0.469. Perplexity is responding to distillation and calibration; task capability is pinned near
  a floor for a 1B model at one bit. Even the 2-BPW run reaches only 0.502 of 0.606. This is the strongest
  single signal that the small-model regime is the wrong place to keep tuning.
- **The best 1-BPW and the 2-BPW recipes diverged on the one lever with the largest measured effect.**
  Experiment 048's 153.9 comes from the tail-aware objective plus adaptive correction, confirmed in the
  ledger as R1. The 2-BPW launchers (059/060, committed the same day 048 published) explicitly disable
  tail-aware distillation and run the conditional top-64 objective for 2,048 steps, which
  [Doc 79](../79-experiment-042-transfer-failure-analysis.md) showed is the horizon at which that
  objective drives probability mass off the teacher's support and inflates block 25. The 2-BPW numbers
  are therefore a lower bound on what the 2-BPW representation can do.

### 2. What is confirmed and should be kept

Retained, protocol-matched positives, best first:

- **Tail-aware top-k KD at a short horizon, plus adaptive capability correction with a baseline escape
  path** (043/044/047/048, ledger R1). 228.6 → 153.9 at identical bits. The mechanism is understood
  (Docs 74/75): conditional top-64 KD is blind to tail mass.
- **Exact-unit KL-calibrated rank allocation (D2)** (022, ledger claim-status "Supported"): 274.9 → 228.6
  over the reconstruction-only allocator at ~equal BPW.
- **Stacked shared-input QKV plus Fisher importance weighting** (adopted 2026-07-19; −36–44% functional
  error at equal bits in activation space; end-to-end KL 16.8 → 4.7 nats in the anatomy harness).
- **A separate, larger binary-latent learning rate in functional block tuning** (054): 235.3 → 203.8;
  moves ~0.8% of signs; weighted matrix error gets *worse* while block loss and KL improve.
- **Selective coordinate binary QAT with a Hamming ball and atomic rollback** (062, ledger R3): on blocks
  12 and 25, KL −0.012/−0.066 and NLL −0.023/−0.120 at 0.17–0.22% sign movement, reproduced on
  untouched rows; block 0 fails and must roll back. Not yet composed into a full run.
- **Foldable MLP scale placement at zero bits** (Docs 69/70): −5.39% perplexity on 5 of 26 blocks.
- **Rate-matched calibration-weighted INT8 outlier columns** (057→058): 248.4 → 223.2 for +0.013 BPW,
  with a small task-mean loss.
- **Operational:** one-block subprocess isolation contains the process-local CUDA contamination
  (054/056/058, ledger R7a); adaptive memory planning completes 270M/1B and selects CPU offload for 4B
  (Doc 35).

### 3. What is rejected and should not be re-run without a new mechanism argument

The [ledger](8-12-26-Theory-Test-Ledger.md) closes the 8-12 review's theories; this table is the union
of that ledger with the format falsifications, so nobody re-derives them.

| Rejected | Regime of the test | Evidence |
|---|---|---|
| Direct binary search tiers, tabu, basin hopping, larger windows | static objective, equal bits | 052/053, BasinSearchStrategyReview |
| Dense covariance refinement as a production stage | pre-tuning and post-refit | 033 (+19.3% ppl), 034 (+5.5%) |
| Hadamard / incoherence rotations, both-side and input-only | equal bits | ReconstructionHeadroom §9, Doc 45 |
| Over-complete rank as an allocator option | 1.0 BPW equal budget, ADMM-only response | 055 (+40% ppl) |
| Reciprocal MLP fusions; attention partitions other than QKV | equal bits | Doc 84, Docs 40/41 |
| Global bias correction; unfunded rank-4 patch | 1.0 BPW equal budget | Doc 33 D3/D5 |
| 2×2 grouped middle core | equal rate | ledger R6 |
| Structured (row-segment, tile) INT8 sidecars | equal bits | 064/065 |
| Per-token tail weights, multi-band tails, top-p targets, hard-label CE blend | matched 32 steps | 063 |
| A universal cheap functional proxy for allocation | nine owners | 064 (Spearman 0.02 with KL) |
| Ternary factors, column blocks, rank-group scales, cyclic scale banks, multi-stage residual binary | **1.0 BPW, rank-unsaturated**, plain or Fisher-weighted objective | ReconstructionHeadroom, 049 |
| Residual-product codebook; learned GF(2) code; functional codebook payload tuning | equal bits | Docs 87, 90, 98, 99, 100 |

The last row is the important one for the 2-BPW track. Every entry in it was measured where the
counterfactual was "spend the same bits on more binary rank". Above the physical rank cap that
counterfactual does not exist (FormatCapacity §10 records this exemption). The 2-BPW recipe is *already*
at the cap on every MLP matrix, so those candidates are legitimately re-testable there, but only there.

### 4. Diagnosis: why progress stalled, in five parts

**4.1 All quality tuning happened in the least compressible regime.** Every completed quality run is
≤1B parameters. The 1-bit PTQ and QAT literature is consistent that relative degradation shrinks with
scale, and the format's own geometry improves with scale: the physical cap for a 1B MLP is 1.19× the
1-BPW rank (FormatCapacity §1.1), whereas Doc 38 puts full-rank core BPW for 8B–70B at 1.28–1.45, and
per unit the full-rank price is `1 + min/max` (1.29 for a 70B MLP matrix, 2.0 for its square attention
projections), so a 1-BPW allocator there has roughly 1.25–2× headroom per unit instead of being pinned
at 1.19×. Meanwhile the goal
names 8B–70B models, and no such model has ever completed. Recipe levers found at 1B may not be the
binding ones at 27B (the 270M and 1B KL depth/type profiles already disagree, D2-findings Finding 3).

**4.2 The proxy problem is solved by evidence, not yet by process.** The repository has now falsified
three generations of local proxies (Frobenius, weighted Frobenius, static held-out KL of isolated
candidates, cheap block-output scores). Every remaining confirmed lever is a *functional* one that
optimizes the deployed model against the teacher. The implication for planning is blunt: new
representation ideas should be screened by installed-candidate splice KL from day one, and the
day-one screen should be sized (48×512, paired bootstrap) so a 1–3% effect is decidable.

**4.3 The second bit is being spent on a convenience representation.** At 2 BPW the recipe fills the
gap between the physical cap (1.19–1.93 BPW depending on aspect ratio) and 2.0 with whole INT8 input
columns. On Gemma 3 1B block 1 that is 8.8% of columns at 8.5 bits each: `gate_proj` lands at 1.894 BPW,
`o_proj` at **2.586 BPW** (rank 992 factors plus 101 INT8 columns) for a weighted error of 0.06. Two things
are wrong with this from a rate–distortion standpoint. First, a full-rank sign factorization of a
near-square matrix costs `1 + min/max ≈ 1.9` BPW for ~0.5 relative error, whereas a plain 2-bit
quantizer of a Gaussian source reaches ~0.25 relative error at 2.0 bits, so for square attention
matrices the factorization is the wrong format above ~1.5 BPW. Second, the exact column is an 8-bit
code for a residual whose per-entry distribution is Gaussian (kurtosis 3.16), which is where vector or
trellis codes at 1–2 bits are near-optimal and scalar 8-bit codes are wasteful. Nobody has measured the
alternative because no rate-matched menu for the second bit exists.

**4.4 There is no external baseline.** A grep of `Docs/`, `tools/`, and `src/` finds llama.cpp's
`IQ1_S`, `IQ1_M`, `IQ2_XXS`, `Q2_K`, and the QTIP/AQLM/PV-tuning family only as literature citations. The
project builds and runs a modified llama.cpp with a protocol-matched GGUF quality runner, so quantizing
the same pinned Gemma 3 1B to `IQ2_XXS` (2.06 BPW) and `IQ1_M` (1.75 BPW) and scoring it under the same
64×128/six-task protocol is a few hours of work. Until that exists the 2-BPW result (1.20× perplexity,
83% task retention at 1B) cannot be judged, and the 1-BPW result cannot be positioned either.

**4.5 Distillation is starved.** The calibration set is 256 × 2,048 = 0.52M tokens; global KD runs
2,048 steps of ≤512 selected tokens, about 1M token-targets, with ~1M trainable scale/norm parameters
and immutable signs. The 1-bit literature that reaches usable quality (OneBit, BitDistiller,
EfficientQAT, PV-tuning) uses 10–1,000× more distillation tokens and trains the discrete weights. The
repository's own evidence points the same way: the functional binary LR (054) and coordinate QAT (062)
are the only confirmed *capacity* levers, and the tail-aware objective is the only one under which
longer training kept improving full KL (Doc 75). Data and horizon scaling of tail-aware KD with
selective sign movement is the one untested combination whose ingredients are all individually
confirmed.

A sixth, operational item: **process-local CUDA contamination** (non-finite forwards after a rollback,
054/056/058) is contained by one-block subprocesses but not root-caused. A 48–80 block run at 8B–70B
multiplies the exposure; the containment costs wall time and makes multi-day runs brittle.

### 5. Verdict: what has the most promise

Ranked by (confidence × expected effect) / cost, for the two stated goals:

1. **Re-distill the retained 2-BPW states (059/060) with the 048 objective and correction policy.**
   Zero representation change, an hour of GPU, high confidence of a large perplexity gain (the same change
   moved 1 BPW from ~230 to 154). This should happen before any other 2-BPW work because it resets the
   2-BPW baseline (Part II, B1).
2. **Run one ≥8B model through the 1-BPW recipe and publish the scaling curve.** This is the
   goal-defining measurement; everything downstream of it is speculation until it exists (A1).
3. **Long-horizon, larger-data tail-aware KD with selective coordinate binary QAT** as the capacity
   lever for ≤1 BPW at scale (A3). All ingredients are confirmed; the composition is not.
4. **A rate-matched menu for the second bit**, including a trainable real-valued low-rank patch and an
   aspect-ratio-gated hybrid with standard 2-bit tensors, gated by installed splice KL and benchmarked
   against `IQ2_XXS` (B2–B6).

Everything else in Part II is second tier or a screen that de-risks one of these four.

---

## Part II — New ideas

Format for each: mechanism, why it is not already falsified, funding, screen and gate, effort,
expected value. "Screen" means candidate generation; promotion always requires paired held-out splice
KL and then a complete run per FormatCapacity §9.

### A. Track 1: very large models at ≤1 BPW

#### A1. The scaling-curve campaign (decision experiment)

**Mechanism.** Run the fixed Experiment-048 recipe (D2 allocation, stacked QKV, binary LR 3e-5,
tail-aware KD + adaptive correction) at 1.0 BPW on Gemma 3 4B, Gemma 3 12B, and Llama 3 8B, and report
perplexity ratio and task retention against size. This answers whether the goal is a quality problem or
an engineering problem: if the 12B ratio is ≤1.3× the way the literature suggests, the ≤1 BPW track is
an execution track and the 1B quality campaign can stop.

**What blocks it today.** (i) `cpu_offload` is a planner decision and a few guards in
`resident_quantization.py`; it refuses inline quality and model-level distillation
(`resource_planning.py:358`). Global KD therefore needs an offloaded student: the immutable-sign
student can be held as packed signs plus fp32 scales (≈1.5 GB for 12B), so this is a loader change, not
a memory wall. (ii) D2 exact-unit profiling is 336 full-model splice forwards at 12B; use the existing
`EXACT_OR_TYPE_BLOCK` granularity, or profile every unit on a 16×512 slice and only the top-KL third on
48×512. (iii) Streaming components M5.3–M5.8 exist but M5.9/M5.11/M5.18 are open; 12B fits host RAM with
offload, 27B+ needs the streaming executor.

**Cost estimate.** Per-block ADMM cost scales with `m·n·r`: 12B `down_proj` (3840×15360, r≈3840) is
~25× a 1B `down_proj`; tuning scales with block parameters (~8×). From the 060 profile (ADMM ~10 s per
unit, tuning ~230 s per block at 1B) that is ~50 min per block, ~40 h for 48 blocks on the 12 GiB card,
plus profiling and KD; ~1 day on an A100 via the existing RunPod bootstrap. 70B is ~200× the ADMM
cost and needs A2 and A6 before it is practical.

**Funding.** None (1.0 BPW). **Effort:** medium engineering, high compute. **Gate:** complete run,
strict validation, GGUF quality, and a published curve. This is the first thing on the large-model
track and it should run on RunPod rather than the workstation.

#### A2. Effort allocation by measured KL share

**Mechanism.** ErrorAnatomy Finding G: blocks 0–10 carry 65% of KL, blocks 18–24 carry 7.8%. The
pipeline spends identical ADMM iterations (800–1,200), tuning epochs, and profile resolution on every
block. Make the per-block ADMM iteration count, factorized-tuning epochs, and rank-probe fidelity a
function of the block's measured KL share (or of depth as a prior before the profile exists), with a
floor. Also apply the sub-additivity finding: since block KLs compose ~0.9-additively, late blocks can
be factorized with a 200-iteration schedule and tuned for two epochs without visible end-to-end cost.

**Why not falsified.** Nothing has varied per-block effort; the 054/055 iteration changes were global.

**Screen.** On the retained 048 state, refactorize blocks 18–24 at 200 ADMM iterations and 2 tuning
epochs, splice, paired KL on 48×512. Success is a KL delta whose interval contains zero.
**Effort:** low. **Expected value:** ~1.5–2× wall-time reduction at 12B+ with no quality change, which
is what makes A1 repeatable and 27B+ feasible.

#### A3. Long-horizon tail-aware KD with selective coordinate binary QAT (the capacity lever)

**Mechanism.** Three confirmed pieces composed for the first time:

1. the tail-aware objective (top-64 + tail, mass weight 0.5), under which full KL kept improving
   through the longest tested horizon (Doc 75) rather than collapsing;
2. 10–50× more distillation tokens from the existing pinned mixture plus the teacher-dataset builder
   (`tools/build_teacher_dataset.py`), with the top-64 teacher cache sized accordingly (8M tokens ×
   64 × 6 bytes ≈ 3 GB);
3. selective coordinate binary QAT (062): one owner family at a time, FP32 shadow margins, projection
   into a 0.8% Hamming ball after every step, evaluated through hardened BF16 signs, accepted only on a
   disjoint joint NLL/KL gate, rolled back otherwise, then scales refit.

Run as a data-scaling ablation at fixed objective: {0.5M, 4M, 16M} tokens × {256, 1,024, 4,096} steps
on the 048 pre-KD state, immutable-sign control versus coordinate-QAT arm.

**Why not falsified.** 042 showed 2,048 conditional steps regress, which is the objective, not the
horizon (Doc 79). R3 confirmed coordinate QAT on two of three blocks but never composed it or scaled
data. R7b's negative was about tuning checkpoint selection, not KD data.

**Funding.** Zero deployed bits. **Effort:** medium (a KD campaign driver; the probe tool exists).
**Gate:** held-out NLL and full KL on the reserved C4 slice, then the six-task suite; temperature-fitted
and raw reported separately (Doc 82) so calibration gains are not mistaken for capability.
**Expected value:** this is where the literature's 1-bit results come from; at 1B it may move the task
floor for the first time, and at 8B+ it is likely the dominant lever.

#### A4. Rank annealing: fit and tune at the physical cap, then prune to budget

**Mechanism.** Instead of factorizing directly at the 1-BPW rank, factorize and functionally tune every
unit at its physical cap (the representation 060 already produces), then remove components down to the
1-BPW budget by a functional criterion (leave-one-component-out block loss, or `|scale_mid| ×` measured
unit sensitivity), refit scales, and run one short functional retune. The hypothesis is that the tuned
full-rank basin is a better starting point than a fresh ADMM solve at the target rank, because
functional tuning has already moved ~1% of signs toward what the block needs.

**Why not falsified.** 055 tested *allocating* over-complete rank from ADMM-only response curves and
shipping it. This keeps the deployed rank at the 1-BPW plan and uses extra rank only during
construction. It is the "train large, prune" pattern, not the "buy more rank" pattern.

**Screen (cheap, the artifacts exist).** Take the 060 pre-KD state, drop the INT8 columns on blocks
0/12/25, prune MLP components from 1,152 to the 054 plan's ranks by the functional criterion, refit
scales, retune 2 epochs, splice against the 054 factors at exactly equal bits. **Effort:** low.
**Expected value:** uncertain; a positive result changes the pipeline's construction order for every
model size, a negative one closes the idea in one afternoon.

#### A5. Nested (prefix-ordered) rank components: one artifact, a runtime BPW knob

**Mechanism.** Order the rank components of every unit so that any prefix of components with its own
`scale_mid` prefix is itself a good approximation, and store a small table of per-unit prefix cut points
for {0.75, 1.0, 1.25} BPW. The loader then materializes the chosen prefix. Fitting: solve at the
largest rank, order components by cumulative functional gain, and re-solve `scale_mid` and boundary
scales for each cut (the scales are cheap; the signs are shared). The sequential-residual construction
that ReconstructionHeadroom flagged as fragile is *not* used: this is one joint solve plus ordering.

**Why it is worth having for this goal specifically.** "Load at 1 bit or less" becomes a load-time
choice from a single shipped file, and the same artifact serves the 12 GiB card at 1.0 and a 24 GiB
card at 1.25 without a second campaign. Cost is one extra small table per unit; the packed layout is
rank-agnostic already (Doc 19).

**Screen.** Fit at 1.25 BPW rank, cut to 1.0, compare splice KL to a direct 1.0 fit on blocks 0/12/25.
Expect a small loss at the cut; the question is whether it is under ~2% KL. **Effort:** medium
(planner, ordering step, loader). **Priority:** below A1–A4; do it once A1 shows the operating point
range that matters.

#### A6. Cheaper factorization for large matrices

Three measurements, all cheap, all needed before 27B+ is practical:

- **Iteration budget curve.** ADMM at {100, 200, 400, 800, 1,200} iterations on 1B blocks 0/12/25,
  followed by production functional tuning; report post-tuning block loss and splice KL, not
  reconstruction error. The 054 evidence that tuning moves ~1% of signs anyway suggests the marginal
  ADMM iterations past ~200 buy little that tuning does not recover.
- **Warm start from a randomized-SVD SVID init** instead of the fresh continuous relaxation, so the
  cubic penalty schedule bites earlier; measure the same way.
- **Forward-only importance at scale.** Fisher calibration needs backward through the full teacher;
  for 27B+ on one GPU that requires streamed forward/backward (M5.11, unimplemented). The fallback is
  forward-only `E[x²]` importance, which is the objective the July activation-space experiments used.
  Measure the 1B quality gap between raw-Fisher and `E[x²]` importance now, so the 70B plan knows what it
  costs.

**Effort:** low each. **Expected value:** decides whether 27B+ is a workstation-plus-RunPod job or a
multi-GPU job.

#### A7. Large-model specifics to design for now

- **GQA k/v.** Llama 3 8B/70B and Qwen3 8B+ have 1,024-row k/v against 4,096–8,192-wide inputs. Standalone
  k/v at full rank costs ~1.14 BPW; fused QKV lifts the group cap to the input width. Fused QKV should be
  mandatory, not optional, for these families.
- **Square attention matrices** (`q`/`o` at 8192²) cost 2.0 BPW at full rank; at a 1.0 target they sit at
  half rank with 1.9× headroom, so allocation has real room. Expect the D2 profile to move bits toward MLP
  as at 1B, but measure it (D2-findings Finding 3).
- **Q8 GGUF teacher for logit caching.** The teacher-dataset builder already drives Q8 GGUF teachers
  through llama.cpp; caching top-64 targets and log-normalizers from a Q8 teacher halves teacher memory
  and uses the fast runtime. Validate on 1B that Q8-teacher targets do not change KD outcomes.
- **Evaluation.** Add a 2,048-context perplexity and an MMLU-style capability eval before publishing
  large-model claims; the 64×128 protocol and six tasks were chosen for 270M–1B and will saturate.

### B. Track 2: 2+ BPW, closer to the source model

#### B1. Re-distill the retained 2-BPW states with the confirmed objective (immediate)

**Mechanism.** The 059/060 pre-KD frozen states are retained and hash-valid. Run global distillation on
them with `top_k_tail`, tail-mass weight 0.5, 8 × 32 steps, then the 048 adaptive correction with its
selection/confirmation slices from the registry, exactly as R1 accepted. Compare on the identical
protocol.

**Why.** Section 4.3 and the scoreboard: the best 1-BPW result and the 2-BPW recipe differ on this
lever alone, and the 2-BPW runs used the horizon Doc 79 identifies as harmful. **Funding:** none.
**Effort:** trivial. **Gate:** the standard quality benchmark; expect a large perplexity move and a
smaller task move. This also produces the honest 2-BPW baseline that B2–B6 must beat.

#### B2. External baselines at 1.5–2.6 BPW

**Mechanism.** Quantize the pinned Gemma 3 1B and 270M BF16 GGUFs with `llama-quantize` to `IQ1_S`
(1.56), `IQ1_M` (1.75), `IQ2_XXS` (2.06), `IQ2_S` (2.5), and `Q2_K` (2.6) using an imatrix built from
the same calibration mixture, keep `token_embd` at Q8_0 as the NanoQuant export does, and score them
with the existing protocol-matched GGUF runner. Report core BPW and total file bytes for every arm.

**Why.** Without it, neither the 1-BPW nor the 2-BPW result is positioned, and the 2-BPW track cannot
tell whether the factorized format is the right tool above ~1.5 BPW. **Effort:** low (tooling exists).
**Decision rule.** If `IQ2_XXS` beats B1's 060 at ≤ equal bytes, the 2-BPW track pivots to the hybrid
in B6 rather than to more work inside the factor format.

#### B3. A rate-matched menu for the second bit

**Mechanism.** Hold the binary factors at the 060 physical-cap plan and compare, at exactly the same
total bits per unit, six ways to spend the remaining ~0.7 BPW on the MLP matrices and ~0.7–1.4 BPW on the
attention matrices:

| Arm | Representation of the extra budget | Notes |
|---|---|---|
| a (incumbent) | 8.8% whole INT8 input columns | 060 |
| b | ~17% INT4 columns, per-column scale | same bits, twice the coverage |
| c | rank-k real-valued patch, fp16 | k ≈ 43 on gate/up/down |
| d | rank-k patch, INT8 values | k ≈ 87 |
| e | second binary stage on the Fisher-weighted residual at ~0.5× rank | the multi-stage construction, retested under the cap exemption |
| f | ternary right factor at full rank (1.58 bits/entry) | cost 1.58 × 1.185 ≈ 1.87 BPW on MLP |

Fit each on blocks 0/12/25 from the retained residual, dense splice, paired KL on a fresh 48×512
registry slice; the winner gets a resident run.

**Why not falsified.** Arms e and f were falsified only in the rank-unsaturated regime (Part I §3, last
row). Arms c/d were falsified at 1.0 BPW where the funding came from binary rank (D5); here the funding
is the outlier budget. Arm b is new. **Effort:** low-medium (all fitters exist; the splice harness
exists). **Expected value:** high information per GPU-hour; ErrorAnatomy Finding B (a rank-16 real
correction removes 25–54% of functional error energy, rank-64 removes 46–76%) predicts c/d compete with
or beat a on every unit type, and the column arm's advantage is that it targets Gemma's axis-aligned
outlier channels, which the patch can also represent.

#### B4. The low-rank patch as the trainable capacity of global KD

**Mechanism.** If arm c or d survives B3, make the patch the primary parameter group of global
distillation. Unlike signs, the patch is continuous: no STE, no Hamming ball, no shadow latents. This is
the LoftQ/LQ-LoRA structure applied to a binary base. Patch rank can then be allocated per unit by
measured functional utility using the existing exact-cost allocator, with the binary base fixed.

**What exists.** `LowRankPatchConfig`, `application/low_rank_patch.py`, `patch_bit_cost`, and packed
runtime support (`runtime/packed.py` carries `patch_left`/`patch_right`); global distillation already
selects patches as trainable parameters. The GGUF converter and llama.cpp kernel path for patches need
verification. **Funding:** the second-bit budget. **Effort:** low for the KD change, medium for the GGUF
path. **Gate:** B1's rebaselined 060 on the same protocol, at equal bytes.

#### B5. A residual lattice or trellis code instead of exact columns

**Mechanism.** After factorization the per-entry residual is Gaussian (kurtosis 3.16), which is
precisely the source for which E8-lattice and trellis-coded quantizers reach within ~0.1 bit of the
rate–distortion bound. Code the Fisher-weighted residual of each MLP matrix at 1 bit per weight with an
IQ-style LUT code; total ≈ 2.2 BPW on MLP. Expected relative error: the residual carries ~0.5 of the
weight energy; a 1-bit Gaussian-optimal code removes ~75% of that, landing near 0.25 relative, against
the 0.15–0.20 *weighted* error the incumbent reaches by spending 8 bits on the channels the Fisher
weighting cares about most. The interesting comparison is therefore *importance-weighted*: code only the
top-x% Fisher-weighted columns at 2 bits and the rest at 1 bit.

**Runtime.** llama.cpp mainline already ships the LUT-decode kernels (`IQ1_S`, `IQ2_XXS`); the fused
path is binary-factor GEMV plus a residual IQ-dequant GEMV on the same input. This is the largest kernel
investment in this document and should wait for B3 to show that a sub-8-bit residual code wins at all
(arm b is the cheap proxy for that question). **Effort:** high. **Expected value:** the only way to
spend the second bit near the rate–distortion floor inside the factor format.

#### B6. Aspect-ratio-gated hybrid: factorize wide matrices, quantize square ones

**Mechanism.** The factor format's full-rank price is `1 + min(m,n)/max(m,n)` BPW plus scales: 1.185 for
Gemma's 6:1 MLP matrices, 1.79 for fused QKV, 1.93 for `o_proj`, and 2.0 for a square 8192² Llama
projection. Above ~1.5 BPW the near-square matrices are strictly better served by a standard 2-bit
tensor type, which reaches ~0.25 relative error at 2.06 BPW where full-rank signs reach ~0.5 at 1.9.
GGUF already mixes tensor types per tensor (the export stores Q8_0 embeddings beside NanoQuant tensors),
so the runtime cost is a planner and exporter change, not a kernel. Rule: at target BPW `t`, a unit is
factorized only if its full-rank price is below `t − margin`; otherwise it receives `IQ2_*`/`Q2_K` (or
`IQ1_*` below 2 BPW) with its bits charged to the same budget.

**Why now.** In 060, `o_proj` sits at 2.586 BPW to reach 0.06 weighted error; B2 will show what a 2-bit
tensor reaches at 2.06. For 70B, `q`/`k`/`v`/`o` are 4 of 7 units and ~18% of decoder parameters; at
2+ BPW they should not be factorized at all. **Funding:** none (bits move within the budget). **Effort:** medium
(planner menu, exporter, tests); no new kernel. **Gate:** B2 baselines exist, then a complete run versus
B1 at equal bytes. **Expected value:** likely the single largest 2-BPW gain on attention-heavy layouts,
and it composes with B3–B5 on the MLP side.

#### B7. Per-layer operating points inside one budget

**Mechanism.** Once B3/B6 give each unit a menu with exact costs, let the exact allocator choose a
different operating point per unit (binary-only 1.2, binary + patch 1.8, `IQ2` 2.06, binary + INT8
columns 2.6) under one global byte budget. R5a rejected a joint menu solved on a cheap proxy; the
difference here is that each menu entry has already passed a functional splice gate on representative
blocks, so the allocator is choosing among validated options rather than extrapolating a proxy.
**Effort:** low once B3/B6 exist. Keep the menu small; the failure mode is a large menu scored by a
proxy.

### C. Cross-cutting

#### C1. Root-cause the process-local CUDA contamination

Run the failing 056/058 boundary under `compute-sanitizer` and `CUDA_LAUNCH_BLOCKING=1` in the same
process rather than a subprocess; bisect between the ParityAdamW Kahan path, the custom kernels, and
allocator reuse after a non-finite rollback. The containment is fine at 26 blocks and 1B; at 80 blocks
and 70B the restart cost and the risk of a silent bad step both scale. This is engineering hygiene that
gates A1 at scale.

#### C2. Size the KD cache and slices for A3 before starting it

A3 needs top-64 caches of 4–16M tokens and fresh registry slices for selection and confirmation at
every horizon. Reserve them now (Doc 81) so no slice is reused across the ablation.

#### C3. Report both BPW columns for every large-model result

Doc 38's core and whole-model columns converge at 70B and diverge at 1B; FormatCapacity §6 asked for
both to be reported. Every A1 result should carry core BPW, whole-model BPW, GGUF bytes, and the
matching llama.cpp baseline bytes so the goal ("loads at ≤1 BPW") is assessed on the number a user sees.

---

## Part III — Ranked plan

| # | Item | Goal | Confidence | Cost | Prerequisite |
|---|---|---|---|---|---|
| 1 | B1 re-distill 059/060 with tail-aware + adaptive correction | 2+ BPW | high | hours | none |
| 2 | B2 `IQ1_*`/`IQ2_*`/`Q2_K` baselines on the same protocol | both | high | hours | none |
| 3 | A1 scaling curve: Gemma 3 4B/12B, Llama 3 8B at 1.0 BPW on RunPod | ≤1 BPW | high (as a measurement) | days | offloaded KD loader, profile granularity |
| 4 | B3 rate-matched second-bit menu, blocks 0/12/25 | 2+ BPW | medium | 1–2 days | B1 |
| 5 | A3 data/horizon-scaled tail-aware KD + coordinate binary QAT | both | medium-high | 1 week | C2 slices |
| 6 | B6 aspect-ratio-gated hybrid export | 2+ BPW | medium-high | 1 week | B2 |
| 7 | A2 effort allocation by KL share | ≤1 BPW at scale | high | 1 day | none |
| 8 | A6 factorization cost curves; Fisher vs `E[x²]` at 1B | ≤1 BPW at scale | high | 1 day | none |
| 9 | A4 rank annealing screen from the 060 artifacts | ≤1 BPW | medium | 1 day | none |
| 10 | B4 patch as KD capacity; B5 residual lattice code | 2+ BPW | medium / uncertain | 1 week / weeks | B3 |
| 11 | A5 nested-rank artifact | ≤1 BPW | medium | 1–2 weeks | A1 |
| 12 | C1 contamination root cause | operational | — | days | before 27B+ |

Suggested order: 1 and 2 this week (they cost nothing and reset both baselines); 3 next, on RunPod,
because every later decision about the ≤1 BPW goal depends on it; 4 and 7/8 on the workstation while 3
runs; 5 once 3 reports, in whichever regime it shows the larger gap; 6 and 10 only after 2 and 4 say the
second bit is worth restructuring.

The through-line: stop refining a 1B recipe whose task metric no longer moves, measure the format where
the goal lives, spend the second bit on a representation chosen by rate–distortion rather than by
export convenience, and let the end-to-end stage own the capacity it has already proven it can use.
