# Review: Bounded Direct Binary-Factor Search as a Basin-Finding Strategy

## Scope

This document reviews Experiment 052
(`Docs/Experiments/052-bounded-direct-binary-factor-search.md`), its
implementation in `src/nanoquant/domain/binary_factor_search.py`, and the
question it implicitly poses: is an ADMM warm start followed by a bounded,
exact-rollback local-search ladder a good strategy for finding additional
basins of the binary factorization?

The review draws on the surrounding evidence chain: Experiments 050/051 (the
oracles), Experiment 053 (the held-out functional gate), Experiment 054 (the
functional binary learning rate), and the retained production-scale headroom
probes (`Docs/ImprovementSuggestions/ReconstructionHeadroom.md`).

## Verdict

Two separate answers:

1. **As solver diagnostics, the strategy is excellent.** Oracle calibration,
   planted zero-error controls, gauge canonicalization, and exact-objective
   rollback make 050-052 among the cleanest experiments in the repository.
   They answered the solver-optimality question definitively and cheaply.
2. **As a strategy for finding basins that matter to the compressed model, it
   is not good, and the repository's own follow-up evidence already proves
   it.** The ladder searches the wrong objective, its dramatic wins occur only
   at matrix sizes where production does not live, its bounded moves cannot
   reach production-scale barriers, and the one screening mechanism shown
   necessary for ranking a new basin correctly is structurally disabled at
   production geometry. Experiment 053 rejected every direct-search arm on
   the functional gate, and Experiment 054 then found large, functionally
   beneficial sign moves by optimizing the functional objective directly.

The pivot from 052/053 to 054 was the right call. This review consolidates
why, records the implementation-level findings, and lists what remains worth
keeping.

## What the strategy gets right

1. **Oracle calibration.** Gap-closed percentages against exhaustive or
   gauge-reduced oracles are real optimality measurements, not
   self-referential improvement claims. The planted zero-error controls
   measure the oracle's own limits (050's 10x10 procedure is explicitly shown
   not to be global).
2. **The cheap explanation was falsified first.** Exact SVID improves the
   retained 3x3 crop by 0.00010%, eliminating "five power iterations" as the
   cause before any search machinery was built.
3. **Exact-objective acceptance with rollback at every tier.** Monotone,
   debuggable, and safe; no tier can silently regress the objective.
4. **Sign-gauge canonicalization** before enumeration and before Hamming
   measurement. This is the correct way to reason about basin distance in a
   representation with row/column/component sign symmetry.
5. **Bounded work by construction.** `2^b` windows, capped pools, capped hard
   sets, and a batched ALS screen keep runtime predictable; the 32x32 scaling
   check confirms it.
6. **Ladder attribution.** The tiers isolate which neighborhood class closes
   which gap: one-bit descent closes 65.5% of the hard 3x3 gap; only a joint
   left/right window crosses the remaining coupled six-bit barrier.

## Why it is not a good basin-finding strategy

### 1. The static objective ranks basins wrong, and this is measured

Experiment 053 is decisive. All three direct-search arms improved the exact
factor objective, the fit covariance, and the *held-out* covariance — and all
three regressed full-splice KL (+0.19% to +0.72%, every paired 95% interval
entirely above zero). This is not overfitting; it is a mismatch between the
separable projection-local objective and the projection's role inside the
nonlinear block. Meanwhile Experiment 054 moved 54,749 signs (0.83% of the
block-0 gate factor) under the functional objective, improved held-out block
loss by 34.5% on the probe, and improved teacher KL by 10% with an interval
wholly below zero — while the *weight-space* matrix error got worse.

The basins worth finding are defined by the functional objective, and near
the ADMM solution the two objectives are anti-correlated at the margin. A
stronger static-objective basin finder is therefore a machine for locating
functionally harmful candidates faster.

### 2. Static headroom vanishes exactly where production lives

The best observed squared-error reductions on natural (non-planted) targets,
across the whole 050-052 chain plus the retained production probes:

| Geometry | Free sign bits | Best natural-target reduction |
|---|---:|---:|
| 3x3 full rank | 10 | 24-96% |
| 4x4 full rank | 21 | 82.6% |
| 10x10 rank 2 | 19 | 8.2% (selected counterexample) |
| 10x10 full rank | 190 | ≤ 3.1% |
| 32x32 rank 3 | ~124 | 0.0% |
| 1024x1152 rank ~542 (production) | ~10^6 | ~0.3% (ALS + one-bit + STE + multistart + basin hopping) |

This monotone collapse is the expected concentration behavior of large dense
binary quadratic landscapes: as dimension grows, local-optimum energies
concentrate in a narrow band, so "another basin" exists but is worth almost
nothing. Tiny matrices are precisely the regime where basins differ
dramatically — and precisely the regime that does not represent production.
The strategy's most dramatic evidence (3x3, 4x4) was gathered where it
generalizes least, and Experiment 050's own interpretation already concedes
this. Production-scale basin hopping returning to the incumbent
(ReconstructionHeadroom) says the same thing from the other direction.

### 3. Bounded windows cannot reach production-scale barriers

Both natural wins required near-total coverage of the free sign space:

- 3x3: the winning joint window enumerated **all 10** free bits;
- 10x10 rank 2: the basin was 11 canonical bits away and required 19-of-19-bit
  enumeration (or six 16-bit windows covering ~75% of the space) *plus* an
  8-pass ALS screen. The default heuristic 12-bit window closed **0.00%**.

Production stacked QKV (1536x1152, rank 638) has ~1.7 million free sign bits.
A 20-bit window covers ~10^-5 of the coordinates, and Experiment 052's own
conclusion 5 states that window *selection* is binding. Margin/residual
heuristics that failed to nominate the correct 11 coupled bits out of 19 will
not nominate them out of 1.7 million. The 32x32 result (nothing found on
natural targets) is the trend's next data point, not an anomaly.

### 4. The screen that finds new basins is disabled at production size

`binary_factor_search.py:717` gates the batched post/pre/mid ALS screen on

```python
use_middle_scale_screen = target.numel() * patterns.shape[0] <= 100_000_000
```

At production geometry (≥1.2M target entries) any window beyond ~6 bits falls
back to the fixed-scale affected-row/column screen. But Experiment 052's
10x10 counterexample explicitly measured that "fixed-scale and
middle-scale-only screening both miss it" — a short batched ALS screen was
*necessary* to rank the new basin correctly. As implemented, the
production-scale configuration of the joint search is structurally the
configuration that was demonstrated not to find new basins. Any deployment of
joint windows on real groups inherits this contradiction unless the screen is
made scalable (see recommendations).

### 5. The ladder is exploitation, not exploration

Every accepted move must beat the incumbent on the exact objective
(`_accept_vectors`, outer-pass rollback). The only barrier-crossing devices
are enumerated windows (≤ 20 bits) and one-shot candidates (continuous
least-squares signs, component replacement). The exploration mechanisms from
`BinomialFactorizationOptimizations-PossibleAdditions.md` — elite sets, path
relinking, fusion between basins, tabu with uphill acceptance — were either
not implemented or implemented in a commit-only-improving-prefix form
(variable-depth chains). A method that only ever accepts improvements can
only find basins whose separating barrier fits inside one enumerated window.
For finding *additional* basins, this is the wrong structural shape; it is a
polisher with a small enumerative escape hatch.

### 6. Reporting conflates scale fitting with sign search

Minor but worth fixing if the tooling is reused:

- `BinaryFactorSearchResult.before_error` is measured before the initial
  `fit_scales`, so headline before/after reductions include plain scale ALS.
  Experiment 053 observed a control where scale fitting alone produced a
  36.80% reduction with zero accepted sign moves. The result should also
  carry the post-initial-fit error so sign-search gains are separable.
- Oracle comparisons were not scale-depth matched: the 3x3 candidate "beat"
  the retained oracle (100.04%) only via deeper continuous fitting, and
  Experiment 051 showed scale multistart depth matters even after complete
  sign coverage. Future ladders should time-match or depth-match the
  continuous budget across arms.
- The hard targets were found by scanning for failures. There is no
  prevalence estimate: no measurement of what fraction of natural
  production-geometry groups have a crossable barrier worth anything.

## What to do differently

Ranked by expected value:

1. **Accept the pivot that Experiments 053/054 already made.** The functional
   block tuner with an unlocked binary learning rate is the basin-finding
   mechanism. It routinely crosses barriers of tens of thousands of coupled
   sign bits — no bounded enumerative neighborhood competes with that — and it
   selects basins under the objective that gates promotion.
2. **Point 052's instrumentation at 054's output.** Canonicalize the tuned
   factors against their ADMM initialization and measure: canonical Hamming
   distance, the margin distribution of flipped bits under the static
   objective, and whether flips cluster in components. This answers, cheaply
   and definitively, whether any bounded static-objective neighborhood could
   ever have proposed the functionally chosen basins, and characterizes what
   the functional objective actually pays for. It is the best remaining use
   of the 052 machinery.
3. **If direct search is ever revisited, change the signal and placement
   before the strength** (this is Experiment 053's decision 4; it deserves
   restating because the search code makes strengthening tempting). The
   gram/cross construction accepts a dense covariance (`Q = X C X^T`) with a
   two-line change; a block-output-aware proxy would be better still. But the
   covariance-refiner precedent (local objective improved in 104/104 groups,
   WikiText perplexity +19.26%) means functional acceptance gating is
   mandatory no matter how the proposal objective improves.
4. **Reposition any retained search as a proposal generator inside the
   functional loop**, not a standalone post-processor: propose
   window/codebook/component candidates, accept on a held-out functional
   mini-batch. This attacks the measured objective-placement mismatch instead
   of the search strength that was never the binding problem.
5. **Measure prevalence before building more machinery.** Sample ~100
   production-geometry groups, run the cheap tiers plus one joint-window
   configuration, and report the distribution of static gains with
   spot-checked functional deltas. If the distribution centers at ≤0.3%
   static and ≤0 functional — which every existing data point predicts —
   close the static basin-search workstream permanently and record it.
6. **Fix the joint-window screen if it ever runs at scale.** Subsample rows
   and columns for the ALS screen, or maintain rank-w incremental prediction
   updates so screen cost scales with the window rather than the full matrix.
   Without this, production joint windows run in exactly the mode shown to
   discard correct signs.
7. **Mind the opportunity cost.** The measured production levers are stacked
   QKV factorization (4-24%), cross-layer rank allocation (8.2%), and
   input-importance weighting (20-30% functional). Static basin polish is
   bounded by ~0.3% on a format whose gap to the 1-bpw rate-distortion floor
   is itself the dominant loss. Solver-side search effort competes directly
   with levers one to two orders of magnitude larger.

## What to keep

- **The oracles and planted controls** (Experiments 050/051, the low-rank
  exhaustive oracle in 052) as permanent regression tests for any future
  solver — Experiment 053's decision 5, endorsed here. They are the only way
  to know whether a future method can cross a known combinatorial barrier.
- **The scale-eliminated update algebra** (profiled alpha/beta scoring with
  O(1) flip evaluation). It is objective-agnostic and transfers unchanged to
  covariance-weighted or proxy-quadratic objectives.
- **Gauge canonicalization** as the standard basis for basin-distance
  measurement, immediately reusable for recommendation 2.
- **The exact-rollback discipline and the evidence protocol.** Whatever the
  strategy's fate, the experimental hygiene of 050-053 is the template the
  next workstream should copy.

## References

- `Docs/Experiments/050-tiny-factorization-optimality.md`
- `Docs/Experiments/051-four-by-four-exhaustive-sign-coverage.md`
- `Docs/Experiments/052-bounded-direct-binary-factor-search.md`
- `Docs/Experiments/053-direct-binary-search-held-out-gate.md`
- `Docs/Experiments/054-functional-binary-learning-rate.md`
- `Docs/ImprovementSuggestions/BinomialFactorizationOptimizations.md`
- `Docs/ImprovementSuggestions/BinomialFactorizationOptimizations-PossibleAdditions.md`
- `Docs/ImprovementSuggestions/ReconstructionHeadroom.md`
- `src/nanoquant/domain/binary_factor_search.py`
