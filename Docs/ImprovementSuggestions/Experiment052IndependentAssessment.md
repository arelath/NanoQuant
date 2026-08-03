# Independent Assessment of Experiment 052's Basin-Search Strategy

**Date:** 2026-08-02
**Scope:** `Docs/Experiments/052-bounded-direct-binary-factor-search.md`, its implementation and tests, and the later repository evidence that bears on promotion.

## Bottom line

Experiment 052 is a good **solver diagnostic and incumbent-polishing method**. It is not, in its current form, a strong general strategy for discovering additional production-relevant basins.

The implementation has several valuable properties: it preserves the packed representation and BPW, profiles out row or column scales for cheap moves, canonicalizes exact sign gauges, bounds exponential work, refits scales, and rolls back against the exact separable weighted objective. The 3x3 and 10x10/rank-2 results convincingly demonstrate that coupled left/right sign barriers exist and that the implementation can cross them.

However, most of the search still enumerates a small coordinate subcube chosen from one incumbent by one-bit margins or residual mass. That is large-neighborhood local search, not broad basin discovery. More importantly, Experiment 053 shows that optimizing this separable projection objective can move in the wrong direction for block output and language KL, while Experiment 054 shows that functional factorized tuning can improve held-out block loss and KL while making weighted matrix error worse. The principal production question is therefore no longer “how can the diagonal objective be optimized more thoroughly?” It is “how can discrete candidate diversity be generated and selected using the functional objective that matters?”

I would retain the Experiment 052 machinery as an oracle-facing regression tool and as one proposal source, but I would not spend the next experiment on larger windows, more passes, or wider exact-SVD use under the same objective.

## What Experiment 052 gets right

### It isolates a real solver gap

Experiments 050–052 correctly separate representational capacity from optimizer quality. Complete gauge-reduced enumeration on 3x3 and near-complete coverage on 10x10/rank-2 show that substantially better signs can exist at the same rank, scale layout, and physical bit cost. The difficult 3x3 case is especially useful because the joint window reaches the oracle basin only when signs from both factors are changed together.

That is meaningful evidence against treating ADMM as globally or consistently near-globally optimal.

### Its safety contract is appropriate

`src/nanoquant/domain/binary_factor_search.py` has the right safeguards for an experimental local solver:

- `_accept_vectors` profiles the affected boundary scale and accepts only an objective improvement.
- `_canonicalize_sign_gauges` removes row, column, and component sign gauges before joint enumeration without changing the represented matrix.
- `_joint_bit_window` caps enumeration at `2^joint_bits`, screens candidates in batches, and fully refits only a bounded top set.
- `refine_binary_factors_separable` performs a final complete scale fit and rolls the whole outer pass back if the exact weighted objective does not improve.
- The defaults keep joint search disabled, and the resident compression path does not call this solver.

Those choices make the code useful for diagnostics even when a proposed tier is ineffective.

### The ladder is scientifically useful

The continuous, one-bit, pair, block, component, and joint tiers identify which neighborhood crosses a known barrier. The result that a weak scale screen can visit the right signs and still discard them is particularly important. It prevents a false conclusion that sign coverage alone measures search quality.

## Why this is not yet a general basin finder

### 1. The production objective is already falsified as a promotion criterion

Experiment 053 is the decisive result. On a real block-12 QKV group, direct search improved:

- the factor objective;
- fit covariance;
- held-out covariance.

Nevertheless, all three tested arms worsened block-output NRMSE and full-splice KL, with paired KL intervals entirely on the harmful side. This is not ordinary fit-split overfitting. It is an objective-placement mismatch caused by the projection's nonlinear role in the block and model.

Experiment 054 supplies the converse evidence: increasing the binary learning rate during functional factorized tuning produced many useful sign changes, improved held-out block loss by 34.50%, and improved splice KL by 10.05%, while weighted matrix error became worse. A static-error improvement requirement would reject precisely this useful direction.

Consequently, exact rollback on the separable objective is a sound solver-debugging contract but a harmful hard constraint for production basin discovery. A production search must be allowed to accept candidates that worsen static matrix error when they improve a disjoint functional gate.

### 2. The strongest results cover almost all signs only because the problems are tiny or very low-rank

After gauge removal, the implementation has `(r - 1)(m + n - 1)` free sign variables.

- 3x3/rank-3 has 10 free signs. The 1,024-pattern joint search is complete.
- 10x10/rank-2 has 19 free signs. A 16-bit window covers most variables, and the 19-bit diagnostic is complete.
- 32x32/rank-3 has 126 free signs. Six 10-bit windows evaluate only 6,144 points in a few selected subcubes.
- The real Experiment 053 QKV group is 1536x1152/rank-638, or 1,711,619 gauge-free sign variables. A 10- or 12-bit window is an infinitesimal neighborhood of one incumbent.

The small cases prove that the move implementation works. They do not establish that the window selector can locate useful subcubes at production rank. The 32x32/rank-3 check is still much closer to the low-rank oracle regime than to the actual rank-638 QKV geometry.

### 3. Window selection is mostly interaction-blind

`_joint_bit_window` selects variables using independently computed one-bit margins for its first three trials and row/column residual mass thereafter. Its left-heavy, balanced, and right-heavy modes change quotas, but they do not estimate which left and right bits interact constructively.

This is exactly the failure mode exposed by the oracle comparison: the destination basin can be several individually unattractive changes away. Low one-bit margin is a reasonable ambiguity heuristic, but it does not identify the cross-factor combination responsible for the gain. Residual mass identifies where error lives, not which rank components and opposite-factor signs can correct it.

After six trials, the selection modes repeat unless earlier accepted moves alter the state. There is no candidate archive, coverage objective, randomized-but-reproducible diversification, interaction graph, or novelty criterion. The method can repeatedly explore nearby variants of the same basin while leaving most of the sign space structurally untouched.

### 4. The screening rule becomes weakest at the scale where it matters most

At `src/nanoquant/domain/binary_factor_search.py:717`, the joint search uses batched pre/mid/post ALS screening only when:

```text
target.numel() * pattern_count <= 100,000,000
```

Above that cliff it ranks candidates with current scales using only affected rows and columns. Experiment 052 already establishes that fixed-scale and middle-scale-only screens can discard the right basin. Production matrices necessarily take the weaker branch for even modest window sizes. Thus the current scaling policy systematically removes the screening behavior that made the low-rank counterexample succeed.

This should be treated as a candidate-recall problem, not just a speed tradeoff. There is currently no reported measurement of whether the screened top `K` contains the candidates that would rank best after a full scale or functional refit.

### 5. Continuous-scale basin dependence remains

`fit_scales` is monotone alternating least squares initialized from the incumbent's pre/mid/post scales. The full refit of a new sign pattern therefore starts in the old continuous basin. The exhaustive experiments needed multiple scale starts and much deeper fitting to make their oracle credible, but joint-search candidates receive a single inherited initialization.

This favors nearby sign patterns whose good scales resemble the incumbent and can hide a genuinely different sign basin whose useful scale configuration requires a different initialization or normalization. More scale passes do not solve that initialization bias.

### 6. Two selection heuristics do not target the opportunities their names imply

There are two concrete implementation choices I would change even for continued static-objective research:

- `_block_pass` chooses “hard” vectors using `vector_weight * (max(current_score) - current_score)`. `current_score` is explained energy after profiling the row or column scale. True residual is `vector_weight * (target_energy - current_score)`. Because target energy varies by vector, the current proxy can prioritize the wrong rows or columns.
- `_component_replacement_sweep` considers components in descending `abs(scale_mid)`. Large middle scale does not imply high replacement headroom. Weak, redundant, or poorly aligned components can be better slots to repurpose. Candidate order should use leave-one-component residual gain, redundancy, and residual alignment, and should include both weak and strong components.

Neither issue invalidates the reported small experiments, but both limit the search's ability to spend a tight production budget intelligently.

### 7. The tests establish safety, not basin-search efficacy

`tests/unit/test_binary_factor_search.py` covers monotonicity, pattern bounds, a represented target, codebook transfer, a constructed two-bit chain, and complete 3x3 pattern count. Those are good unit contracts. Missing search-quality contracts include:

- gauge-equivalent inputs producing equivalent canonical outcomes;
- screen top-`K` recall against fully refitted rankings on small exhaustive cases;
- correct hard-vector ranking from actual weighted residual;
- interaction-aware window selection on a constructed coupled-factor barrier;
- distinct-window coverage and deterministic diversity;
- production-shape bounded-memory behavior for the same screening algorithm used in small tests;
- candidate selection under block-output loss rather than only separable weight error.

## What I would do differently

### 1. Make functional tuning the main basin explorer

The strongest current evidence favors the existing STE-based factorized tuner with separate binary learning rates. It changes signs under the actual block objective, and Experiment 054 has already shown a useful regime that the static matrix objective would reject.

I would use Experiment 054's validated binary-rate arm as the principal baseline, then search around the **functionally tuned** state rather than around raw ADMM. Direct discrete search should be placed after, or alternated with, short functional tuning segments. Scales, patches, outliers, and biases should receive a short functional refit after any material sign perturbation.

### 2. Maintain a small population, not one monotone incumbent

Use a bounded archive of gauge-canonical candidates, for example 8–32 per selected owner. Seed it from:

- the ordinary ADMM result;
- two or more genuinely diverse ADMM or SVID initializations when affordable;
- selected functional-tuning checkpoints;
- confidence-guided perturbations of low-margin latents;
- component retirement/replacement proposals;
- a small deterministic set of structured random perturbations.

Deduplicate candidates by canonical sign hash and track both objective values and Hamming novelty. Run cheap local improvement from every surviving candidate. This tests multiple basins instead of repeatedly improving only the best static incumbent.

Temporary uphill moves should be permitted inside a proposal. Acceptance into the archive happens after local reoptimization and functional evaluation, so safety does not require every intermediate bit flip to improve the diagonal objective.

### 3. Build coupled, component-centered neighborhoods

For the product `L diag(mid) R`, useful cross-factor interactions are organized by rank component. A better window generator would form small coupled rectangles such as:

- several `L[i, k]` signs for high-residual output rows;
- several `R[k, j]` signs for high-residual input columns;
- one or a few shared components `k` with high residual interaction or redundancy.

Rank windows by a block-loss gradient/Jacobian surrogate or at least by the predicted second-order cross effect, not by two independent one-bit-margin lists. Use a portfolio of window sources: interaction clusters, latent uncertainty, residual alignment, weak/redundant components, and deterministic random coverage. Record selected variables and reject duplicate subcubes.

For larger windows, use beam search, tabu/iterated local search, or perturb-and-reopt rather than exhaustive enumeration. The important bounded quantity should be functional evaluations and retained candidates, not only `2^bits`.

### 4. Use a multi-stage scorer, but do not make static error a hard gate

A practical hierarchy is:

1. **Proposal score:** cheap gradient/Jacobian or diagonal reconstruction estimate.
2. **Fit functional score:** exact block-output loss on cached calibration activations, followed by a short functional scale/parameter refit.
3. **Held-out block score:** disjoint activations, used for candidate selection and stopping.
4. **Language gate:** paired splice KL and NLL for the few survivors.

Static factor and covariance errors remain useful diagnostics and compute priors. They should not veto a functionally better candidate. Experiment 054 proves why.

For any approximate screen, measure recall directly: on tractable cases, fully refit all patterns and report the rank of the eventual winner under the cheap screen, plus top-1/top-4/top-16 recall. Replace the current fixed 100-million-element algorithm switch with a chunked, memory-budgeted scorer so small and large cases use the same semantics.

### 5. Improve continuous refitting and component allocation

For the small number of retained candidates, use several deterministic scale initializations rather than one inherited state. Suitable starts include incumbent scales, normalized unit boundary scales with a solved middle scale, and a component-balanced normalization. Canonicalize scale gauges before comparison so scale magnitude drift does not masquerade as diversity.

For component replacement, score at least three pools:

- weakest or most redundant components;
- strongest components;
- components with the largest residual bilinear response after removal.

The best replacement slot is an empirical question; `abs(mid)` alone should not decide it.

## Recommended next experiment

I would run a bounded four-arm study on several real owners from different blocks and projection families:

1. current ADMM plus normal functional tuning;
2. Experiment 052 search before functional tuning;
3. normal functional tuning plus population-based coupled proposals and short functional refit;
4. functional tuning alone with an equalized compute budget.

Use actual production ranks, not only low-rank crops. Predeclare separate fit and held-out activation slices. Report:

- unique canonical basins visited;
- sign distance and reconstruction distance between survivors;
- candidate-screen recall where exhaustive checking is possible;
- fit and held-out block loss;
- paired splice KL and NLL;
- static diagonal and covariance error as explanatory metrics;
- wall time, peak GPU memory, and evaluations per accepted candidate;
- exact BPW and unchanged representation fields.

Promotion should require a repeatable held-out functional gain across more than one owner and block family, followed by the normal complete-model gate. If the population arm does not beat equal-compute functional tuning, the correct conclusion is that the direct combinatorial machinery should remain an oracle and diagnostic rather than a production stage.

## Priority order

1. Do not expand the current separable-objective joint search into the resident recipe.
2. Treat Experiment 054 functional tuning as the basin-search baseline.
3. Fix hard-vector scoring, component candidate ordering, and screen-recall instrumentation.
4. Add a small gauge-canonical candidate archive and interaction-aware, component-centered proposal generator.
5. Rerank and refit candidates with disjoint block-output objectives.
6. Only after a multi-owner functional win, run a complete compressed-model comparison.

In short: Experiment 052 found a real combinatorial phenomenon and implemented a useful bounded laboratory for it. The next advance should come from changing the **objective, candidate diversity, and placement**, not from searching the same local static objective more exhaustively.
