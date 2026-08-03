# Experiment 052: Bounded Direct Binary-Factor Search

## Question

Can NanoQuant's ADMM output be improved by optimizing the final binary product
directly, while keeping combinatorial work bounded and preserving the existing
packed representation?

This experiment follows
`Docs/ImprovementSuggestions/BinomialFactorizationOptimizations.md`. It is a
solver-objective experiment. It does not establish compressed-model language
quality and does not enable the new search in the resident recipe.

## Implemented search ladder

The implementation in `src/nanoquant/domain/binary_factor_search.py` retains

```text
diag(scale_post) L diag(scale_mid) R diag(scale_pre)
```

and adds exact-objective rollback around these moves:

1. a continuous least-squares sign candidate;
2. scale-eliminated one-bit updates;
3. pair updates from low-margin bit pools;
4. exact row/column component windows;
5. complete rank-component replacement;
6. a joint window containing signs from both factors.

The joint search canonicalizes row, column, and component sign gauges before
enumeration. Its exponential work is capped at `2^joint_bits`. It selects
complementary balanced, left-heavy, and right-heavy windows using one-bit
margins and residual energy. Every workload now receives the same post/pre/mid
ALS screening semantics. A memory-derived dynamic batch size replaces the old
100-million-element algorithm cliff, and bounded `bmm` operations avoid the
multi-gigabyte einsum intermediates found by the first production-shape probe.
Only the best screened candidates receive the full scale refit.

The screen uses at least as many ALS passes as the retained full refit. On the
retained 5x5/rank-3 recall corpus, an 8-pass screen recovered the fully refitted
winner at rank 1 in 28/32 cases and within the top four in 30/32. Matching the
16-pass refit depth recovered the winner at rank 1 in all 32 cases. The old
fixed-scale screen recovered none at rank 1 and only one within the top 16.

Two other selection defects were corrected during the review:

- hard rows and columns are ranked by actual weighted residual energy,
  `weight * (target_energy - explained_energy)`, rather than distance from the
  maximum explained energy;
- component replacement interleaves weak/removal-cost, strong-middle-scale,
  and residual-aligned component pools instead of considering only descending
  `abs(scale_mid)`.

The ADMM implementation also has an opt-in `exact_svd` SVID projection method.
The existing `power` method remains the default, preserving existing behavior.

## Low-rank exhaustive oracle

The exhaustive oracle now supports `m x n` targets at arbitrary rank `r`. After
fixing exact sign gauges, it visits

```text
2^((r - 1)(m + n - 1))
```

sign pairs. This makes larger matrices enumerable when rank is small:

| Geometry | Gauge-distinct sign pairs | Measured oracle time per target |
|---|---:|---:|
| 5x5, rank 3 | 262,144 | 4.7-5.3 s |
| 6x6, rank 3 | 4,194,304 | 16.7-17.1 s |
| 10x10, rank 2 | 524,288 | 6.3-6.7 s |

The times use different scale-search depths and therefore compare discrete
coverage cost, not identical continuous-oracle strength.

## Exact-SVID diagnosis

Exact SVID does not explain the large full-rank real-crop gap. On the retained
3x3 Gemma/Fisher crop it improves weighted squared error by only 0.00010%.
At 10x10 full rank it is target-dependent: it improves the sampled Gaussian by
5.98%, the planted target by 8.46%, and regresses the sampled real crop before
the best warm start is selected. A full SVD at every ADMM projection is also not
a reasonable production default.

Conclusion: exact SVID is useful as a diagnostic and possible small-matrix warm
start, but direct final-product optimization is the important mechanism.

## Full-rank 3x3 ladder

The pinned block-12 `q_proj` crop at rows 142-144 and columns 393-395 uses the
same Fisher-weighted objective as Experiments 050 and 051.

| Stage | NRMSE | Oracle gap closed | Stage time |
|---|---:|---:|---:|
| Best scaled ADMM warm start | 0.156834 | 0.00% | - |
| Continuous candidate | 0.156827 | 0.02% | 0.74 s |
| One-bit search | 0.124391 | 65.51% | 1.16 s |
| Pair search | 0.124378 | 65.54% | 0.77 s |
| Full row/column block search | 0.124377 | 65.54% | 0.91 s |
| Component replacement | 0.124377 | 65.54% | 0.85 s |
| Joint factor window, all 1,024 signs | 0.103269 | 100.04% | 13.26 s |
| Retained exhaustive oracle | 0.103297 | 100.00% | 0.21 s |

The candidate slightly beats the retained oracle because it receives deeper
continuous scale fitting. Canonical comparison shows the oracle basin is six
coupled sign changes away across both factors. This explains why independently
optimal row/column moves and globally optimized single-component replacement
both stall.

## Larger low-rank results

The sampled natural Gaussian and real targets at 5x5/rank-3 and 6x6/rank-3 are
already at their exhaustive-sign floors. Low rank is therefore often easier for
ADMM even as matrix dimensions grow.

The broader 10x10/rank-2 sweep found a counterexample. On the real Fisher crop
at rows 14-23 and columns 799-808:

| Search | NRMSE | Squared-error reduction vs ADMM | Joint patterns | Joint-stage time |
|---|---:|---:|---:|---:|
| Scaled ADMM | 0.585725 | - | - | - |
| Initial 12-bit joint window | 0.585725 | 0.00% | bounded | < 1 s |
| Complete sign oracle | 0.561104 | 8.23% | 524,288 | 6.25 s |
| Full 19-bit joint search with 8-pass ALS screen | 0.561113 | 8.23% | 524,288 | 22.77 s |
| Six complementary 16-bit windows with ALS screen | 0.560825 | 8.32% | 393,216 | 16.67 s |

The oracle-equivalent solution differs from the ADMM basin in 11 canonical
bits: four in `L` and seven in `R`. Fixed-scale and middle-scale-only screening
both miss it. A short batched post/pre/mid ALS screen is necessary to rank the
new basin correctly. Six complementary 16-bit windows slightly outperform the
retained oracle because their final candidates receive deeper scale refits.

## Scaling check

On 32x32/rank-3, six 10-bit windows evaluate 6,144 joint patterns in about
4.0-4.5 seconds per target. They do not improve the sampled natural Gaussian or
real crop, while the complete bounded ladder reduces the planted target's
squared error by 97.99%. This confirms bounded runtime and numerical rollback,
but it does not justify running joint windows on every production group.

## Representative real-layer crop study

The follow-up study moved beyond 10x10 while retaining tractable search. The
ladder probe now accepts rectangular crops, so attention used 256x256/rank-128
and MLP projections used 512x128 or 128x512/rank-128 crops. Ten deterministic
Fisher-weighted crops covered Q, gate, and down projections in blocks 0, 12,
and 24.

Starting from 200-iteration ADMM/SVID candidates, direct polishing improved
weighted squared error on all ten crops by 0.34%-1.69%, averaging 0.85%. The
stage-average cumulative gains were:

| Last included tier | Mean gain vs power ADMM |
|---|---:|
| Continuous candidate | 0.11% |
| One-bit | 0.44% |
| Codebook | 0.44% |
| Variable-depth | 0.75% |
| Pair | 0.78% |
| Block | 0.84% |
| Component | 0.85% |
| Joint | 0.85% |

Thus codebook transfer, full-component replacement, and joint enumeration add
almost nothing at this crop scale. Cheap sign descent, short variable-depth
chains, and small block moves account for nearly all measured benefit.

### ADMM compute control

The initial crop gain was mostly an under-convergence artifact. On the same
block-12 crops, four-times-longer ADMM beat the complete 200-iteration direct
ladder at comparable wall time:

| Crop | 200-iteration direct NRMSE | 800-iteration ADMM NRMSE | Squared-error gain of longer ADMM vs direct |
|---|---:|---:|---:|
| Q, 256x256/rank-128 | 0.411828 | 0.407202 | 2.23% |
| Gate, 512x128/rank-128 | 0.338912 | 0.329457 | 5.50% |
| Down, 128x512/rank-128 | 0.480876 | 0.468933 | 4.91% |

Direct search after mature 800-iteration ADMM remained complementary, but the
residual gains were only 0.29% for Q, 0.53% for gate, and 0.25% for down. Four
gauge-distinct mature gate starts, each receiving the full combined polish,
did not beat the polished best incumbent.

## Deep 32x32 and 64x64 basin search

The intensive oracle was extended with:

- chunked population scale fitting with bounded batched systems and `bmm`
  contractions;
- arbitrary heuristic rank;
- geometric mutation radii rather than nearly uniform destructive radii;
- random and residual-gradient component-centered coupled mutations;
- local reoptimization before generation selection;
- a gauge-canonical archive split between quality and Hamming novelty;
- canonical deduplication before archive selection.

Synthetic 32x32/rank-16 controls showed 47%-56% improvements over deliberately
short ADMM, proving that the machinery can escape to a materially better basin
when one is accessible. Real Fisher-weighted crops were much closer to their
measured floors:

| Real crop | Mature production NRMSE | Deep-search NRMSE | Squared-error gain | Canonical sign distance |
|---|---:|---:|---:|---:|
| Block-12 Q, 32x32/rank-32 | 0.237324 | 0.236913 | 0.35% | not retained by the early schema |
| Block-12 Q, 64x64/rank-64 | 0.273126 | 0.272757 | 0.27% | not retained by the early schema |
| Block-12 Q, 32x32/rank-16 | 0.416987 | 0.416898 | 0.043% | 2 |
| Block-12 gate, 32x32 crop A | 0.279255 | 0.278882 | 0.27% | 8 |
| Block-12 gate, 32x32 crop B | 0.278871 | 0.276754 | 1.51% | 20 |
| Block-12 gate, 32x32 crop C | 0.291879 | 0.290959 | 0.63% | 1,028 |

Crop C proves that a genuinely distant real basin can survive the novelty
archive and improve the objective. Its advantage is nevertheless modest. For
the best crop B, a maximum-depth confirmation doubled the population to 8,192,
used 64 elites, 32 generations, 32 mature ADMM starts, and 12 final local
sweeps. It saturated at exactly the same 1.51% gain.

## Conclusions and deployment policy

1. Five power iterations are not the principal cause of the known real-crop
   failures.
2. Scale-eliminated one-bit descent is the best cheap first tier. It closes
   65.5% of the difficult 3x3 gap in two accepted bit moves.
3. Pair, row/column block, and single-component moves do not cross the measured
   coupled-factor barriers.
4. Joint factor windows with continuous-scale screening can close those
   barriers and preserve the current format exactly.
5. Window selection and scale screening are both binding. Enumeration with a
   weak scale screen can visit the optimal signs and still discard them.
6. Natural low-rank targets are frequently already solved by ADMM. Joint search
   must therefore be selective rather than universal.
7. On representative real crops, spending comparable compute on ADMM
   convergence is substantially better than polishing an under-converged
   incumbent.
8. Deep novelty-preserving population search can find distant real basins, but
   the best measured static-objective gain is 1.51% and saturates under a much
   larger confirmation.

The production recommendation is therefore narrower than the original one:

- increase or convergence-check ADMM before invoking any combinatorial tier;
- retain one-bit, variable-depth, pair, and small block moves as diagnostic or
  capped incumbent-polishing tools;
- do not spend production compute on codebook, full-component, joint, or broad
  static-objective population search based on the current evidence;
- do not promote any static improvement without disjoint block-output and
  language gates.

Exact SVID, 16-20-bit exhaustive diagnostics, and complete low-rank oracles
should remain diagnostic tools. The prior covariance experiment showed that a
better static factor objective can still worsen language quality, so this
solver improvement is not a substitute for a compressed-model benchmark.

## Evidence

- `evidence/052/ladder3-real-v4.json`
- `evidence/052/ladder5-rank3-v1.json`
- `evidence/052/ladder6-rank3-v1.json`
- `evidence/052/ladder10-rank2-v1.json`
- `evidence/052/ladder10-rank2-joint19-als8.json`
- `evidence/052/ladder10-rank2-joint16-als8-sixwindows.json`
- `evidence/052/ladder10-v1.json`
- `evidence/052/ladder32-rank3-v1.json`
- `evidence/052/crop-study-b0-{q,gate,down}.json`
- `evidence/052/crop-study-b12-{q,gate,down}-*.json`
- `evidence/052/crop-study-b24-{q,gate,down}.json`
- `evidence/052/crop-control-b12-*-admm800*.json`
- `evidence/052/deep-basin-b12-q-{32x32,64x64}*.json`
- `evidence/052/deep-basin-b12-gate-32x32-*.json`

The evidence directory is intentionally ignored; the reproducible tools,
tests, and this report are the durable record.
