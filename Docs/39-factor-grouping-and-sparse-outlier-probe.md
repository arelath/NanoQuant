# Factor grouping and sparse-outlier probe

## Question

This probe tests two possible improvements to the pinned Gemma 3 1B recipe:

1. whether correlations beyond the existing Q/K/V fusion support a different
   factor-sharing topology; and
2. whether isolated large weights are concentrated enough to justify a sparse
   residual matrix instead of whole outlier columns.

This is analysis evidence, not an accepted compression algorithm. Any candidate
still requires a complete resident run and the retained WikiText-2 comparison.

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Target: 1.0 physical BPW per compared source-weight group
- Factorization: production ADMM, 400 outer by 5 inner iterations, cubic
  penalty schedule, `0.03` regularization, and wide-matrix transposition
- Scale fit: production two-pass alternating fit
- Rank alignment: one, to remove rank-rounding noise from the research result
- Weighted objective: the corrected 256-sample `gemma-cce-fisher-state` with
  0.6 importance shrinkage
- Checkpointing: `tools/probe_factor_grouping.py` writes each completed group
  to its JSON output and acquires the normal cross-process CUDA lease

The obsolete `gemma-full-fisher-state` was also run before its supersession was
noticed. Its output is retained as a diagnostic but is not used for the
conclusions below.

For the reciprocal attention candidate, `o_proj` is transposed so that
`v_proj` and `o_proj.T` share the 1,152-wide factor axis:

```text
current:   [Q; K; V] + O.T
candidate: [Q; K]    + [V; O.T]
```

The V and O-transpose Fisher input profiles differ. The candidate therefore
includes and charges one additional 1,152-element, 16-bit input-scale vector.
At equal total storage, the current ranks are 638 and 522; the candidate ranks
are 586 and 578.

Raw results are in `evidence/m4/factor-grouping-probe/results.json`. Corrected
CCE-weighted results and the rank-shift and adjacent-block extensions are in
`evidence/m4/factor-grouping-probe/cce-weighted-results.json`.

## Reciprocal attention result

On the raw Frobenius objective, reciprocal grouping wins 22 of 26 blocks and
reduces global normalized RMSE from 0.453614 to 0.437813, a 3.48% reduction.
That result does not survive Fisher weighting as a universal replacement.

| Corrected CCE measure | Current | Reciprocal | Change |
|---|---:|---:|---:|
| Global weighted normalized RMSE | 0.407149 | 0.408558 | +0.35% |
| Global original-space normalized RMSE | 0.487800 | 0.475090 | -2.61% |
| Blocks won | — | 18 of 26 | — |
| Mean per-block weighted change | — | — | -4.73% |
| Median per-block weighted change | — | — | -5.44% |

The apparent contradiction between 18 block wins and a 0.35% global loss is
caused by a few high-energy early-block losses. A calibration-selected oracle
that keeps the better topology independently in each block improves global
weighted normalized RMSE by 3.32%. This is an optimistic selection bound, not
held-out evidence.

The aggregate per-projection weighted result explains the trade:

| Projection | Current RMSE | Reciprocal RMSE | Change |
|---|---:|---:|---:|
| Q | 0.547589 | 0.493784 | -9.83% |
| K | 0.317883 | 0.235260 | -25.99% |
| V | 0.333129 | 0.321843 | -3.39% |
| O | 0.471157 | 0.570263 | +21.03% |

Removing V from Q/K gives Q and K substantially more coherent factors, while
sharing V with O-transpose consistently damages O. The net result depends on
the block's Fisher energy distribution.

Moving equal-cost capacity from Q/K to V/O confirms that the O loss is real:

| Block | Current | Shift 0 | Shift 32 | Shift 64 | Shift 96 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.396831 | +7.63% | +5.33% | +3.42% | +1.70% |
| 4 | 0.422786 | +12.75% | +9.93% | +7.22% | +4.84% |
| 11 | 0.340230 | -11.56% | -11.44% | -10.99% | -10.09% |
| 16 | 0.376724 | -5.82% | -6.95% | -7.88% | -8.37% |
| 22 | 0.334727 | -22.61% | -22.33% | -21.79% | -20.61% |

Percentages are weighted RMSE changes relative to the current topology. All
candidate columns remain at 0.998850 BPW. More V/O capacity nearly closes the
early-block gap and improves block 16, but slightly weakens blocks 11 and 22.
There is no single best rank split.

## Adjacent-block sharing

The earlier covariance screen found related activation spaces in adjacent
blocks, especially for gate and up projections. Direct equal-bit production
factorization does not turn that covariance into useful shared binary factors:

| Group shared across blocks | 0-1 | 10-11 | 24-25 |
|---|---:|---:|---:|
| Q/K/V | +9.99% | +5.92% | +9.71% |
| Gate | +3.21% | +2.69% | +1.45% |
| Up | +0.82% | +0.24% | +0.47% |
| Down-transpose | +2.25% | +0.12% | +2.09% |

Every entry is a regression in corrected-CCE weighted RMSE. Adjacent-block
weight sharing should therefore not be added to the recipe at this budget.
The small up/down losses may still motivate delta coding or a shared
initialization dictionary, but not direct factor tying.

## Sparse large-weight structure

The largest individual weights are concentrated enough to justify a residual
probe. Across all 26 blocks:

| Projection | Energy in top 0.1% | Density above 8x layer mean absolute weight | Energy above 8x |
|---|---:|---:|---:|
| Q | 5.01% | 0.1229% | 5.96% |
| K | 2.75% | 0.0295% | 1.25% |
| V | 1.67% | 0.0018% | 0.06% |
| O | 4.99% | 0.0731% | 4.48% |
| Gate | 2.57% | 0.0111% | 0.65% |
| Up | 1.80% | 0.0023% | 0.15% |
| Down | 2.73% | 0.0120% | 1.06% |

A flattened coordinate plus BF16 value costs approximately 35 to 39 bits for
these matrix sizes. A 0.1%-dense patch would therefore cost roughly 0.035 to
0.039 BPW before alignment and decode metadata, while covering 5% of raw Q/O
weight energy. This is promising for Q and O, much less so for V and up.

The exact coordinates do not repeat enough to support one mask shared by
neighboring blocks:

| Projection | Adjacent top-0.1% overlap enrichment over random | Mean Jaccard |
|---|---:|---:|
| Q | 3.66x | 0.1837% |
| K | 3.66x | 0.1844% |
| V | 1.22x | 0.0613% |
| O | 5.59x | 0.2811% |
| Gate | 1.30x | 0.0651% |
| Up | 1.12x | 0.0558% |
| Down | 5.95x | 0.3016% |

Some projections are enriched relative to a very small random expectation,
but even the best absolute overlap is only about 0.3%. Sparse patches should
be layer-local.

## Decision and next gate

- Do not globally replace Q/K/V fusion with reciprocal grouping.
- Add reciprocal grouping only as a research planning candidate, with
  calibration-based per-block topology and rank selection. Validate the
  selection on block outputs not used to choose it before a complete run.
- Reject direct adjacent-block factor sharing for Q/K/V, gate, up, and down at
  the tested 1.0-BPW budget.
- Continue the sparse idea by applying equal-bit column and element patches to
  the **post-factorization residual**, not the raw weight. Compare corrected
  Fisher error, block-output error, index/scale overhead, packed size, and
  kernel cost. Q and O are the first targets.
- Before changing the resident schema or algorithm version, test the remaining
  within-block attention partitions and demonstrate a held-out block-level
  gain. No production boundary changes are justified by this probe alone.
