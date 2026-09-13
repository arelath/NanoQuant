# Additive row and column offsets on the retained best fit

Date: 2026-09-13. Supplementary CPU diagnostic for Experiment 068.

Optimal additive row-plus-column weight offsets reduce the current winner's
Fisher-weighted error by **0.099263%**, holding all binary signs and scales fixed.
The reduction is small. Earlier heatmap colors represented squared error above
or below the panel average, not positive or negative reconstruction residuals.

| Correction | FP64 weighted error | Error reduction | Weighted NRMSE |
| --- | ---: | ---: | ---: |
| None | 4.8500448898 | 0% | 0.4769712602 |
| Row offsets | 4.8493510176 | 0.014307% | 0.4769371400 |
| Column offsets | 4.8459244338 | 0.084957% | 0.4767686067 |
| Both | 4.8452305706 | 0.099263% | 0.4767344725 |
| Both, rounded to BF16 | 4.8452305841 | 0.099263% | 0.4767344731 |

Both offsets reduce weighted NRMSE by 0.049644% and raw error energy by
0.094172%. Fitting both offsets directly to the unweighted objective instead
reduces raw error by 0.099551%, and weighted error by 0.093516%.
The original baseline and best 10%-perturbed ADMM candidate behave similarly:
their weighted-error reductions from both offsets are 0.102834% and 0.098152%.

## Exact conditional fit

Let `R = target - prediction`, with positive separable Fisher weights `u[i] v[j]`.
The weighted row means, column means, and grand mean are:

```
r[i] = sum_j v[j] R[i,j] / sum_j v[j]
c[j] = sum_i u[i] R[i,j] / sum_i u[i]
g = sum_i u[i] r[i] / sum_i u[i]
```

The optimal correction is `r[i] + c[j] - g`. A gauge choice stores row bias
`r-g` and column bias `c`. This is the exact global optimum over these additive
offsets for the fixed factors. It does not bound the gains from jointly refitting
the binary factors. The seven protected outlier columns retain zero correction;
the row offset is gated off there and their column offset is zero.

The corrected residual's maximum absolute weighted row/column mean is below
4.3e-19. The measured reduction also equals the correction's weighted squared
norm. Independent tiny-fixture tests compare the formula with a full weighted
least-squares design matrix, verify removal of a pure row-plus-column signal,
and verify the projection-energy identity.

## Representation and scope

Two full BF16 vectors use 16,128 payload bytes and add 0.016204 BPW on this
1152-by-6912 matrix; FP32 doubles that cost. BF16 rounding loses approximately
1.35e-8 weighted energy. Existing sidecar metadata already identifies the protected
columns; additional encoding metadata is not included in this payload estimate.
This is a representation change and therefore not an improvement at exactly the
original storage budget.

These are offsets to weights, not a normal constant output bias. With active-column
mask `m`, their inference contribution would be `row_bias * dot(m, x)` plus
`dot(column_bias, x)` broadcast to every output row. The existing ordinary
`BiasCorrectionStage` corrects activation means and cannot substitute for these
input-dependent terms. A dedicated runtime path or rank-two correction would
be required to use them in a model. No production path or model export changed,
and no ADMM search restarted for this diagnostic.

The analysis reuses the hash-verified Experiment 066 fixture and retained
Experiment 068 comparisons. CPU FP64 reconstruction reproduces their existing
weighted-error values. Corrected coefficients, input hashes, optimality checks,
and a true signed-residual before/after plot are saved in the
[complete report](../Results/068/axis-bias/report.md) and
[machine-readable summary](../Results/068/axis-bias/summary.json).

Run with `.\.venv\Scripts\python.exe tools\analyze_fit_axis_bias.py`.

Validation: four focused least-squares tests passed. The full default suite
passed 1,499 tests with 49 deselected in 98.85 seconds. Repository Ruff and
mypy (217 source files) passed. Source and saved-coefficient hashes were verified.
