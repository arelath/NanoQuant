# Exact row/column encoding coverage on the retained best fit

Date: 2026-09-13. Supplementary CPU diagnostic for Experiment 068.

Eliminating half of the current best fit's Fisher-weighted squared error requires
separately encoding **553 rows or 2,887 columns**, assuming those entries become
exact and the remaining signs and scales stay fixed. The column selection covers
41.81% of the active matrix. Error is too broadly distributed for a small row or
column sidecar to eliminate most of it.

The matrix has 1,152 rows and 6,912 columns. Seven previously protected outlier
columns are excluded, leaving 6,905 active columns. These are additional selected
columns; the seven existing outliers are not counted again.

| Weighted squared error removed, at least | Rows alone | Active entries via rows | Columns alone | Active entries via columns |
| --- | ---: | ---: | ---: | ---: |
| 25% | 272 | 23.61% | 1,262 | 18.28% |
| 50% | 553 | 48.00% | 2,887 | 41.81% |
| 75% | 842 | 73.09% | 4,707 | 68.17% |

Rows alone and columns alone are alternatives, not counts to combine. Within
each single-axis choice, these are the exact smallest counts crossing each
threshold: rank the rows or columns by their summed residual energy and take
the shortest prefix.

The mixed greedy selection chooses the greatest remaining error per newly
stored entry, subtracting intersections from both energy and storage. It chooses
the same columns-only selections at 25% and 50%. At 75% it chooses 6 rows and
4,696 columns, covering 68.18% of entries, slightly more than the columns-only
choice. This heuristic gives no meaningful weighted storage improvement and
does not prove a global mixed-selection optimum.

For ordinary, unweighted squared error:

| Error removed, at least | Rows alone | Columns alone | Mixed greedy rows + columns | Mixed active entries |
| --- | ---: | ---: | --- | ---: |
| 25% | 228 | 1,315 | 80 + 864 | 18.59% |
| 50% | 488 | 2,857 | 157 + 2,173 | 40.81% |
| 75% | 774 | 4,565 | 211 + 4,027 | 65.95% |

The original baseline behaves similarly. Its weighted thresholds are
272 / 554 / 843 rows or 1,241 / 2,862 / 4,688 columns, respectively.

## Meaning and representation cost

This measures full selected-entry error removal, not the additive bias vectors
from the previous diagnostic. It does not run ADMM, refit a smaller matrix,
change ranks, or measure a lossy side encoder. Lossy storage would generally
recover less than the ideal coverage measured here; refitting could change the
remaining residual.

The denominator is the original fit's squared residual energy. Removing
25%, 50%, or 75% of that energy reduces NRMSE by approximately 13.4%, 29.3%,
or 50%, respectively, with the original full-target normalization retained.

At two bytes per separately stored entry, the weighted columns-only selections
would require 2.908 / 6.652 / 10.845 MB of payload. These are hypothetical BF16
payload estimates; they do not assert exact error removal after BF16 rounding.
They exclude index metadata and potential savings from removing unused factor
rows or columns. The JSON's additional-BPW values assume the original base fit
is retained. No production representation or model export was changed.

## Evidence and validation

The script reuses the hash-verified fixture and comparison records retained by
Experiment 068. CPU FP64 reconstruction reproduces the best and baseline errors.
All 36 thresholds (two fits, two objectives, three methods, three targets) are
independently checked by summing the unselected row/column complement; each
preceding prefix remains below its target. Original row and column IDs are saved.

- [Detailed results](../Results/068/axis-excision/report.md)
- [Exact metrics and selected indices](../Results/068/axis-excision/summary.json)
- [Coverage curves](../Results/068/axis-excision/coverage.png)
- [Reproduction script](../tools/analyze_fit_axis_excision.py)

Run with `.\.venv\Scripts\python.exe tools\analyze_fit_axis_excision.py`.

Nine focused tests passed, including comparison with exhaustive single-axis
subset selection, explicit mixed union masks and marginal densities, original
column mapping, and invalid-input checks. The full suite passed 1,508 tests with
49 deselected in 151.16 seconds. Repository Ruff and mypy (217 source files)
passed. Validation and a fresh source/fixture/state/curve hash audit are recorded
in `Results/068/axis-excision/verification.json`.
