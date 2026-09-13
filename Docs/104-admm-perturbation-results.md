# Experiment 068: cancelled 10% perturbation search and residual anatomy

Date: 2026-09-13. Status: cancelled at the user's request after 38.2380 minutes.
This is a partial experiment, not a completed one-hour run.

The search completed 172 Bernoulli 10% perturbations and one unperturbed
candidate-initialized control, each with 800 ADMM outer iterations and two
scale-fit passes. That totals 138,400 ADMM iterations. The inherited Experiment
067 winner remained best. The unfinished candidate was excluded; its completed
initial checkpoint and all completed fits remain preserved.

| Fit | Independent CPU FP64 weighted error | Weighted NRMSE |
| --- | ---: | ---: |
| Original 800-step baseline | 5.0857531418 | 0.4884239712 |
| Inherited winner / overall best | 4.8500448898 | 0.4769712602 |
| Unperturbed 800-step ADMM restart | 5.0864755937 | 0.4884586613 |
| Best 10% + 800-step ADMM | 5.0833959182 | 0.4883107669 |

The best new candidate improves baseline weighted error by 0.046350%, but has
4.811317% more error than the inherited winner. Twenty-four candidates beat the
baseline; none beat the incumbent. The unperturbed restart also lost the long
fit's advantage. This result applies to the ordinary cubic schedule restarted
from zero, not all possible candidate-initialized ADMM continuation policies.
Both radius and refinement method changed relative to Experiment 067's 5%
direct-binary-refinement phase, whose best error was 4.8505008966.

There were 172 distinct canonical final sign hashes, not proven independent
basins. Initial flips ranged from 926,728 to 931,244. Final raw corresponding-entry
distance from the parent ranged from 4,637,296 to 4,651,837 signs, out of
9,289,728; sign gauges are not removed from that distance. Median complete
proposal time was 13.126 seconds. Evidence retains 3,264,441,600 bytes of states.
CUDA allocator peaks were unavailable after termination; host-memory samples
remain preserved. No new neighborhood audit or optimization ran after cancellation.

The user's error-location question was answered on CPU using a preserved
25.61-minute snapshot. For the inherited best, the worst 10% of rows account
for 10.88% of Fisher-weighted error and the worst 10% of columns account for
15.06%. Raw shares are 13.55% and 13.98%, respectively. Error is broadly spread,
with modest column concentration. Seven fixed outlier columns are excluded from
these axis counts; their residual error is zero. This does not rule out structure
along combinations of columns. At the user's request, heatmaps were rescaled to
a common +/-25% of each panel's mean squared error, with no clipped bins, while
preserving the exact analyzed snapshot.

The search adapter reuses production solve/projection kernels and matches the
production random-initialized path exactly in CPU FP32 and BF16 tests. Production
factorization, resident versions, exports, and model quality gates are unchanged.
The original numerical runner is preserved as `search-runner.py` with its hash
bound in the final report; the only subsequent search-runner change prevents
accidental restart of a directory carrying a cancellation receipt.

Validation: 30 focused checks passed before the run. The default full suite
passed 1,494 tests with 49 deselected in 99.74 seconds. After adding the cancellation
guard, all 12 focused adapter tests passed, including a test that cancelled runs
return before source loading or CUDA acquisition. Repository Ruff and mypy
(217 source files) passed. All completed initial/final checkpoints and frozen
numerical sources passed fresh hash checks; final comparisons used CPU FP64.

See the [frozen protocol](103-hour-admm-perturbations.md),
[partial results and plots](../Results/068/report.md), and
[residual analysis with corrected heatmaps](../Results/068/residual-patterns/report.md).

Best retained factors: `evidence/068/admm-perturb/best.safetensors`, SHA-256
`f8a9d27b3ef4987bd3a3b59f6705aec2c65384742a985272274e9c7f562054e3`.

Best new candidate: `evidence/068/admm-perturb/candidate-0124.safetensors`, SHA-256
`9433d06dced375d2019b9976c3a4db52e23b56e1d7699236fc49dfbd97646f47`.
