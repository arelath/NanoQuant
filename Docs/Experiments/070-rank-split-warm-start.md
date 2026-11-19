# Experiment 070: rank-split warm-start controls

The [initial screen](070-rank-split-probes.md) compared one stage to two stages
that had 32 fewer total components and were fit sequentially from scratch.
This follow-up isolates those two changes on block 25's up, attention-output,
and stacked QKV projections. No resident configuration or artifacts change.

For each owner and both seeds, freshly fit and tune four arms:

| Arm | Initialization | Purpose |
| --- | --- | --- |
| Full | Original-rank single-stage fit | Original budget reference |
| Reduced | Single-stage fit with 32 fewer components | Rank penalty |
| Warm | Split the reduced fit's components; copy and untie its outer scales | Effect of giving the two stages independent scales from the known solution |
| Scratch | Sequentially fit the two stages to the target and residual | Initialization penalty versus Warm |

Warm and Scratch have identical component counts and stored bit costs, within
the Full arm's cap. Reduced has the same component count but no extra outer
scales. Use the previously best local split for each owner, fixed across
both seeds. Use the prior screen's 800-step ADMM configuration and same
64 joint-tuning steps and captured activations. All arms are scored by
reconstructing from persisted sign and BF16-scale tensors. The same eight
WikiText validation windows as the first screen are used for diagnosis, not
for a new quality claim.

Before tuning, also split the Full arm without dropping any components. Audit
its dense reconstructed weight and measure the difference in BF16 forward
output. Split and unsplit forward operations are not bitwise equivalent in
BF16: the two-stage path rounds each stage output before addition. This
full-rank replay is an equivalence check, not an equal-budget candidate.

## Results

All 24 arms completed. Across both seeds and all owners, the same-rank split
reproduced the original weight to within 4.3e-7 relative Frobenius error.
The BF16 forward path differed by 0.349–0.367% relative output RMSE. The full
and scratch arms reproduced the previous local results.

Mean held-out activation NMSE across the two seeds (lower is better):

| Owner | Full | Reduced | Warm | Scratch |
| --- | ---: | ---: | ---: | ---: |
| Up | 0.09954091 | 0.10316529 | 0.10332282 | 0.10912650 |
| Attention output | 0.01172460 | 0.01244081 | 0.01269834 | 0.01272566 |
| Stacked QKV | 0.06843249 | 0.07298757 | 0.07280186 | 0.07578899 |

Removing 32 components alone increased error by 3.64%, 6.11%, and 6.66%
respectively. At the exact same ranks and cost, Scratch was worse than Warm
by 5.62% (up), 0.22% (output), and 4.10% (QKV). Warm versus Reduced was
+0.15%, +2.07%, and -0.25%, respectively. Thus the scratch initialization
explains a substantial part of the up/QKV gap, but almost none of the average
output gap. The independently trainable scales did not offer a clear benefit
over the reduced single-stage control in this tuning budget. All warm arms
still trailed Full. These are local results, not end-to-end KL/NLL or a basis
for changing the production allocator.

Raw per-seed metrics, payload hashes, and protocol:
`evidence/070/rank-split-warm-start-20261111`.
Five focused tests, repository Ruff, and mypy passed. The bounded full suite
stopped at 1,277 passes and the same four environment-dependent failures as
the previous screen (three denied hardlinks and one in-repository temp-path
assertion), with 53 deselected. See the run's `regressions.xml`.

## Reproduction

Reproduce from the repository root with a free CUDA device and a new output
directory. The tool acquires the same CUDA lease as the initial screen.

```powershell
$warmSnapshot = 'C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'
.\.venv\Scripts\python.exe tools/probe_residual_split_warm_start.py --previous evidence/070/rank-split-screen-20261111 --snapshot $warmSnapshot --output evidence/070/rank-split-warm-start-replay
```
