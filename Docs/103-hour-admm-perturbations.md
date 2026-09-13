# Experiment 068: 10% sign perturbations followed by 800-step ADMM

Protocol frozen before the timed run on 2026-09-13. Results will be recorded in
`Results/068/report.md`; this protocol is hash-bound and remains unchanged.

The user requested another one-hour search using larger, 10% sign perturbations,
then 800-step ADMM refinement. Reuse Experiment 067's best fit and Experiment
066's exact full Gemma block-12 `mlp.down_proj` fixture: 1152 by 6912, rank 1152,
pinned revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`, retained Experiment
041 diagonal-Fisher objective, and seven unchanged outlier columns. This is
additional search time on an inherited fit, not a fresh one-hour end-to-end fit
or a full-model quality measurement.

## Candidate initialization

The production ADMM entry point always starts randomly. The experiment-only
adapter in `tools/probe_hour_admm_perturb.py` uses the same solve, SVID projection,
update order, and export operations with a candidate-derived initialization.
CPU tests compare the adapter's random-initialized path exactly with production.
No production code, resident algorithm version, or persisted model format changes.

Given exported signs A, B and row/component/column scales post, mid, pre, form:

```
L = sqrt(output_importance)[:, None] * post[:, None] * A
R = B * mid[:, None] * pre[None, :] * sqrt(input_importance)[None, :]
n = column_norm(L)
L = L / n
R = R * n[:, None]
```

The product preserves the candidate in Fisher-weighted coordinates before BF16
rounding. Unit left columns account for the column normalization in ADMM export.
Signed scales remain signed. Convert lifted latents to BF16, apply the existing
initial SVID projection, and initialize duals as latent minus projection, exactly
as production initializes its random latents. No old latent or dual state is
available or reused. Restart the cubic penalty schedule at zero for each fit.

## Fixed search policy

1. Start the clock after input verification/loading and incumbent scoring.
2. Run one unperturbed candidate-initialized ADMM control, seed 67999.
3. Repeatedly clone the best eligible state and flip each sign independently
   with Bernoulli probability 0.10 in both factors. Expected total changes are
   928,972.8 of 9,289,728 signs. Preserve all three starting scale vectors.
   Use seed `68000 + attempt_index` for each perturbation and ADMM generator.
4. Each candidate receives exactly 800 outer ADMM iterations, five inner power
   iterations, regularization 0.03, cubic rho, native matrix orientation, BF16
   factors, and IEEE FP32 solves. No early stopping, direct sign descent, tabu,
   adaptive horizon, or extra finalist ADMM is applied. The ordinary two-pass
   IEEE scale fit follows, matching the original baseline's export treatment.
5. Score all candidates with the same IEEE FP32 reconstruction and FP64 energy
   reduction. Retain a monotonically best fit; track the best 10% result
   separately even if it is worse than the inherited fit or unperturbed control.
6. All control, perturbation, ADMM, scoring, hashes, and checkpoint writing count
   against 3600 seconds. Admit another fit only if the largest observed complete
   attempt duration times 1.2 plus two seconds fits; the initial estimate is 15
   seconds. Check the deadline at every ADMM iteration. A late result cannot
   replace the best. The search can stop slightly early at a complete-fit boundary.

Save each initial candidate and completed final candidate, exact flip counts,
parent/hash, deterministic seeds, ADMM residual traces, initial/post-ADMM/final
errors, sign hashes, raw corresponding-entry sign distances, and wall time.
Canonical hashes remove sign gauges; distances do not. Neither establishes
independent attraction basins. These are correlated warm starts around the best
fit, not independent random initializations of the full factor space.

Atomic reports and an append-only event log retain completed work. Resume checks
input/source hashes and preserves the original deadline; interrupted proposals
are skipped. If the initial control is interrupted, it is retried only if time
remains. CUDA runs require the device lease and process/device preflight.

## Comparison and audit

After the search, independently reconstruct the baseline, inherited best,
unperturbed control, best perturbed result, and overall best in FP64. Report
weighted error, weighted NRMSE, raw NRMSE, and relative gains. Verify hashes of
every initial/final checkpoint and numerical sources. Audit the full fixed-scale
single-bit neighborhood of the overall and best perturbed fits in FP64, using
absolute energy tolerance 1e-10; this is not a joint local/global optimality proof.
Final audits do not optimize factors or count as additional search time.

Compare with Experiment 067's 5% results while making clear both radius and
refinement method changed. Report completed proposals, distinct canonical hashes,
throughput, trajectory, GPU allocator peaks, and state bytes. Broad independent
random starts and alternative warm-start penalty schedules remain separate work.

Run a 60-second lifecycle canary first; discard its candidates for the main run.
Then run the default one-hour command:

```powershell
.\.venv\Scripts\python.exe tools\probe_hour_admm_perturb.py
```
