# Experiment 080 follow-up: importance-shaped noise

## Completed result — 2026-11-16

All 520 probes completed; the embedding tensor remained exactly unchanged. The
maximum mismatch between a synthetic probe's post-cast norm and its actual
residual control was 0.0011%. Clean-control KL was 2.566e-7 nats/token.

IMatrix-shaped noise improves prediction over isotropic noise for all three
formats. It does not improve on the existing weighted-error curve for NanoQuant.
The metric below is median multiplicative prediction error: ideal 1.00, and both
underprediction and overprediction increase it.

| Format | Cases | Isotropic noise | IMatrix-shaped noise | Block + IMatrix noise | Existing weighted curve |
| --- | ---: | ---: | ---: | ---: | ---: |
| NanoQuant 080 | 15 | 1.816 | 1.243 | 1.250 | **1.115** |
| IQ4_NL | 15 | 1.659 | 1.256 | 1.402 | **1.233** |
| IQ3_S | 10 | 1.836 | 1.234 | **1.218** | 1.304 |

On the 10 projections shared by all three formats, NanoQuant has a lower
importance-weighted/raw residual-energy ratio than either standard format in
**all 10 paired comparisons**. Median ratios are 0.576 for NanoQuant, 0.686 for
IQ4_NL and 0.631 for IQ3_S. This supports the proposed difference in error
placement, without establishing the causal effect of enabling IMatrix during
compression. The fixed inverse-square-root noise model has a lower median ratio,
0.476, for every format: it places more error in low-importance columns than these
actual residuals do, and does not represent their different degrees of steering.

Aggregate predicted/actual isolated KL with this shaped noise is 0.854 for
NanoQuant, 0.663 for IQ4_NL and 0.676 for IQ3_S. The existing weighted estimator is
0.809, 0.833 and 0.815 respectively. Thus the tested shaped model has a larger
family-dependent aggregate bias, despite being better than isotropic noise.
On confirmation blocks 6 and 19, NanoQuant's median error is 1.165 with shaped
noise versus 1.115 with the existing weighted curve; the result is not limited
to the previously sampled blocks.

All six estimators order the same 34 of 35 same-projection format pairs correctly.
The exception is block 6 attention output, where IQ3_S is actually better than
NanoQuant (KL 0.009842 versus 0.016350), but the existing weighted curve narrowly
prefers NanoQuant (0.013246 versus 0.013147). Improved average accuracy did not
fix this remaining format-ordering error.

Keep the existing weighted estimator as the cheap shortlist score. These results
do not justify replacing it with the tested inverse-square-root sweep or rerunning
the full sweep in that form. Actual conditional replacements and the matched refit
canary remain the next allocation gates: the present comparison holds all other
projections clean and cannot measure their compensation in compressed 080.

Full per-projection results, seed variability, window uncertainty and the scatter
figure are in the [local report](../../Results/080/080-imatrix-shaped-noise-comparison.md).

## Question and predeclared comparison

Does shaping synthetic noise with the exact IMatrix used by NanoQuant improve
prediction of real quantization residual damage, and does the improvement differ
between NanoQuant and standard quantizers?

This is a projection-only diagnostic, not a compression run. Embeddings, tied
output weights and normalization weights remain at the clean teacher. No model
artifact is promoted by this experiment.

The [fixed plan](080-imatrix-shaped-noise-plan.json) compares reconstructed 080
residuals, IQ4_NL residuals, and IQ3_S residuals against four synthetic controls:

1. Isotropic Gaussian noise.
2. Gaussian noise scaled by the clean weights' RMS in contiguous, per-row blocks
   of 32 weights.
3. Gaussian noise divided by the square root of mean-one input-column importance.
4. The block-scaled and importance-shaped controls combined.

The importance floor is 0.01 times the tensor's mean importance, applied before
inversion. Each probe is globally normalized to the actual residual's Frobenius
norm, then cast into the model's evaluation dtype. Both raw and weighted errors
are recorded after casting. Seeds are 17, 29 and 43, with a target-dependent
deterministic offset; the same draw is used across noise modes and formats for a
given target and seed. No floor, exponent, block size or KL multiplier is fitted
to measured damage. Block scaling is a heuristic control, not an implementation
of any particular quantization grid or block format.

The actual residual control is the dequantized GGUF projection cast to the same
evaluation dtype. Each projection is swapped alone into the clean model and
restored exactly afterward. The embedding tensor is also checked for exact
equality at the end.

## Inputs and scope

- Pinned `google/gemma-3-1b-it`, revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- The completed FP16 sensitivity teacher: 16 windows of 256 tokens, 4,080 scored
  positions, full-vocabulary KL(teacher || perturbed model).
- The exact Unsloth input-column IMatrix used by 080 and the local standard
  quantizations, SHA-256
  `86a81192de49a6480f8dc19c5a7fd3f4377af59172fd4c7b8d93d296a0a18515`.
- Blocks 0, 13 and 25 reuse the earlier intervention sample. Blocks 6 and 19 are
  additional confirmation projections, not previously included in that sample.
- MLP gate, MLP down and attention output: 15 projections each for NanoQuant and
  IQ4_NL. IQ3_S covers the 10 down/output projections; gate is excluded because
  that file stores it as Q8_0 rather than IQ3_S.
- 40 actual residual cases, each with 12 noise probes: 520 scores total, plus the
  clean control. Three random seeds measure noise variability; they are not three
  independent evaluation corpora.

The original Gaussian response curve evaluated at raw error and at IMatrix-weighted
error supplies two existing-estimator baselines. The comparison records whether
either curve is extrapolated. Direct noise measurements at the matched norm do
not require curve extrapolation.

## Evaluation rules

Compare predicted/measured KL ratios and median absolute multiplicative error
per format, not just agreement of aggregate sums. Report projection rankings,
same-projection pairwise format orderings, and the confirmation blocks separately.
Record seed variation and paired-window uncertainty, but do not interpret windows
from one text sample as independent-corpus validation.

The test isolates residual shape inside individual matrices. It cannot establish
the joint cost of a mixed model or recover compensation between compressed blocks.
It is also not an IMatrix-on/off compression ablation: a greater reduction under
weighting does not by itself establish a larger causal training benefit.

## Reproduction

Run from the repository root with an idle GPU. The measurement helper holds the
cross-process device lease and atomically saves every scored probe. Repeating the
command resumes only if plan, source, sweep and helper hashes match.

```powershell
.\.venv\Scripts\python.exe tools/compare_imatrix_shaped_noise.py `
  Datasets/gemma-3-1b-it-fp16-sensitivity-16x256/measurements/307b4cf1f11bc153222ddc75e79e9da66462b497ed7f4ce131f3f3ef1fc66114/results.json `
  Docs/Experiments/080-imatrix-shaped-noise-plan.json `
  --output outputs/imatrix-shaped-noise-1b/comparison.json

.\.venv\Scripts\python.exe tools/summarize_imatrix_shaped_noise.py `
  outputs/imatrix-shaped-noise-1b/comparison.json `
  --output outputs/imatrix-shaped-noise-1b/analysis.json
```

The raw report records source/code hashes, protocol, actual tensor storage types,
clean-control scores, every window's KL, and post-cast residual metrics.

## Implementation validation

19 focused tests passed, including noise direction/norm/determinism controls,
invalid importance rejection, aggregate-versus-individual error accounting,
confirmation-set analysis, and the existing swap/weighted-error tests. Repository
Ruff and mypy passed. The full suite reported 1,866 passed, 54 deselected and the
same three existing Experiment 079 launcher failures (thin-runfile contract,
missing `ExperimentIdentity`, and missing `EXPERIMENT`). The 079 launcher was not
changed. Full output is retained with the local measurement evidence.
