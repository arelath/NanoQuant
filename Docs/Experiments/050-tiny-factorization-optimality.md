# Experiment 050: Tiny Full-Rank Factorization Optimality

## Question

When NanoQuant factorizes a matrix at full rank, is the remaining error mainly a
limit of the binary-diagonal-binary format, or can a substantially better fit of
the same format exist beyond the production ADMM solution?

This experiment deliberately uses tiny square matrices so much stronger
discrete searches become possible. It is a solver diagnostic, not a compressed
model candidate, and therefore does not consume held-out language-quality data.

## Representation and objective

Every arm uses the existing representation at full rank `R = N`:

```text
W_hat = diag(scale_post) U diag(scale_mid) V diag(scale_pre)
U, V in {-1, +1}
```

Synthetic targets use unit importance. Real targets are deterministic crops of
block-12 `q_proj` from the pinned Gemma-3-1B checkpoint and use the matching
full-Fisher input/output importance vectors with shrinkage 0.6. All comparisons
use the exact separable Fisher-weighted squared-error objective and report its
normalized square root (NRMSE) for readability.

## Search levels

### Production control

- Production ADMM, 800 outer and 5 inner iterations, cubic schedule and
  regularization 0.03.
- Sixteen seeds for the main comparison; 32 seeds in the deep 3x3 confirmation.
- Sixteen scale-ALS passes; 32 in the deep confirmation.

The reported production result is the best seed, not merely seed zero. This is
especially important at tiny dimensions, where ADMM is much more seed-sensitive
than it is on production-sized matrices.

### Exhaustive-sign 3x3 oracle

The sign gauges can be removed without losing represented matrices: fix the
first row and first column of `U` positive and the first row of `V` positive,
while retaining signed pre/mid/post scales. A 3x3 problem then has

```text
2^((3-1)^2) * 2^(3(3-1)) = 1,024
```

distinct sign configurations. Every one is enumerated. For each configuration,
the deep confirmation runs 64 continuous scale initializations and 256 exact
ALS passes. This is exhaustive over the discrete signs but, because the scale
problem is multilinear, remains a multistart numerical oracle rather than a
formal proof of the continuous global optimum. Three format-generated controls
with known zero error validate it: the oracle reaches NRMSE 0 to 2.6e-5.

### Brute-force block-coordinate 10x10 search

Exhausting all 200 signs is impossible. Instead, with `V` fixed, each entire row
of `U` has only `2^10 = 1,024` patterns. The probe enumerates every pattern and
its exact optimal row scale, independently selects the best pattern for all ten
rows, then performs the symmetric exhaustive update for every column of `V` and
its input scale. `scale_mid` is solved globally and the cycle repeats up to 32
times.

This can cross correlated ten-bit barriers that one-bit coordinate descent
cannot. It is combined with 4,096 random sign candidates, evolution, exact
one-bit descent, and the best production basin. A confirmation also starts the
block search from all 16 ADMM basins. It is still only block-coordinate and is
not a global 10x10 oracle; the known-zero planted controls explicitly measure
that limitation.

## Evidence

- Deep 3x3 exhaustive-sign confirmation:
  `evidence/050/tiny-factorization-exact3-deep.json`
- Main 10x10 block-coordinate screen:
  `evidence/050/tiny-factorization-block-coordinate10.json`
- All-ADMM-basin 10x10 confirmation:
  `evidence/050/tiny-factorization-diverse-basins10-v3.json`
- Initial no-block-search comparison:
  `evidence/050/tiny-factorization-optimality.json`
- Reproducible tool: `tools/probe_tiny_factorization_optimality.py`

## 3x3 results: a real global-style gap

Deep confirmation results use the best of 32 production seeds:

| Target | Production NRMSE | Exhaustive-sign NRMSE | Squared-error reduction |
|---|---:|---:|---:|
| Gaussian 0 | 0.108437 | 0.049208 | 79.41% |
| Gaussian 1 | 0.089525 | 0.089525 | 0.00% |
| Gaussian 2 | 0.346319 | 0.069721 | 95.95% |
| Real Fisher crop 0 | 0.157667 | 0.103268 | 57.10% |
| Real Fisher crop 1 | 0.253960 | 0.065102 | 93.43% |
| Real Fisher crop 2 | 0.307653 | 0.267675 | 24.30% |
| Format-generated 0 | 0.000134 | approximately 0 | approximately 100% |
| Format-generated 1 | 0.000848 | 0.000026 | approximately 100% |
| Format-generated 2 | approximately 0 | 0 | approximately 100% |

Five of six non-planted targets have substantially better same-format fits than
production ADMM finds. Increasing from 16 to 32 ADMM seeds improves some
production results but does not close those gaps. Increasing the exhaustive
oracle from 16 starts/64 passes to 64 starts/256 passes leaves its non-planted
answers essentially unchanged.

This falsifies the strongest possible claim that the production fitter is
always near the representational limit. At very small dimension it often is
not.

## 10x10 results: usually close to a strong local search, not globally certified

| Target family | Cases improved by exhaustive row/column updates | Best squared-error reduction |
|---|---:|---:|
| Gaussian | 0 / 3 | 0.00% |
| Real Fisher crop | 1 / 3 | 3.12% |
| Format-generated, known optimum zero | 1 / 3 | 13.93% |

The improved real crop moves from NRMSE 0.185248 to 0.182336. The improved
planted case moves from 0.142319 to 0.132031. Random evolution and exact one-bit
descent alone improve none of the nine best-of-16 ADMM controls; the gains come
specifically from exhaustive multi-bit row/column moves. Starting from all 16
ADMM basins reproduces the real-crop gain and finds no better answer on its
Gaussian and planted confirmation cases.

However, all three planted 10x10 targets have a known zero-error representation,
while production and the stronger search remain at NRMSE 0.112–0.169. The
10x10 procedure is therefore demonstrably not a global oracle. A tie means only
that ADMM is close to this aggressive practical search, not necessarily close
to the format's true optimum.

## Interpretation

The answer is size- and target-dependent:

1. **A better same-format fit can absolutely exist.** The exhaustive 3x3 result
   is too large and too consistent across Gaussian and real crops to dismiss as
   numerical noise.
2. **Production ADMM becomes much harder to beat by practical searches at
   10x10.** Seven of nine cases are already fixed points of random evolution,
   one-bit descent, and exhaustive ten-bit row/column updates.
3. **It is not globally optimal at 10x10.** The planted controls prove large
   undiscovered basins exist, and one real crop exposes a smaller practical gap.
4. **This does not contradict the production-sized evidence.** The retained
   large Gemma probes found only about 0.3% recovery from scale ALS, one-bit
   descent, STE, multistart, and basin hopping. Tiny matrices have fewer averaging
   effects, more rank degeneracy, and much stronger seed sensitivity. Experiment
   050 shows that the old evidence supports “hard to improve with tested local
   methods,” not a global-optimality claim.

## Next useful experiment

Full row enumeration costs `2^R` and cannot scale to production ranks. Its useful
generalization is **component-window enumeration**: partition a row's rank bits
into windows of 8–12 components, enumerate all `2^B` joint flips in one window
while holding the remaining components fixed, sweep windows and rows, then do
the symmetric `V` update and refit scales. This preserves the deployed format
and directly tests the correlated-bit mechanism found here at manageable cost.

The next gate should be a 32x32 or 64x64 real Fisher crop comparing one-bit
descent with 8-, 10-, and 12-bit window enumeration. Only if the multi-bit arm
produces a repeatable material gain should it be attempted on a complete Gemma
projection and then judged by held-out splice KL. No compressed-model quality
benchmark is warranted from this diagnostic alone because it has not produced
a deployable model change.
