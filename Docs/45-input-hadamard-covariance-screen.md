# Input-Only Hadamard Covariance Screen

**Date:** 2026-07-29
**Status:** completed; decisively rejected
**Model:** pinned `google/gemma-3-1b-it` revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`

## Question

The controlled covariance bound in
[Document 44](44-covariance-headroom-probe.md) found a 41.53% held-out
same-rank error-energy reduction when the full input covariance replaces its
diagonal. Can an inexpensive structured input transform make the existing
diagonal binary factorizer realize a useful part of that headroom without
changing factor storage?

## Mechanism

For an orthogonal input transform `Q`, a linear projection is unchanged when:

`x W^T = (x Q) (W Q)^T`.

The candidate therefore factorizes `W Q` and would apply `Q` to activations at
runtime. Its diagonal objective uses `diag(Q^T C_fit Q)`. Output Fisher
importance, ranks, scale representation, ADMM settings, and physical factor
bits remain unchanged. Dense reconstructed weights are mapped back with
`Q^T` only for held-out and model-splice evaluation.

The transform is a deterministic random sign vector, a fixed channel
permutation, and normalized 128-channel block Walsh-Hadamard transforms.
Both Gemma input widths in scope are exactly divisible by 128: 1,152 uses
nine blocks and O projection's 1,024 uses eight. Gate and up share the same
transform because they consume the same input tensor. Transform metadata is
derived from the global seed, block, and input role rather than stored per
weight.

This adds runtime activation arithmetic, so it is not free. The screen asks
whether the functional gain is large and seed-robust enough to justify
implementing and benchmarking that transform.

## Pre-registered protocol

`tools/probe_input_hadamard.py` compares:

- the plain diagonal baseline;
- randomized structured-Hadamard seeds `{0, 1, 2}`.

The screen uses:

- complete blocks `{0, 12, 24}`;
- transformed fused QKV, O, gate, and up groups;
- down projection reconstructed once and held tensor-identical across arms;
- raw corrected CCE output Fisher importance;
- 2,048 fit and 2,048 disjoint held-out covariance rows;
- 1% mean-diagonal covariance damping for the fit objective;
- target 1.0 BPW with identical ranks, factor seeds, and scale costs;
- 400 outer and 5 inner ADMM iterations plus two scale-fit passes;
- 12 additional disjoint WikiText sequences of 512 tokens for joint-splice
  KL;
- isolated block-output error on four of those sequences;
- paired 10,000-resample sequence-bootstrap KL intervals.

The primary functional gate requires at least two of three transform seeds to
improve joint KL by at least 5% with their paired 95% intervals entirely below
zero. The supporting covariance gate requires at least a 10% median held-out
error-energy reduction over the three seeds. Both gates must pass.

The candidate and baseline have identical factor payload bits. A passing
result still requires runtime cost measurement before adoption. A failing
result promotes the generalized covariance-weighted ADMM direction in
`NextQualityLevers.md` §13 rather than another rotation search.

## Validation

Focused CPU tests cover:

- exact transform inversion;
- orthogonality and energy preservation;
- rotated-covariance diagonal equivalence to explicit `Q^T C Q`;
- covariance trace preservation;
- transform dimension/seed validation;
- complete fused-QKV block topology.

Validation completed with 13 focused tests passing (four new transform tests
plus nine covariance/exponent regression tests), a clean focused Ruff check,
successful bytecode compilation, and a complete CLI help smoke test.

## Result

The pinned GPU run completed on 2026-07-29 without an overlapping NanoQuant
worker. The retained artifact is:

`evidence/m4/input-hadamard-probe/blocks-0-12-24-seeds-0-1-2.json`

Its SHA-256 is
`aae24435ccb8c8d9af5a3335b9803392409535fbebf607275623e66a4e93f6b8`.

All four arms use exactly 80,468,496 factor bits, or 0.999472370 BPW.
Every candidate rank matches the baseline and down projection is identical.
The transform therefore receives no hidden capacity advantage or penalty.

### Aggregate result

| Arm | Original-space normalized RMSE | Held-out covariance normalized RMSE | Held-out covariance error-energy change | Joint KL | Joint KL change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain diagonal | 0.620157 | 0.150204 | — | 0.398949 | — |
| Hadamard seed 0 | 0.623490 | 0.201059 | **+79.18%** | 0.584679 | **+46.56%** |
| Hadamard seed 1 | 0.623588 | 0.199432 | **+76.29%** | 0.610942 | **+53.14%** |
| Hadamard seed 2 | 0.623323 | 0.200246 | **+77.73%** | 0.627740 | **+57.35%** |

The paired 95% intervals for the absolute joint-KL changes are:

- seed 0: `[+0.162635, +0.211658]`;
- seed 1: `[+0.199850, +0.223932]`;
- seed 2: `[+0.201531, +0.261208]`.

The result is not seed-local. None of the 36 transformed block/group cases
improves held-out covariance error. Mean error-energy changes by projection
are +53.34% for gate, +103.62% for O, +119.92% for fused QKV, and +56.64%
for up. Every isolated block-output RMSE is also worse: the nine increases
over three blocks and three seeds range from 4.70% to 24.14%. Every block-level
KL interval lies above zero.

### Interpretation

The nearly unchanged original-space RMSE is important: an ordinary
Frobenius screen would have described these candidates as close to neutral.
The disjoint covariance and joint-splice measurements instead show large,
consistent functional damage.

This does not show that the transform fails to decorrelate activations.
It shows that decorrelation is not beneficial after imposing the current
binary-factor plus diagonal-scale representation. The original coordinate
basis contains structure that this format exploits; the randomized
Hadamard basis makes the target materially harder for the binary factors even
when its diagonal activation metric looks more regular. Exact inversion,
orthogonality, covariance-diagonal equivalence, trace preservation, identical
ranks, and agreement across all three seeds make an implementation accident
an implausible explanation.

### Verdict

Reject input-only structured Hadamard for this format. It fails both
pre-registered gates in the wrong direction, so runtime benchmarking and a
production transform path are not justified. The full-covariance real-valued
bound remains positive, but it cannot be captured by rotating the problem
back into the existing diagonal fitter. Promote the runtime-compatible
covariance-weighted binary-factorization screen described in
`NextQualityLevers.md` §13.
