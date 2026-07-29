# Input-Only Hadamard Covariance Screen

**Date:** 2026-07-29
**Status:** harness validated; GPU measurement pending
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

Pending the pinned GPU run.
