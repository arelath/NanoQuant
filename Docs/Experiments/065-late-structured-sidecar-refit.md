# Experiment 065: late structured sidecar selection and refit

## Question

Can selecting an outlier sidecar from the mature factorization residual and
then refitting the factor scales improve error? Do shapes smaller than a whole
column improve the rate/error tradeoff?

## Protocol

The analysis starts from completed Experiment 058 artifacts. It keeps the
persisted binary signs, product-codebook selectors, rank, and existing outlier
sidecars fixed. Each candidate receives no more bits than one additional INT8
whole column, including one BF16 scale and its exact logical index width.

One refinement round selects a symmetric-INT8 patch from the current retained
diagonal-Fisher-weighted residual and refits the separable pre/mid/post scale
axes against the patch-subtracted target. Two rounds were tested. The shapes
were:

- whole column;
- vertical column segments: 16x1, 32x1, and 64x1;
- horizontal row runs: 1x32;
- compact tiles: 8x2 and 4x4.

Early, middle, and late blocks 0, 12, and 25 were screened for down, up, gate,
attention output, and the shared QKV owner. Functional comparisons use
previously unopened C4 validation slices and the globally tuned Experiment 058
state. This is a fixed-codebook scale-refit test, not a new 1,200-step ADMM
factorization.

## Local reconstruction results

For down projection, 16x1 was the best local shape at all three depths. Its
two-round weighted-error gains relative to scale-refit-only were 0.0820%,
0.0928%, and 0.1622% at blocks 0, 12, and 25. The whole-column gains were
0.0501%, 0.0379%, and 0.1083%. The second selection/refit round improved 16x1
by another 0.0015, 0.0037, and 0.0173 percentage points respectively.

The best local geometry depends on owner orientation:

| Owner | Best shape | Mean weighted-error gain | Whole-column gain |
| --- | --- | ---: | ---: |
| down | 16x1 | 0.1123% | 0.0655% |
| up | 1x32 | 0.3441% | 0.0996% |
| gate | 1x32 | 0.3496% | 0.0970% |
| attention output | 1x32 | 0.2360% | 0.1716% |
| shared QKV | 16x1 | 0.4946% | 0.1506% |

This is a clear representation result: a universal tile is inferior to shapes
aligned with the owner's matrix orientation.

## Held-out functional results

On C4 validation rows 456--463, down 16x1 lost decisively to the whole-column
control despite its stronger local reconstruction result:

| Candidate minus column | KL delta | 95% interval | NLL delta | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 16x1 | +0.005683 | [+0.003948, +0.007459] | +0.003226 | [+0.000761, +0.005323] |
| 32x1 | +0.001201 | [-0.000681, +0.003116] | +0.001195 | [-0.000554, +0.003039] |
| 64x1 | +0.001882 | [-0.000222, +0.004701] | +0.001334 | [-0.000785, +0.003730] |

The additional whole column itself was indistinguishable from scale refit:
KL changed by -0.000204 with interval [-0.002055, +0.001361], while NLL
changed by +0.000318 with interval [-0.002896, +0.003445].

The owner-specific composite used 1x32 for up/gate/O and 16x1 for QKV on C4
rows 464--471. Against the corresponding whole-column composite it changed KL
by +0.000141 `[-0.003735, +0.003948]` and NLL by -0.003061
`[-0.007332, +0.001347]`. Neither metric separated from zero. Against
scale-refit-only, shaped NLL improved at the point estimate by 0.004772 but KL
worsened by 0.000555; both intervals crossed zero.

## Decision

- Late selection plus scale refit produces a repeatable local reconstruction
  gain, and a second round can help, especially in later blocks.
- Narrow shapes recover more weighted residual per bit than whole columns.
- The best shape is owner-dependent: vertical for QKV/down, horizontal for
  expansion layers.
- The reconstruction gains did not establish held-out KL improvement. The
  strongest down shape regressed both functional metrics with confidence; the
  owner-specific composite was a statistical tie with a KL/NLL tradeoff.
- Do not add a structured-sidecar runtime schema or launch a full experiment
  from this evidence. A future attempt would need functionally selected shapes
  or a complete codebook-constrained refactorization with a disjoint gate.

## Evidence

- `evidence/065/late-structured-sidecar-refit-down-b0-b12-b25-v3.json`
- `evidence/065/late-structured-sidecar-refit-{up,gate,o,qkv}-b0-b12-b25.json`
- `evidence/065/late-structured-sidecar-refit-control-c4-validation456-8x512.json`
- `evidence/065/late-owner-shapes-c4-validation464-8x512.json`
- `tools/probe_late_structured_sidecar_refit.py`
