# Post-Refit Covariance Placement

**Date:** 2026-07-29  
**Status:** bounded screen passed; production integration and complete-run gate pending

## Question

Experiment 033 proved that dense-covariance sign/scale refinement improves its
local objective but harms language quality when it initializes factorized
tuning. This study held Experiment 022's ranks, outliers, patches, and factor
format fixed and moved refinement after factorized tuning and post-block
refit, before global KD.

The analysis tool is
`tools/probe_covariance_refinement_placement.py`. It loads the immutable
pre-KD Experiment 022 factors, retains every non-factor addition exactly, and
changes only binary signs and the existing pre/mid/post scale vectors.

## Broad placement failures

The representative all-group screen on blocks 0, 12, and 24 reduced held-out
covariance error by 48.48%, but joint KL was effectively flat and slightly
worse. Block 0 regressed, block 12 was inconclusive, and block 24 improved.

The apparent late-depth effect did not generalize:

- jointly refining all eligible groups in blocks 20–25 worsened KL by 16.48%;
- refining fused QKV in all 26 blocks worsened KL by 26.96% and NLL by
  0.26156;
- choosing all 15 blocks whose isolated QKV block-output error improved still
  worsened splice KL by 10.15%;
- on the exact complete frozen model that 15-block gate improved perplexity
  only 0.60%, substantially less than the best singleton.

These results reject covariance reduction, depth, and isolated block-output
improvement as sufficient placement rules. Improvements are strongly
non-additive across blocks.

## Direct per-block decomposition

The 15 isolated-output-selected QKV candidates were evaluated one at a time
inside the actual complete Experiment 022 pre-KD frozen model on the exact
retained 64×128 WikiText protocol.

| Block | Perplexity change |
| ---: | ---: |
| 5 | −3.192% |
| 24 | −1.998% |
| 11 | −1.575% |
| 25 | −1.246% |
| 4 | −0.901% |
| 3 | −0.791% |
| 2 | −0.662% |
| 21 | −0.634% |
| 6 | −0.494% |
| 18 | −0.203% |
| 22 | −0.057% |
| 20 | +0.533% |
| 17 | +1.419% |
| 12 | +1.972% |
| 8 | +2.109% |

The four strongest singletons, blocks 5, 11, 24, and 25, were retained for
the composition gate.

## Selected composition

The selected candidate refines only the fused-QKV owner in blocks 5, 11, 24,
and 25 after block-local tuning/refit.

On the actual complete pre-KD model, using the canonical token hash
`sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`:

| Metric | Experiment 022 baseline | Selected candidate | Change |
| --- | ---: | ---: | ---: |
| Mean NLL | 5.612664 | 5.542392 | −0.070273 |
| Perplexity | 273.872886 | 255.287808 | −18.585078 (−6.786%) |

The paired baseline differs slightly from the earlier retained pre-KD
Experiment 022 measurement of 273.516089, so the same-process paired delta is
authoritative.

Three disjoint 12×512 functional slices all improved:

| Functional slice | Relative KL | NLL change | 95% KL-delta interval |
| --- | ---: | ---: | ---: |
| rows 20–31 | −3.912% | −0.060835 | [−0.032301, −0.008842] |
| rows 32–43 | −4.595% | −0.070014 | [−0.030871, −0.009973] |
| rows 44–55 | −2.826% | −0.082994 | [−0.022635, −0.000276] |
| Aggregate, 36 sequences | −3.804% | −0.071281 | [−0.024183, −0.011014] |

The aggregate interval used 50,000 paired bootstrap resamples. All ranks,
outliers, patches, representation fields, and physical bits are identical.

## Decision

Promote only an explicit post-refit QKV placement for blocks 5, 11, 24, and
25 to a numbered complete-run experiment. Do not enable all-group,
all-late-block, all-QKV, or isolated-output-gated refinement.

The complete-run gate must still prove:

1. deterministic resident persistence and resume;
2. unchanged effective BPW and export validity;
3. the exact pre-KD quality improvement;
4. interaction with global KD and final retained quality;
5. strict artifact validation and complete GGUF lifecycle.

The selected block set is pinned-workload evidence, not a general
architecture rule. A future general selector needs a held-out
language-functional allocation stage rather than a covariance or block-output
proxy.
