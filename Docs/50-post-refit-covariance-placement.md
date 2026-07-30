# Post-Refit Covariance Placement

**Date:** 2026-07-29
**Status:** bounded screen passed; complete production transfer failed the
perplexity gate

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

## Bounded-screen decision

Promote only an explicit post-refit QKV placement for blocks 5, 11, 24, and
25 to a numbered complete-run experiment. Do not enable all-group,
all-late-block, all-QKV, or isolated-output-gated refinement.

Resident algorithm version 51 implements this as the explicit
`block_tuning.post_refit_covariance_refinement` option. Experiment 034 is the
only recipe enabling it and selects the fused-QKV owner in blocks 5, 11, 24,
and 25. The complete-run gate must still prove:

1. deterministic resident persistence and resume;
2. unchanged effective BPW and export validity;
3. the exact pre-KD quality improvement;
4. interaction with global KD and final retained quality;
5. strict artifact validation and complete GGUF lifecycle.

The selected block set is pinned-workload evidence, not a general
architecture rule. A future general selector needs a held-out
language-functional allocation stage rather than a covariance or block-output
proxy.

## Complete production transfer

Experiment 034 completed the full resident, global-distillation, packed,
checkpoint, GGUF, publication, and retained-quality lifecycle. Its interrupted
resident run adopted the 48 valid layer/group commits through block 7 and
replayed block 8 onward. Fresh strict validation then checked 712 artifacts,
all 156 active journal records, and the contiguous 26-block prefix.

The production calibration covariance reduced its own objective at every
selected placement:

| Block | Covariance error reduction |
| ---: | ---: |
| 5 | 6.45% |
| 11 | 7.50% |
| 24 | 5.09% |
| 25 | 4.02% |

These reductions are much smaller than the offline WikiText placement probe.
The probe optimized covariance captured from its WikiText functional workload,
whereas the resident path used the recipe's calibration stream. The selected
blocks therefore did not transfer across covariance sources.

The exact retained 64x128 WikiText comparison confirms the failure before
global KD:

| Metric | Experiment 022 | Experiment 034 | Change |
| --- | ---: | ---: | ---: |
| Pre-KD mean NLL | 5.611360 | 5.656276 | +0.044916 |
| Pre-KD perplexity | 273.516089 | 286.081276 | +4.59% |
| Post-KD perplexity | 228.550618 | 241.121781 | +5.50% |

Global KD improved Experiment 034 from 286.08 to 241.12 perplexity, but did
not erase the pre-KD regression. The complete-run result is better than the
all-group Experiment 033 failure (272.56 perplexity), showing that selective
placement contains the damage, not that it beats the D2 baseline.

The six 200-row task checks are a mixed secondary signal: Experiment 034
improved PIQA by 3.0 points, ARC Easy by 0.5, ARC Challenge by 1.0,
HellaSwag by 0.5, and Winogrande by 4.0, while BoolQ tied Experiment 022.
Their unweighted mean rose from 0.4692 to 0.4842. These small evaluations do
not override the pre-registered exact WikiText failure.

The representation cost remained effectively unchanged:

| Measure | Experiment 022 | Experiment 034 |
| --- | ---: | ---: |
| Effective BPW | 1.024494712 | 1.024496179 |
| Rank sum | 111,776 | 111,840 |
| GGUF bytes | 417,334,656 | 417,340,672 |

The tiny rank and byte difference comes from the fresh self-measured D2
allocation, not a covariance-refinement payload. The method adds no persisted
weight field, but the literal no-BPW-increase gate was not met.

## Final decision

Reject the selected post-refit covariance placement as a production default.
It passes persistence, resume, artifact, and export gates and improves its
local objective, but fails both the pre-KD and post-KD language-quality gates.
Keep the option explicit for research only.

The result also rejects selecting sparse factor edits using a covariance
captured from a different workload. A future sparse residual or binary-edit
selector must score candidates on held-out language behavior representative
of the intended evaluation distribution, and must validate compositions
rather than assuming singleton improvements add.
