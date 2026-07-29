# Attention Partition Functional Gate

**Date:** 2026-07-29
**Status:** completed negative result; do not adopt fixed QV/KO grouping
**Model:** pinned `google/gemma-3-1b-it` revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`

## Question

[The exhaustive matrix probe](40-attention-partition-topology-probe.md) found that
grouping `[Q; V]` and `[K; O.T]` reduced the global corrected-Fisher reconstruction
RMSE by 1.03% relative to the adopted `[Q; K; V]` plus `O.T` topology. This gate
asks whether that apparent win survives:

1. real, disjoint WikiText activations;
2. the coupled attention computation; and
3. end-to-end teacher KL when every attention projection is replaced.

The answer is no. QV/KO is substantially worse on every functional gate.

## Reproducible harness

`tools/probe_attention_partition_functional.py` extends the scalar-only topology
probe just enough to retain fitted dense reconstructions in memory. It restores
transposed members such as `O.T` to the model's original weight orientation, then
uses the production `DenseKlSpliceEvaluator` to install and restore the candidate
weights around each measurement.

The run used:

- corrected CCE Fisher state `evidence/m4/gemma-cce-fisher-state`;
- importance shrinkage 0.6;
- production ADMM settings: 400 outer iterations, 5 inner iterations, cubic
  penalty schedule, regularization 0.03;
- two alternating scale-fit passes;
- rank alignment 1 and a 1.0 BPW target;
- seed 0;
- all 26 attention blocks;
- the retained WikiText protocol, 12 independent sequences of 512 tokens;
- four of those held-out sequences for isolated attention-output measurements;
- eager BF16 teacher execution on CUDA.

The ignored scalar result is
`evidence/m4/factor-grouping-probe/attention-partition-functional-cce.json`.
Dense reconstructed weights are deliberately not persisted as compression or
runtime artifacts.

An initial attempt supplied the Experiment 022 teacher-cache directory. It failed
closed after factorization because that cache identity did not match the exact
request. No metric was emitted or adopted from that attempt. The completed run
built an exact in-memory teacher cache instead.

## Reconstruction check

The functional run reproduced the earlier full-model matrix result:

| Topology | Corrected-Fisher RMSE | Original-space RMSE | Actual BPW |
|---|---:|---:|---:|
| QKV/O | 0.407104 | 0.487822 | 0.999349 |
| QV/KO | **0.402891** | 0.498222 | 0.998850 |
| QV/KO relative change | **-1.035%** | **+2.132%** | -0.000499 |

The QV/KO functional loss cannot be explained by a material storage advantage for
QKV/O. QV/KO actually used 38,272 fewer bits across the whole attention inventory;
that 0.0005 BPW difference is negligible beside the measured quality regression.

## Held-out whole-model result

Both arms replace Q, K, V, and O in all 26 blocks. The teacher baseline NLL on this
slice is 3.901023 (perplexity 49.45).

| Topology | NLL | Perplexity | KL vs BF16 teacher |
|---|---:|---:|---:|
| QKV/O | 4.706732 | 110.69 | 1.281309 |
| QV/KO | 5.073019 | 159.66 | 1.631011 |
| QV/KO minus QKV/O | +0.366287 | +48.97 | **+0.349701 (+27.29%)** |

The paired sequence-bootstrap 95% interval for the KL delta is
**[+0.295702, +0.402536] nats/token**. It excludes zero by a wide margin.

The two strongest matrix-level QV/KO blocks also fail when spliced alone:

| Arm | QKV/O KL | QV/KO KL | Delta | Relative |
|---|---:|---:|---:|---:|
| block 7 | 0.060195 | 0.111254 | +0.051060 | +84.82% |
| block 17 | 0.138436 | 0.188718 | +0.050282 | +36.32% |

Their paired 95% intervals are `[+0.041957, +0.060631]` and
`[+0.040053, +0.060967]` nats/token respectively. Thus the negative full-model
result is not merely an interaction among many replaced blocks.

## Isolated attention-output result

For each reported block, only that block's four attention weights were replaced.
The model forward therefore supplied the same teacher hidden input to the selected
attention module. Normalized output RMSE was accumulated over four held-out
512-token sequences:

| Block | QKV/O | QV/KO | Relative regression |
|---|---:|---:|---:|
| 7 | 0.430264 | 0.515134 | +19.73% |
| 17 | 0.271612 | 0.324334 | +19.41% |

This locates the failure inside the coupled attention operation, before downstream
block propagation. Block 17 is especially informative: QV/KO improved its
corrected-Fisher matrix RMSE by 13.97% and changed original-space aggregate RMSE by
only -0.43%, yet its actual attention output became 19.4% worse.

## Verdict and implications

Fixed QV/KO grouping is rejected. The production QKV/O topology remains the
attention baseline.

The broader lesson is more useful than this one rejection:

- A row-stack correlation is not sufficient evidence for grouping projections
  that occupy different roles in a nonlinear operator.
- Q, K, V, and O errors have different downstream geometry. A scalar sum of
  diagonally weighted matrix errors can reward trades that are destructive after
  rotary position encoding, Q/K dot products, softmax, value aggregation, and the
  O projection.
- Original-space aggregate RMSE gave the correct global warning here, but even it
  missed block 17. Promotion of any cross-role attention topology must therefore
  include attention-output error and paired KL, not just another matrix norm.
- The exhaustive per-block Fisher oracle from Document 40 is an upper bound in the
  wrong objective. It must not be implemented as an adaptive topology selector.

## Next useful direction

Further attention-topology work should change the selection objective before
enumerating more partitions. The cheapest defensible screen is:

1. materialize candidate reconstructions for a small set of blocks;
2. measure isolated attention-output error on disjoint activations;
3. promote only candidates that improve that operator-level metric; and
4. require paired held-out KL before changing persisted grouping contracts.

The existing 15-partition matrix enumeration remains useful for proposing
candidates, but not for selecting them. Given that the attention side is already a
minority of the measured total KL budget, broad production work should remain
focused on higher-value MLP recovery and allocation unless an operator-level
attention candidate produces a much larger win.
