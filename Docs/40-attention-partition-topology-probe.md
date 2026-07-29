# Exhaustive attention partition topology probe

## Question

The reciprocal factor-grouping probe showed that `[Q; K] + [V; O.T]` can
substantially improve corrected-Fisher reconstruction in many blocks, but
damages O enough to lose globally when applied to every block. This follow-up
asks whether another partition of Q, K, V, and transposed O provides a better
fixed topology, and whether per-block topology selection has additional
headroom.

This is reconstruction evidence only. No production topology, resident schema,
runtime, packed format, or algorithm version is changed by this result.

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Objective: corrected 256-sample `gemma-cce-fisher-state`, with 0.6
  importance shrinkage
- Factorization: production ADMM, 400 outer by 5 inner iterations, cubic
  schedule, 0.03 regularization, and wide-matrix transposition
- Scale fit: production two-pass alternating fit
- Budget: at most 1.0 physical BPW for every source-weight group, including
  16-bit factor scales and an additional 1,152-element scale whenever a group
  mixes distinct Fisher input profiles
- Rank alignment: one, to isolate topology from rank-quantum rounding
- Seed: zero for the exhaustive sweep; seed one for confirmation arms
- Checkpointing: one durable JSON record per unique member subset, protected by
  the normal CUDA device lease

After transposing O, all four matrices have a 1,152-wide factor axis. Four
members have Bell number 15 set partitions. The probe evaluates all 15:

- four singletons;
- each of the six possible pairs plus two singletons;
- the three pair/pair partitions;
- each of the four triples plus one singleton; and
- one four-member group.

`tools/probe_factor_grouping.py` now supports `attention-partitions` and caches
each member subset once per block. The full seed-zero output is
`evidence/m4/factor-grouping-probe/attention-partitions-cce.json`.

## Fixed-topology result

Only one fixed alternative beats the adopted QKV-plus-O topology globally:

| Fixed topology | Global weighted RMSE | Change vs QKV/O | Actual BPW |
|---|---:|---:|---:|
| QV / KO | 0.402891 | **-1.035%** | 0.998850 |
| QKV / O | 0.407104 | baseline | 0.999349 |
| QK / VO | 0.408553 | +0.356% | 0.998850 |
| QKVO | 0.415431 | +2.045% | 0.999750 |
| KVO / Q | 0.430785 | +5.817% | 0.999181 |
| QV / K / O | 0.434393 | +6.703% | 0.999278 |
| Four singletons | 0.507773 | +24.728% | 0.999316 |

The remaining eight fixed partitions regress by 8.8% to 24.4%.

QV/KO means:

```text
[Q; V]       share one right/input-side binary basis
[K; O.T]     share one reciprocal basis
```

It wins 14 of 26 blocks. Its global result was repeated over all 26 blocks with
an independent ADMM seed:

| Seed | QKV/O weighted RMSE | QV/KO weighted RMSE | Change | Blocks won |
|---|---:|---:|---:|---:|
| 0 | 0.407104 | 0.402891 | -1.0349% | 14/26 |
| 1 | 0.407120 | 0.402903 | -1.0358% | 14/26 |

The agreement is much tighter than the measured topology gain. This is a
stable numerical result, not initialization noise.

### Member tradeoff

QV/KO redistributes corrected-Fisher error as follows:

| Projection | QKV/O RMSE | QV/KO RMSE | Change |
|---|---:|---:|---:|
| Q | 0.547562 | 0.533092 | -2.64% |
| K | 0.317908 | 0.285836 | -10.09% |
| V | 0.333038 | 0.276339 | -17.02% |
| O | 0.471078 | 0.531741 | +12.88% |

The earlier QK/VO candidate improved Q/K/V but worsened O by about 21%.
QV/KO preserves most of the V benefit while cutting the O penalty to 12.9%,
which is enough to win the global corrected-Fisher objective.

Plain original-space RMSE moves in the opposite direction, from 0.487822 to
0.498222 (+2.13%). That is expected for an objective-weighted topology but
makes held-out activation and model-output validation mandatory.

## Exhaustive per-block ceiling

Choosing the lowest corrected-Fisher error among all 15 partitions in every
block gives:

| Policy | Global weighted RMSE change vs QKV/O |
|---|---:|
| Fixed QV/KO | -1.035% |
| Per-block choice between QKV/O and QK/VO | -3.320% |
| Per-block choice among all 15 partitions | **-3.626%** |

The exhaustive search adds only 0.316% beyond the simpler two-topology oracle.
Its winners are:

| Winning topology | Blocks | Count |
|---|---|---:|
| QK / VO | 1, 3, 6, 8, 11, 12, 13, 15, 16, 19, 20, 21, 22, 23, 24, 25 | 16 |
| QKV / O | 0, 2, 4, 9, 10, 14 | 6 |
| QV / K / O | 5, 18 | 2 |
| QV / KO | 7 | 1 |
| KVO / Q | 17 | 1 |

Only four blocks improve beyond the QKV/O-versus-QK/VO choice:

| Block | Two-topology choice | Exhaustive winner | Incremental change |
|---|---|---|---:|
| 5 | QKV/O | QV/K/O | -1.156% |
| 7 | QK/VO | QV/KO | -1.674% |
| 17 | QK/VO | KVO/Q | -3.192% |
| 18 | QKV/O | QV/K/O | -0.036% |

Independent-seed full-partition repeats reproduce the winners and deltas:
block 5 reaches -1.180%, block 7 -8.758% versus QKV/O, block 17 -18.53%, and
block 18 -0.04%. Block 18 is numerically stable but too small to be useful.

## Representation and runtime implications

QV is an ordinary shared-input group and fits the existing group abstraction.
KO is different: K consumes the block input while O consumes the attention
output, and O participates transposed. A shared binary basis would be used as
K's right factor and O's left factor. Implementing it requires:

- a reciprocal/cross-orientation factor owner rather than only row-stacked
  parallel projections;
- atomic tuning and commit ownership across two serial attention operations;
- distinct absorbed scale vectors for K's input side and O's output side;
- group-aware outlier semantics that do not incorrectly zero both members;
- logical, packed, GGUF, PyTorch, CUDA, and llama.cpp support; and
- a topology identity that prevents incompatible resume or artifact adoption.

The heterogeneous per-block oracle additionally requires five topology kinds.
Its 0.316% advantage over a two-topology policy is unlikely to justify that
format and tuning complexity without a much larger held-out quality gain.

## Matrix-stage decision and next gate

- Record fixed QV/KO as a **successful reconstruction candidate**. It is the
  only universally applied alternative among all 15 partitions that beats
  QKV/O under the corrected-Fisher objective.
- Do not implement reciprocal factor ownership yet. Attention accounts for
  only about 13% of the retained type-summed splice KL, and QV/KO worsens raw
  reconstruction and O specifically.
- Next, evaluate QV/KO on retained calibration activations and a disjoint
  held-out activation partition, including attention-output error rather than
  only four independent linear objectives.
- If that passes, run exact unit/block splice KL for the affected attention
  units. Promotion requires a paired held-out improvement large enough to pay
  for the new representation and runtime complexity.
- Treat the 15-way per-block oracle as a planning ceiling. Do not encode its
  five topology types unless held-out block-output evidence materially exceeds
  the simpler fixed QV/KO or QKV/O-versus-QK/VO policies.

The broad compression goal remains open. This result identifies a real new
candidate and also falsifies 13 fixed alternatives, but it is not sufficient
evidence for a production format change.

## Functional-gate outcome

The next gate is now complete in
[Document 41](41-attention-partition-functional-gate.md). QV/KO fails decisively:
it raises full-attention held-out teacher KL by 27.29%, with a paired 95%
interval wholly above zero, and worsens isolated attention-output RMSE by about
19.5% in both tested high-gain blocks. QV/KO is therefore a successful
matrix-objective probe but a rejected compression topology. The per-block
matrix oracle must not be promoted without a new operator-level selection
objective.
