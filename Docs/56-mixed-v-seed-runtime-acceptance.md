# Mixed-V Seed and Runtime Acceptance

**Date:** 2026-07-30
**Status:** accepted as a selective compact `down_proj` representation with
load-time packed predecode; uniform application and direct prefill decode
rejected

## Decision question

[Selective Mixed-V Down-Projection Screen](55-selective-mixed-v-down-projection.md)
showed a large functional gain for a reconstruction-selected mixed basis, but
left two acceptance questions open:

1. Is the selected block inventory stable across factorization seeds?
2. Can the exact 10-bit codebook plus 9-bit correction payload execute
   without an unacceptable runtime cost?

This experiment closes both questions for the pinned Gemma `down_proj`
workload.

## Seed protocol

The complete 26-block reconstruction screen was repeated with logical seeds
1 and 2. Seed 0 is the original screen. Every run uses:

- rank-970 free-word control;
- rank-1,344 mixed candidate;
- fully free U;
- 256 free V rows and 1,088 coded V rows;
- a 10-bit, 1,024-entry word codebook;
- two correction positions represented by one 9-bit unordered-pair ID;
- 800 ADMM outer iterations;
- identical corrected-CCE Fisher state and 0.6 shrinkage.

Each seed writes one durable JSON per block. The two new screens add 52
complete baseline/candidate comparisons.

## Reconstruction stability

All 52 repeated matrices improve weighted RMSE. Combined with seed 0, the
candidate is better in 78/78 block/seed comparisons.

| Seed | Mean block RMSE change | Aggregate RMSE change | Best block | Weakest block |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.672% | -0.694% | -1.311% | -0.232% |
| 1 | -0.677% | -0.691% | -1.283% | -0.235% |
| 2 | -0.674% | -0.696% | -1.293% | -0.217% |

Spearman rank correlations of per-block gain are:

| Seeds | Rank correlation |
| --- | ---: |
| 0 and 1 | 0.9938 |
| 0 and 2 | 0.9918 |
| 1 and 2 | 0.9918 |

The gain ranking is therefore highly repeatable.

### Stable selection boundary

The earlier 0.9% rule selected block 16 only under seed 0:

| Block | Seed 0 gain | Seed 1 gain | Seed 2 gain |
| ---: | ---: | ---: | ---: |
| 0 | 1.062% | 1.050% | 1.063% |
| 10 | 1.005% | 1.000% | 0.969% |
| 11 | 1.311% | 1.284% | 1.293% |
| 12 | 0.962% | 0.957% | 0.950% |
| 16 | 0.901% | 0.880% | 0.895% |
| 25 | 1.081% | 1.118% | 1.070% |

Tightening the primary-seed threshold to **0.95%** yields the same inventory
under every seed:

`0, 10, 11, 12, 25`

This five-block rule supersedes the exploratory six-block 0.9% rule.

## Three-seed functional confirmation

The five-block policy was evaluated on the disjoint WikiText windows 24
through 47 for all three factorization seeds:

| Seed | Free-word KL | Mixed KL | Change | Paired 95% delta interval | Sequence wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.393903 | 0.337782 | **-14.25%** | [-0.072385, -0.038983] | 22/24 |
| 1 | 0.388321 | 0.339039 | **-12.69%** | [-0.062580, -0.036932] | 24/24 |
| 2 | 0.382379 | 0.363872 | **-4.84%** | [-0.034698, -0.001501] | 15/24 |

Every seed passes independently. Pooling the 72 factorization-seed/sequence
pairs gives:

- free-word KL: 0.388201;
- mixed KL: 0.346898;
- relative change: **-10.64%**;
- paired interval: `[-0.051059, -0.031739]`;
- 61/72 wins.

The pooled result is descriptive because the three runs share token text,
but the independent per-seed intervals already establish the required
stability.

## Exact packed runtime prototype

`tools/benchmark_mixed_v_runtime.py` implements two analysis-only Triton
operations:

1. a direct mixed stage 1 that reads the exact 19-bit row-major payload,
   decodes the 10-bit codebook index and 9-bit correction-pair ID in-register,
   and reuses the production packed stage 2;
2. a one-time predecode kernel that expands the compact V payload into the
   current packed sign-word layout at model load.

The payload is not represented by convenient 16-bit fields. Records cross
32-bit boundaries exactly as the bit-cost model specifies. CPU round-trip
tests cover boundary packing, and both CUDA paths are bit-exact against an
independently materialized corrected-word matrix:

`maximum absolute output error = 0`

### Resident memory

The benchmark uses the current F32 runtime scales and row-padded packed
factors:

| Representation | Bytes per `down_proj` |
| --- | ---: |
| Rank-970 current packed control | 1,017,064 |
| Rank-1,344 compact mixed payload | 1,014,596 |
| Rank-1,344 predecoded packed execution | 1,392,384 |

The compact resident representation is 0.24% smaller than the control. The
predecoded execution representation adds 375,320 bytes per selected layer.
Across five blocks this is 1,876,600 bytes, or 1.79 MiB:

- 0.25% of the retained 764 MB steady runtime allocation;
- 0.14% of the retained 1.296 GB shell-load peak.

The 512-entry correction-pair lookup occupies 2,048 shared bytes in the
prototype and does not need per-layer storage.

## Runtime measurements

Hardware is the same RTX 4000 Ada workstation used by the retained runtime
profiles. Each of three benchmark runs uses 10 warm-ups and 50 synchronized
samples. The table reports the median of the three p50 measurements.

| Tokens | Rank-970 packed | Rank-1,344 predecoded | Direct compact decode |
| ---: | ---: | ---: | ---: |
| 1 | 89.10 us | 92.10 us | 140.95 us |
| 16 | 568.50 us | 792.15 us | 1,222.40 us |
| 128 | 2,853.15 us | 3,881.10 us | 7,703.35 us |

Relative affected-layer latency:

| Tokens | Predecoded / control | Direct compact / control |
| ---: | ---: | ---: |
| 1 | 1.034x | 1.597x |
| 16 | 1.401x | 2.150x |
| 128 | 1.354x | 2.714x |

If five of 26 `down_proj` calls use the candidate and the remaining 21 use
the control, the isolated `down_proj` family ratios are:

| Tokens | Predecoded hybrid | Direct compact hybrid |
| ---: | ---: | ---: |
| 1 | 1.006x | 1.115x |
| 16 | 1.077x | 1.221x |
| 128 | 1.068x | 1.330x |

One-time predecode takes **34.45 us per layer** at p50. All five selected
layers therefore decode in roughly 0.17 ms during load, outside inference
timing.

### Full-decode projection

In the retained grouped-MLP runtime profile, the five selected `down_proj`
calls total 0.4345 ms of a 22.6278 ms model CUDA p50, or 1.92%.

Applying the measured one-token ratios sample-by-sample projects:

| Policy | Projected model CUDA p50 | Change |
| --- | ---: | ---: |
| Rank-970 control | 22.6278 ms | baseline |
| Five predecoded mixed layers | 22.6416 ms | +0.06% |
| Five direct compact layers | 22.8721 ms | +1.08% |

This is a profile-backed projection, not a substituted end-to-end benchmark.
It is sufficient to choose the execution design; production promotion still
requires integration followed by the standard full runtime benchmark.

## Acceptance

The mixed free/coded V basis is **accepted** with this scope:

- `mlp.down_proj` matrices only;
- rank 1,344 with 256 free V rows;
- k=10 table and two 9-bit-encoded corrections;
- use only when the primary-seed weighted RMSE gain is at least 0.95%;
- on the pinned Gemma workload, blocks `0, 10, 11, 12, 25`;
- store the exact compact payload;
- predecode selected V factors once at load into current packed words.

The following variants are rejected:

- uniform application to all 26 blocks, because full-splice KL regressed;
- the 0.9% six-block inventory, because block 16 is not seed-stable;
- direct compact decoding as the default prefill path, because its measured
  128-token affected-layer cost is 2.71x the control.

Direct compact decode remains a viable future memory-preserving
specialization for single-token generation, where its projected model
overhead is only 1.08%. It is not needed for the accepted first
implementation.

## Production boundary

This accepts the research idea, not an unimplemented production claim. The
next implementation may define a packed schema and runtime preparation path,
but must preserve:

- exact 19-bit coded-word payload accounting;
- the free-row count and codebook identity;
- load-time predecode bit equality;
- per-layer fallback to ordinary rank-970 packed factors;
- complete quality, BPW, memory, and runtime benchmark gates.

Other projection shapes need their own equal-bit allocation and selection
evidence. Acceptance here must not be generalized to every U/V orientation
without measurement.
