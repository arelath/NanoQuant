# Experiment 049: Cyclic Scale Rank

## Question

Can the single `scale_pre` and `scale_post` vectors be replaced by several
implicit banks, cycling binary factor components through those banks, and use
the added scale freedom to reduce the stored factor rank at lower total cost?

This is a local format-capacity screen. It does not change or resume Experiment
048, which is paused at the validated handoff in
[`Docs/83-experiment-048-paused-handoff.md`](../83-experiment-048-paused-handoff.md).

## Refined representation

For scale rank `K`, factor component `r` belongs to `g = r mod K`:

```text
W_hat = sum_g diag(post[:, g]) U[:, G_g]
                    diag(mid[G_g]) V[G_g, :]
                    diag(pre[g, :])
G_g = {r : r mod K = g}
```

The component order is arbitrary, so a future fitter could permute components
offline and still use the same modulo rule without storing group indices. A
runtime could likewise pack the groups contiguously after that permutation.

`scale_mid` is already one value per binary component. Giving it `K` banks while
the contraction still pairs component `r` only with component `r` adds unused or
duplicate parameters. A genuinely rank-increased middle scale would mix
different left and right components, creating an `R x R` interaction and a
different, much more expensive format. Experiment 049 therefore increases only
the pre/post bank count while retaining the fully granular existing mid scale.

At BF16 scale storage, the cost is

```text
bits(K, R) = bits(K=1, R) + 16 (K - 1) (out + in).
```

Because one binary factor component costs approximately `out + in` bits, every
extra BF16 pre/post bank costs about 16 factor components. No group indices are
charged because assignment is implicit.

## Prior evidence

This idea substantially overlaps the rank-group scale probe recorded in
`ImprovementSuggestions/ReconstructionHeadroom.md` section 9.2. That probe used
block-12 q/down matrices and found only 0.1–0.7% fixed-rank error recovery; at
equal bits, ordinary factor rank improved error by 2–12% instead. The new screen
was still run because it uses the rewrite's pinned model, full Fisher profiles,
current ADMM implementation, modulo assignment, and a 1.2-BPW operating point.

## Bound protocol

- Model: pinned `google/gemma-3-1b-it` revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Calibration: `evidence/m4/gemma-full-fisher-state`, shrinkage 0.6.
- Blocks: 1, 12, and 24.
- Projection shapes: q, gate, and down.
- Target: 1.2 BPW; factor rank aligned to 32.
- ADMM: 800 outer iterations, 5 inner iterations, cubic penalty schedule,
  regularization 0.03, production deterministic logical seeds.
- Scale fit: eight ALS passes. A two-pass precursor produced the same conclusion.
- Scale ranks: `K in {1, 2, 3, 5}`.
- Metrics:
  1. fixed-factor-rank Fisher-weighted squared-error gain, where extra scale bits
     are allowed;
  2. exact-baseline-bit comparison, where factor rank is reduced to pay for all
     extra BF16 scale values.

Evidence:

- `evidence/049/cyclic-scale-rank-block1-q-gate-down-als8.json`
- `evidence/049/cyclic-scale-rank-blocks12-24-q-gate-down-als8.json`

The screen tool is `tools/probe_cyclic_scale_rank.py`. Its protocol hash binds
the model revision, calibration location, selected blocks/projections, numerical
recipe, scale ranks, bit price, and device.

## Results

The result is consistent across all nine matrices:

| K | Fixed-rank gain, range | Fixed-rank gain, mean | Equal-bit regression, range | Equal-bit regression, mean |
|---:|---:|---:|---:|---:|
| 2 | 0.09–0.20% | 0.12% | 4.16–7.21% | 5.14% |
| 3 | 0.17–0.39% | 0.24% | 4.07–7.00% | 5.02% |
| 5 | 0.34–0.75% | 0.47% | 8.34–14.76% | 10.35% |

The modulo banks are therefore real additional capacity, but their marginal
value is much lower than factor components. Eight ALS passes do not materially
improve the two-pass result, so the loss is not explained by stopping the scale
fit too early.

For an intentionally optimistic bit-efficiency bound, the fixed-rank gains were
converted to equivalent factor components using each matrix's measured local
rank slope:

| K | Best equivalent components across nine matrices | BF16 added-scale cost | Impossible ideal 1-bit cost |
|---:|---:|---:|---:|
| 2 | 0.89 | 16 | 1 |
| 3 | 1.72 | 32 | 2 |
| 5 | 3.17 | 64 | 4 |

Even before quantization loss, metadata, or the production 32-component rank
alignment, the BF16-fit gain is smaller than the storage cost obtained by
pretending every added scale value costs only one bit. This local-slope
conversion is an inference rather than an exact unaligned-rank experiment, but
the actual aligned equal-bit arms provide the stronger practical result: all 27
candidate comparisons regress substantially.

## Decision

Reject cyclic scale rank as a compression or bit-saving mechanism at this
operating point. It fails the local equal-bit gate on every tested projection
and depth, independently confirming the earlier rank-group result. It should
not proceed to splice KL, a compressed-model quality benchmark, serialization,
or runtime implementation.

There is one narrow non-compression use: low-bit scale banks could fill otherwise
unusable per-layer alignment slack and buy roughly 0.1–0.7% local error reduction
without lowering aligned factor rank. That is a tiny quality-only add-on, not a
bit saving, and it would add scale decoding plus multiple partial accumulations.
It is lower priority than mechanisms that buy additional binary components or
change the functional objective. A future revisit needs a genuinely new source
of gain—such as learned component grouping that clears an equal-bit gate—not
only more scale-fit iterations or lower scale precision.
