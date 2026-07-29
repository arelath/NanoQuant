# Experiment 032: Gemma 3 1B raw-Fisher D2 compression

## Status

**In progress.** The raw-Fisher uniform control, exact-unit KL profile, and
candidate measured rank-response profile completed on 2026-07-29. The
KL-allocated candidate is compressing blocks; complete compression, export,
and quality results remain pending.

- Model: `google/gemma-3-1b-it`
- Launcher:
  `experiments/032-raw-fisher-d2-compress-and-benchmark-gemma-3-1b-it.py`
- Baseline: Experiment 022

## Question

Does the raw diagonal Fisher advantage measured in
[Document 42](../42-fisher-importance-shrinkage-probe.md) survive the complete
compression lifecycle?

## Controlled change

Experiment 032 is structurally identical to Experiment 022 except for
`calibration.shrinkage`:

| Setting | Experiment 022 | Experiment 032 |
| --- | ---: | ---: |
| Fisher shrinkage toward the vector mean | 0.6 | **0.0** |

The candidate retains Experiment 022's fused QKV representation, automatic
same-run exact-unit KL profile, calibration-weighted measured rank responses,
D2 allocation, per-layer and factorized tuning, post-block refit, global
distillation, packing, GGUF export, and quality protocol.

This is intentionally a stricter gate than the static probe. Allocation,
outlier selection, tuning, and distillation can interact with the weighting
change, so the 13.13% static KL gain is not assumed to transfer.

## Success criteria

- all 26 blocks and the complete artifact graph validate;
- logical, packed, checkpoint, GGUF, and export-summary outputs complete;
- the retained 64-by-128 WikiText protocol matches Experiment 022's token
  identity;
- effective BPW does not exceed Experiment 022;
- candidate perplexity improves on Experiment 022's 228.550618 result;
- task accuracy, memory, runtime, ranks, and allocation changes are reported
  even if the primary perplexity gate fails.

## Result

### Completed control and profile

The raw-Fisher uniform control completed all 26 blocks:

- 156 durable journal records, including 26 block commits;
- resident compression time: 2:10:09;
- final block entry loss: 2380.173584;
- final block committed loss: 1939.293457;
- control run artifact:
  `sha256-e6dafe60222fb09d123b35f2af4e2ba503f5af6d7194988015e83f9d33a0c460`.

The resumable exact-unit KL profile then completed all 162 arms. Its profile
key is
`sha256:ab870f3438dd7f4023bb17523fa1f6140504127041cfce58b49e3996877e2a92`
and its artifact is
`sha256-3ec0557df912a4af652d4ea6e290c3b1f683edd4b06d128c0d1cd674a8f56ee9`.

The exact-unit measurements also provide preliminary evidence of exploitable
cross-block structure:

| Projection type | Mean unit KL | Coefficient of variation | Adjacent-block Pearson |
| --- | ---: | ---: | ---: |
| `mlp.down_proj` | 0.071911 | 0.350 | 0.625 |
| `mlp.gate_proj` | 0.040913 | 0.616 | 0.840 |
| `mlp.up_proj` | 0.040620 | 0.446 | 0.841 |
| `self_attn.attn_qkv` | 0.024226 | 0.570 | 0.434 |
| `self_attn.o_proj` | 0.026989 | 0.560 | 0.448 |

Across blocks, the exact-unit KL vectors for `mlp.gate_proj` and
`mlp.up_proj` have Pearson correlation 0.943. `mlp.down_proj` has the largest
mean sensitivity, while the smallest exact-unit arms are concentrated in
late-block attention. This supports block-aware allocation and confirms that
nearby blocks are not independent, but it does not justify tying their
weights or ranks: the coefficients of variation remain substantial and the
complete D2 allocator must still prove that using the individual measurements
improves end quality.

### Completed candidate rank-response profile and plan

The candidate completed all 130 resumable rank probes in 1430.58 seconds. Each
unit measured the calibration-weighted objective at its baseline rank and at
aligned lower and upper ranks, rather than borrowing a type-wide decay curve.
The resulting profile artifact is
`sha256-2a97ce9bba5bf86e76b9b59cc8eca615152464d99323be36749b95b62300dfd2`.

The finalized candidate plan is
`sha256-71d1bd677956bbdc78691a62d884339273948f814c093b58d7505b6134903ad9`.
It contains 111,552 total unit ranks, 390 outlier columns, and 714,812,818
planned bits. Against Experiment 022 at the same target, raw Fisher changes 36
of 130 unit ranks while retaining the same outlier count:

| Projection type | Changed units | Rank delta vs 022 | Planned-bit delta |
| --- | ---: | ---: | ---: |
| `mlp.down_proj` | 7/26 | +64 | +517,120 |
| `mlp.gate_proj` | 11/26 | +704 | +5,688,320 |
| `mlp.up_proj` | 12/26 | -672 | -5,429,760 |
| `self_attn.attn_qkv` | 3/26 | -224 | -605,696 |
| `self_attn.o_proj` | 3/26 | -96 | -210,432 |
| **Total** | **36/130** | **-224** | **-40,448** |

The largest individual move is block 9 `mlp.up_proj`, from rank 1088 in
Experiment 022 to 864 here. The candidate therefore satisfies the planned-cost
side of the no-higher-BPW gate, although final effective BPW still requires
completed artifacts and export.

The completed measurements show that the allocation signals are related but
not interchangeable:

| Signal pair over 130 units | Spearman correlation |
| --- | ---: |
| Exact-unit KL vs final rank | 0.815 |
| Exact-unit KL vs baseline weighted normalized squared error | 0.260 |
| Exact-unit KL vs lower-rank response slope | -0.344 |
| Exact-unit KL vs upper-rank response slope | -0.325 |

Raw-Fisher and Experiment 022 exact-unit KL values preserve nearly the same
ordering (Spearman 0.966; median raw/022 ratio 0.993). The changed allocation
therefore does not come from a wholesale change in functional sensitivity
order. It comes from combining those anchors with the new raw-Fisher
reconstruction errors and measured response slopes. This supports retaining
all three signals rather than replacing the measured allocator with a single
per-type or per-block heuristic.

### Same-rank prefix exposes the objective tradeoff

Blocks 0 through 6 use exactly the same 35 unit ranks in Experiment 032 and
Experiment 022, and neither run spent retry ranks there. This prefix therefore
provides a controlled comparison of the reconstruction selected by each
Fisher objective before allocation differences begin at block 7.

Raw Fisher increases objective-independent, unweighted weight-reconstruction
error for every one of the 35 same-rank units:

| Projection type | Same-rank units | Raw-Fisher unweighted-error delta |
| --- | ---: | ---: |
| `mlp.down_proj` | 7 | +20.54% |
| `mlp.gate_proj` | 7 | +17.83% |
| `mlp.up_proj` | 7 | +15.21% |
| `self_attn.attn_qkv` | 7 | +36.64% |
| `self_attn.o_proj` | 7 | +64.18% |
| **All units** | **35** | **+30.88%** |

The overall median increase is 21.83%, with a range from +11.49% to +92.08%.
This is not a raw-Fisher failure: the paired held-out probe in Document 42
already found 13.13% lower full-model KL for raw Fisher at identical BPW.
Instead, the same-rank result demonstrates that raw Fisher deliberately
sacrifices considerably more low-importance Frobenius mass, especially in
`o_proj`, to preserve the directions the functional objective values. Plain
Frobenius error would reject every raw-Fisher unit and select the worse
held-out model, so it is falsified as a promotion or allocation metric for
this recipe.

Within Experiment 032's raw-Fisher objective, the first seven D2 block
boundaries have 27.16% to 33.46% lower normalized error than its uniform
control (mean 29.96% lower). This is encouraging evidence that the measured
allocation spends its early-block budget effectively, but it is not a
substitute for the retained WikiText comparison: the control and candidate
have different per-block ranks and their resident boundary metrics are not
the final model-level evaluator.

### Redistributed blocks pass the resident gates

The first three blocks with rank changes relative to Experiment 022 also
completed without retries:

| Block | Raw-Fisher rank changes vs 022 | Normalized boundary delta vs raw-Fisher uniform control |
| ---: | --- | ---: |
| 7 | `gate_proj` +64 | -32.09% |
| 8 | `gate_proj` +64; `up_proj` -96 | -24.77% |
| 9 | `gate_proj` +64; `up_proj` -224 | -23.73% |

Block 9 is the largest individual redistribution in the plan. Its rank-864
`up_proj` passed on the first attempt with raw error 0.41688 and weighted error
0.13310, both below the 0.5 acceptance thresholds. The measured allocator can
therefore remove 224 ranks from that unit without triggering the safety retry
that would erase the planned bit saving.

Experiment 032's block-9 normalized boundary error is also 9.82% below
Experiment 022's recorded boundary, but this is diagnostic only. The runs
propagate their own compressed activations and use their own calibration
receipts, so only the retained, token-identity-checked model-level evaluator
can establish a cross-run quality win.

### Halfway trajectory is positive against control but mixed against 022

At 13 of 26 durable candidate blocks, every normalized boundary is below the
corresponding raw-Fisher uniform-control boundary. The mean improvement is
26.40%, the median is 27.24%, and the range is 15.68% to 33.46%. No layer has
spent retry bits, so these gains have not silently exceeded the planned
budget.

The diagnostic comparison with Experiment 022 is less uniform. Experiment
032 is lower at 11 of the first 13 recorded boundaries and has a mean delta of
-13.02%, but block 10 is 19.82% higher and block 12 is 45.36% higher. Those
middle-depth regressions prevent treating the early trajectory as proof of a
complete quality win. As above, this is not a protocol-matched comparison:
each run has its own calibration receipt and propagated compressed
activations. The retained final WikiText evaluator must decide whether the
regressions are objective-scale artifacts, local errors repaired by later
blocks and distillation, or a real raw-Fisher quality cost.

### Pending

The candidate is currently compressing its 26 blocks. The final verdict
remains pending candidate block completion, strict artifact validation,
export, effective BPW measurement, and the retained WikiText quality
comparison.
