# Experiment 032: Gemma 3 1B raw-Fisher D2 compression

## Status

**In progress.** The raw-Fisher uniform control and its exact-unit KL profile
completed on 2026-07-29. The KL-allocated candidate is running; complete
compression, export, and quality results remain pending.

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

### Pending

The candidate is currently running its resumable 130-unit measured rank-probe
stage. The final verdict remains pending candidate block completion, strict
artifact validation, export, effective BPW measurement, and the retained
WikiText quality comparison.
