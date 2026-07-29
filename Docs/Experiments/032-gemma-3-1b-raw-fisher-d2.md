# Experiment 032: Gemma 3 1B raw-Fisher D2 compression

## Status

**Prepared; complete run pending.**

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

Pending.
