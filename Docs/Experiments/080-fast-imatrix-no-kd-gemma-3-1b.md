# Experiment 080: Gemma 3 1B IT, imatrix, and no global KD

Prepared on 2026-11-15. Compression and quality evaluation have not started.

080 transfers Experiment 074's recipe to `google/gemma-3-1b-it` at revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`. It keeps both full-rank stages,
ADMM 600 × 2, compilation, shared QKV, per-layer tuning and block refitting.
Fused AdamW starts at block 0 on the same projections as 074. The tuning
batch is measured from block 0 up to 8, the block-forward batch is 8, and
the activation GPU cache is automatic. Global KL distillation is disabled.
The launcher uses its own run directory and no imported block continuation.

The 1B input imatrix is the Unsloth file pinned in 073, at
`Datasets/gemma-3-1b-it-imatrix-unsloth/imatrix_unsloth.dat`. The original 073
fitting tokens were validated and materialized under 080's own run identity
without regenerating them. Shape is 256 × 2,048 and the fingerprint is
`sha256:09b718228b3692d5fee551dfb664a2598b95075d66a7124a03a9096c6ca6fed2`.

## Quality and error attribution

- The standard dense-reference quality report compares BF16 and NanoQuant on
  WikiText-2 and the six existing tasks at the 074 limits. Every 080 evaluation
  also runs standard MMLU on all 14,042 test questions, with five dev examples
  from each question's subject. It scores the highest next-token logit among
  A through D. MMLU does not use the six-task 1,000-example cap. The report
  includes overall accuracy, and the JSON includes per-subject counts and
  per-question predictions. Other historical launchers keep their recorded
  benchmark choices. New dense quality recipes can enable `include_mmlu=True`.
- After GGUF validation, the shared attribution step reads the exported
  NanoQuant projection weights and the matching 1B source matrices. It weights
  squared residuals by mean-one column importance and maps that error through
  the 1B sensitivity curves. It writes estimates and shares for every block,
  projection type and projection to `Results/080/080-imatrix-error-attribution.md`
  and its JSON companion. Shares are relative to the sum of the fitted
  isolated-effect estimates. They do not measure joint model KL. The report
  flags errors outside the fitted sweep range.
- The pinned FP16 sensitivity sweep uses 16 × 256 tokens and 2,184 completed
  trials across all 182 projections. The sweep and fits are under
  `Datasets/gemma-3-1b-it-fp16-sensitivity-16x256/measurements/022fc4f64542b546eebd3bbbe21ee50cf36d36946dcd0a871d20ae5e24abfc6f`.

## Setup verification

Run the following without starting the model. `--plan-only` prints the
resolved plan, and `--dry-run` also checks the local MMLU inputs, projection
coverage, imatrix and sensitivity hashes, and the pinned checkpoint.

```powershell
.\.venv\Scripts\python.exe experiments/080-fast-imatrix-no-kd-compress-and-benchmark-gemma-3-1b-it.py --dry-run
```

The checked plans and input receipts are in `outputs/080`. When ready to
launch, check that `cuda:0` is free, use the local evaluation cache at
`outputs/imatrix-calibration-comparison-v1/hf-datasets` via `HF_DATASETS_CACHE`,
and remove `--dry-run`. The measured tuning batch and 1B GPU fit are runtime
checks. The original 0.75 GiB WDDM shared-memory limit still applies.
