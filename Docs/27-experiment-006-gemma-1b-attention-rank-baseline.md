# Experiment 006: Gemma 3 1B attention-rank baseline

Experiment 006 is the first complete compression run to inherit the promoted attention allocation policy:

- `self_attn.v_proj`: physical maximum rank;
- `self_attn.k_proj`: physical maximum rank;
- `self_attn.q_proj`: 1.25x its ordinary packed-factor budget, rounded down to the greatest supported aligned rank.

The experiment compresses pinned `google/gemma-3-1b-it` through the complete tuning and global-distillation
pipeline, exports the NanoQuant GGUF, and compares it with the BF16 source on the matched WikiText-2 protocol and
the six common multiple-choice tasks. Calibration uses the pinned 256-sample, 2048-token mixture with equal
UltraChat and WikiText representation.

The run writes durable state and evidence under `evidence/m13`, deployment outputs under
`outputs/006-gemma-3-1b-it`, and publishable GGUF/statistics junctions under `Results/006`. It fails if WDDM shared
GPU memory exceeds 0.75 GiB. The zero-argument launcher is:

```powershell
.\.venv\Scripts\python.exe experiments\006-compress-and-benchmark-gemma-3-1b-it.py
```

This is a new benchmark baseline, not a replay of Experiment 001. Its result should be used as the parent comparison
for subsequent Gemma 3 1B allocation or tuning experiments.
