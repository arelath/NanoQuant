# Experiment 007: Gemma 3 270M quality benchmark

Experiment 007 is the `unsloth/gemma-3-270m-it` counterpart to Experiment 006. It pins revision
`23cf460f6bb16954176b3ddcc8d4f250501458a9` and inherits the same numerical recipe:

- equal UltraChat and WikiText calibration representation;
- physical maximum rank for `self_attn.v_proj` and `self_attn.k_proj`;
- 1.25x packed-factor budget for `self_attn.q_proj`;
- complete per-block tuning and global distillation;
- matched WikiText-2 and six-task BF16-versus-NanoQuant quality evaluation.

The model configuration contains 18 transformer blocks. The run writes durable state and evidence under
`evidence/m14`, deployment outputs under `outputs/007-gemma-3-270m-it`, and publishable GGUF/statistics junctions
under `Results/007`. It retains Experiment 006's 0.75 GiB WDDM shared-memory limit.

The zero-argument launcher is:

```powershell
.\.venv\Scripts\python.exe experiments\007-compress-and-benchmark-gemma-3-270m-it.py
```

Do not launch it concurrently with another resident CUDA compression run.
