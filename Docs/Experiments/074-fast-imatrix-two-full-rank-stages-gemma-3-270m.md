# Experiment 074 Gemma 3 270M IT with imatrix importance and speed settings

Prepared on 2026-11-14. **Setup only; not started.** Experiment 074 compresses
Gemma 3 270M IT with Experiment 073's recipe and Unsloth's 270M llama.cpp imatrix.
It turns on every speed setting the pipeline currently supports.

## Recipe

The frozen reference is `experiments/recipes/references/experiment073.json`, with
semantic hash `sha256:c289aeaf4072b773ffa7f814a97a6958070f6407d80c02667c51701f6f2bc468`.
Compared with it, 074 changes the model, the imatrix and the settings below. ADMM,
full-rank stages, tuning epochs and learning rates, allocation and global
distillation are unchanged.

`unsloth/gemma-3-270m-it-GGUF` contains GGUF files only, and the pipeline
compresses safetensors. Compression therefore reads the matching
`unsloth/gemma-3-270m-it` snapshot, the repository's existing 270M template
(revision `23cf460f6bb16954176b3ddcc8d4f250501458a9`). The imatrix comes from the
GGUF repository.

| Setting | Experiment 073 | Experiment 074 |
| --- | --- | --- |
| Model | `google/gemma-3-1b-it`, 26 blocks | `unsloth/gemma-3-270m-it`, 18 blocks |
| Imatrix | Unsloth 1B, legacy format | Unsloth 270M, GGUF format |
| Tuning batch | 072's 4B sizes, then a measured batch up to 8 from block 11 | Measured batch up to 8 from block 0 |
| Fused FP32-scale AdamW (gate, down, attention output) | From block 19 | From block 0 |
| Block-forward batch | 4 | 8 |
| GPU activation cache | Off | Auto |
| Process | One block per slice | One uninterrupted process |

## Speed settings and their effect on results

- **Measured tuning batch.** `block_tuning.batch_schedule` measures a block
  forward/backward pass at run start. It selects the largest batch up to 8 that
  fits with a 1.5× safety margin, and runs it without gradient accumulation.
  Experiment 071 used batch 8 on 1B. This changes optimization relative to 073's
  early blocks. The selection is saved in
  `state/tuning-batch-selection.json` and reused on resume.
- **Fused AdamW from block 0.** Experiment 072 enabled it from block 19 after a
  held-out quality gate passed for these three projections. Up and shared QKV
  stay on parity, as in 072. Its effect at the first blocks and at batch 8 has
  not been validated.
- **Block-forward batch 8 and GPU activation cache.** These change execution only.
  The cache keeps activations on the GPU when they fit and falls back otherwise.
  It is not part of the commit identity.
- **No slicing.** 073 restarted the process after every block. 074 runs all 18
  blocks, export and evaluation in one process. It still commits every layer and
  resumes from the journal after an interruption.

Compiled block forward stays enabled. The WDDM shared-memory guard (0.75 GiB)
is kept as a safety limit, not a speed setting.

## Importance data

[The user-selected imatrix](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/blob/main/imatrix_unsloth.gguf_file)
was downloaded from repository revision `c90975dbd40c0c7b275fefaae758c3415c906238`.

- Local file: `Datasets/gemma-3-270m-it-imatrix-unsloth/imatrix_unsloth.gguf_file`
- Format: llama.cpp GGUF imatrix
- Bytes: 471,040
- SHA-256: `2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3`
- Source receipt: `Datasets/gemma-3-270m-it-imatrix-unsloth/source.json`

Resolving it against the pinned model's safetensors inventory mapped all 126
projection vectors (18 blocks × 7 projections). This check read model metadata
and importance vectors only.

## Execution

As in 073, `reuse_maximum_rank_control=True` performs one compression pass and
publishes it as the numbered result. No KL teacher dataset is read in this mode,
so 074 does not need a 270M teacher dataset.

Inspect without starting any stage:

```powershell
.\.venv\Scripts\python.exe experiments\074-fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --plan-only
.\.venv\Scripts\python.exe experiments\074-fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --dry-run
```

Start once the GPU is free; the same command resumes after an interruption:

```powershell
.\.venv\Scripts\python.exe experiments\074-fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py
```

The durable compression directory is
`evidence/074/074-d2-uniform-control-gemma-3-270m-it`; numbered publication uses
`Results/074/gemma-3-270m-it-nanoquant.gguf`. Both inspection commands passed.
No 074 calibration, fitting, tuning, export or quality benchmark has run.
