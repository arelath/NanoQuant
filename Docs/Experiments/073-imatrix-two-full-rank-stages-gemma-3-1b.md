# Experiment 073 Gemma 3 1B IT with imatrix importance

Prepared on 2026-11-13. **Setup only; not started.** Experiment 073 compresses
`google/gemma-3-1b-it` with Experiment 072's numerical settings and the supplied
Unsloth llama.cpp imatrix. Experiment 072 remains independent and can continue.

## Recipe

The frozen reference is `experiments/recipes/references/experiment072.json`, with
semantic hash `sha256:e9a730f78beaeb8f0b1b2813b331ad36ebb0bf6a67a274acca0fa32fcc629dc5`.
Compared with this reference, the numerical recipe changes only the model identity
and adds imatrix input importance. It also removes the journal digest that tied
072's optimizer transition to its already completed blocks. Experiment identity,
output paths and the same-model quality reference are specific to 073.

| Setting | Experiment 073 |
| --- | --- |
| Source | `google/gemma-3-1b-it` |
| Model and tokenizer revision | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Decoder blocks | 26 |
| Factors | Two independent full-rank stages, including shared QKV owners |
| Rank budget | Same nominal 2 BPW target; full-rank promotion permits exceeding it |
| ADMM | 600 outer iterations, 2 inner iterations |
| Compiled block forward | Enabled, as in 072 |
| Nonfactorized batch / factorized batch / refit batch | 4 / 1 / 1 |
| Microbatch / block-forward batch | 1 / 4 |
| Factorized tuning / post-block refit | 8 / 2 epochs |
| Optimizer schedule | Parity through block 18; fused FP32-scale AdamW for gate, down and attention output from block 19 |
| Other projection optimizers | Parity, including up and shared QKV |
| Global distillation | Same 8-epoch top-64 protocol as 072 |
| Benchmark backend and task limit | Dense, 1000, as in 072 |
| Process recovery | One new durable block per slice; restore from the saved activation boundary |

The block-19 optimizer switch and 4B-sized batches are deliberately retained to
match 072, rather than substituting 071's original 1B batches. Only the old
continuation journal binding is removed. The recipe records Experiment 071 as the
same-model quality reference; differences from 071 extend beyond imatrix because
073 inherits 072's later optimizer and batch settings.

## Importance data

[The user-selected imatrix](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/blob/main/imatrix_unsloth.dat)
was downloaded from repository revision `f0b45be0aac41bd6a100a4b5734cad5f67255bfb`.

- Local file: `Datasets/gemma-3-1b-it-imatrix-unsloth/imatrix_unsloth.dat`
- Format: legacy llama.cpp binary
- Bytes: 1,430,421
- SHA-256: `86a81192de49a6480f8dc19c5a7fd3f4377af59172fd4c7b8d93d296a0a18515`
- Source receipt: `Datasets/gemma-3-1b-it-imatrix-unsloth/source.json`

Validation against the pinned model's safetensors inventory mapped all 182
projection vectors across 26 blocks, with no missing, ambiguous or unused entries.
Widths and shared-QKV input importance agree. This check read model metadata and
importance vectors; it did not load model weights or execute CUDA work.
The receipt is `evidence/073/imatrix-validation.json`.

The file replaces input-channel importance for fitting; output/Fisher importance
still comes from the inherited calibration method. It weights ADMM and subsequent
weighted fitting/selection consistently. The file is pinned in the recipe and a
hash mismatch prevents starting or resuming under different importance data.

## Preparation and execution

As in current 072, `reuse_maximum_rank_control=True` performs one compression pass
and publishes it as the numbered result. Full-rank promotion fixes all ranks, so
KL allocation profiling, rank-response profiling and a duplicate candidate fit
are skipped. The optional teacher path is the existing
`Datasets/gemma-3-1b-it-kl-48x512`; this path is not loaded in this execution mode.
No 4B teacher or allocation dataset is reused, and no new dataset generation is
needed. Global distillation remains enabled and prepares its own targets as before.

Inspect without starting any stage:

```powershell
.\.venv\Scripts\python.exe tools/run_experiment073_campaign.py --plan-only
.\.venv\Scripts\python.exe tools/run_experiment073_campaign.py --dry-run
```

Start later, once the GPU is available:

```powershell
.\.venv\Scripts\python.exe tools/run_experiment073_campaign.py
```

The same command resumes after interruption. The campaign allows 64 slices for
26 blocks plus final work. Its durable compression directory is
`evidence/073/073-d2-uniform-control-gemma-3-1b-it`; numbered publication uses
`Results/073/gemma-3-1b-it-nanoquant.gguf` and the 073 quality report paths.

Both inspection commands passed. Their resolved output is saved under
`evidence/073/plan-only.json` and `evidence/073/dry-run.json`. These modes inspect
configuration; the separate imatrix validation above establishes actual input
coverage. No 073 calibration, fitting, tuning, export or quality benchmark was run.

Validation passed 20 experiment/architecture checks and 18 broader recipe checks,
plus Ruff and mypy. The underlying imatrix feature passed the full 1,673-test suite
before this experiment was added.
