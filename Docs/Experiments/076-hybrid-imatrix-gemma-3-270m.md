# Experiment 076: Unsloth columns with local Eaddario rows

Experiment 076 repeats Experiment 075 with a new hybrid activation IMatrix:

- **Columns:** `Datasets/gemma-3-270m-it-imatrix-unsloth/imatrix_unsloth.gguf_file`,
  from `unsloth/gemma-3-270m-it-GGUF` revision
  `c90975dbd40c0c7b275fefaae758c3415c906238`.
- **Rows:** `Datasets/gemma-3-270m-it-imatrix-eaddario-huge-rows/imatrix.gguf`,
  the exact activation-row dataset used by Experiment 075. These are the
  user-selected Eaddario rows, not either of the Fisher-row prototypes.

The generated dataset is
`Datasets/gemma-3-270m-it-imatrix-unsloth-columns-eaddario-rows/imatrix.gguf`.
Its SHA-256 is
`f4b1e6e371afbc34d3296e16559d4ad707be41075f88f4a5049fcfaaf8ef678d`
and size is 988,864 bytes. The source hashes are
`2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3`
for columns and
`3deceafea3aadd5fb6b75e22dfa0123bb99f22e0be95816354b374e80ebcc0c9`
for rows.

## Dataset construction and validation

`tools/build_gemma_270m_hybrid_imatrix.py` copies Unsloth's `.in_sum2` and
`.counts` tensors and the local dataset's `.out_sum2` and `.out_counts`
tensors. No axis is renormalized, rescaled, or recalibrated. The two axes keep
their original independent observation counts; the hybrid does not claim a
single shared calibration corpus or chunk count.

Validation checks exact equality of both the raw sums/counts and the normalized
importance vectors. The real Gemma 270M safetensors inventory accepts all 126
input and 126 output vectors across 18 blocks, with correct dimensions, no
unused tensors, and valid shared Q/K/V columns. Source receipts and the merged
validation report live beside the GGUF. Rebuilding audits the existing artifact
and refuses incompatible sources or changed bytes.

```powershell
.\.venv\Scripts\python.exe tools/build_gemma_270m_hybrid_imatrix.py
```

## Experiment settings

The launcher is
`experiments/076-fast-hybrid-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py`.
It inherits a hash-pinned snapshot of 075's input recipe from
`experiments/recipes/references/experiment075.json`. Only
`calibration.imatrix.path` and `calibration.imatrix.sha256` differ;
`use_rows=True` remains enabled. Experiment identity, baseline reference, and
numbered artifact paths advance to 076.

The pinned model, two full-rank stages, ADMM settings, tuning and optimizer
settings, global distillation, batch sizes, GPU activation cache, quality
protocol, profile options, maximum-rank control reuse, single-process execution,
and complete compression/export workflow all remain identical to 075.
This isolates the **column source** relative to 075, whose row values are retained.

Plan and dry-run checks passed. Plan mode explicitly does not validate dataset
contents; the separate builder validation above provides that check. No
compression or evaluation job was started, so no quality result is claimed.

```powershell
# Inspect the fully resolved plan or effective control recipe.
.\.venv\Scripts\python.exe experiments/076-fast-hybrid-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --plan-only
.\.venv\Scripts\python.exe experiments/076-fast-hybrid-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --dry-run

# Start or resume Experiment 076 when ready.
.\.venv\Scripts\python.exe experiments/076-fast-hybrid-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py
```

The durable control run is
`evidence/076/076-d2-uniform-control-gemma-3-270m-it`;
the completed deployment GGUF will be published at
`Results/076/gemma-3-270m-it-nanoquant.gguf` by the normal completion workflow.
Dataset provenance and preflight receipts are retained under `Results/076/setup/`.

Validation: 25 focused IMatrix/recipe tests passed; the full default suite passed
1,802 tests with 54 expected marker deselections. Repository Ruff and mypy passed.
