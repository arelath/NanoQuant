# Experiment 075 Gemma 3 270M IT with row and column imatrix weights

Experiment 075 uses [Experiment 074's settings](074-fast-imatrix-two-full-rank-stages-gemma-3-270m.md)
with a locally generated row and column imatrix. The model remains
`unsloth/gemma-3-270m-it` at revision `23cf460f6bb16954176b3ddcc8d4f250501458a9`.
The recipe keeps two full-rank stages, the measured tuning batch up to 8 from
block 0, fused AdamW from block 0, block-forward batch 8, automatic GPU activation
caching, and the single-process compression, export, and quality workflow.

The launcher pins `Datasets/gemma-3-270m-it-imatrix-eaddario-huge-rows/imatrix.gguf`
with `use_rows=True`. It was collected from the same pinned model used for
compression, using all complete 512-token chunks from
`eaddario/imatrix-calibration/combined_all_huge.parquet`. Local `source.json` and
`validation.json` record the model and corpus revisions, hashes, vector coverage,
and observation counts.

Validation matched all 126 projections across 18 blocks, covering 124,416 output
rows. Each projection had 6,810,112 observations across 13,301 chunks. The
imatrix SHA-256 is `3deceafea3aadd5fb6b75e22dfa0123bb99f22e0be95816354b374e80ebcc0c9`.

The comparison changes the calibration corpus for the imported imatrix as well
as enabling output rows. To isolate the effect of row weights, a control would
use the same local imatrix with `use_rows=False`.

Inspect the recipe or run its preflight from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\075-fast-row-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --plan-only
.\.venv\Scripts\python.exe experiments\075-fast-row-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --dry-run
```

Start or resume compression with the same command without a preview flag.
The durable run lives at `evidence/075/075-d2-uniform-control-gemma-3-270m-it`
and publishes `Results/075/gemma-3-270m-it-nanoquant.gguf` after successful export.
The experiment is prepared for launch.
