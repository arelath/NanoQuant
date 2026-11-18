# Experiment 085 — Curated KL rank allocation for Gemma 3 1B IT

Repeat 083 using the completed 160-window curated FP16 sensitivity dataset.
Test whether the new sensitivity curves and isolated KL anchors yield lower
joint decoder KL and better benchmark quality at the same factor budget.
083 is the primary comparison. 080 and 082 provide the uniform two-stage
and reconstruction-error allocation references.

The decoder factor budget is **219,368,448 bytes**, or 2.51511 bits per
decoder projection weight, including signs and scales. Use 083's pinned
checkpoint, Unsloth IMatrix, shared QKV, and 256 × 2,048 compression tokens.
Keep ADMM, layer and block tuning, refit, and Q8_0 embeddings matched. Global
distillation and rank retries are disabled. The staged rank-response dataset
was produced from 080, whose compression tokens have the same fingerprint
as those used by 082 and 083.

## Allocation and evaluation

1. Reuse the 130 staged rank-response curves and the 082 reference plan.
   Use the new sensitivity fits from 3,276 trials across 182 projections,
   with six noise levels and three seeds. The 160 × 256 windows give
   40,800 scored positions per trial.
2. Measure each isolated projection substitution from the 082 export on
   the curated teacher using batch-eight scoring. Keep the teacher
   embeddings fixed. Anchor each projection to this measured KL and use
   the fitted sensitivity exponent to price changes in its IMatrix error.
3. Publish the per-rank KL options. Rank options are aligned to 32 and
   capped at three full stages. Every member's predicted error must stay
   within its measured 1–35% sensitivity range. The production allocator
   minimizes the sum of estimated isolated KL under the byte budget.
4. Compress, export, and benchmark 085. Verify byte-identical shared Q8_0
   embeddings across 080, 082, 083, and 085. Score joint decoder KL and
   full-model KL on the same curated teacher. Compare dense WikiText-2,
   the six tasks at the retained 1,000-example protocol, and full five-shot
   MMLU. Report predicted isolated KL savings and measured joint KL separately.

## Launch and resume

The isolated scoring step checkpoints after each projection. The local
run sequence waits for the measured anchors before publishing the KL
options and launching compression. The 085 launcher validates the
allocation dataset and records its content key in the run config.

```powershell
$python = '.\.venv\Scripts\python.exe'
$snapshot = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'
$sweep = 'Datasets/gemma-3-1b-it-fp16-sensitivity-curated-160x256/measurements/8c98f7faa2a08fafcd818847d87a5c3237dba139043607180d39f35f1be2ea5c/results.json'
$imatrix = 'Datasets/gemma-3-1b-it-imatrix-unsloth/imatrix_unsloth.dat'
$control = 'Results/082/gemma-3-1b-it-nanoquant.gguf'
$attribution = 'outputs/085/082-curated-attribution.json'
$isolated = 'outputs/085/082-curated-isolated.json'
$rankData = 'Datasets/gemma-3-1b-it-staged-rank-response-080-v1'
$options = 'Datasets/gemma-3-1b-it-curated-kl-rank-options-085'
$env:HF_DATASETS_CACHE = (Resolve-Path 'outputs/imatrix-calibration-comparison-v1/hf-datasets').Path
& $python tools/predict_imatrix_weighted_kl.py $sweep --predict-only --snapshot $snapshot --imatrix $imatrix --gguf $control --output outputs/085/082-curated-attribution.md
& $python tools/validate_imatrix_attribution.py $attribution --gguf $control --output outputs/085/082-curated-isolated.md
& $python tools/rerank_staged_kl_budget.py outputs/082/preflight.json $attribution $isolated --dataset $rankData --output outputs/085/085-curated-kl-rerank.md --publish-kl-options $options
& $python experiments/085-curated-kl-allocation-compress-and-benchmark-gemma-3-1b-it.py
& $python tools/compare_staged_rank_runs.py $attribution 80 82 83 85 --output Results/085/085-quality-and-joint-kl-comparison.md
```
