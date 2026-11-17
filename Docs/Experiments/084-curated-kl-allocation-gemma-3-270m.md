# Experiment 084 — Curated KL rank allocation for Gemma 3 270M IT

084 repeats 083's in-range KL allocation test on Gemma 3 270M IT. The
comparison against 074 pre-KD tests whether moving rank between decoder
projections lowers measured joint KL at the same factor budget. Experiment
081 provides a three-stage reference at a higher budget.

| Experiment | Decoder allocation | Decoder factor budget | Decoder BPW |
| --- | --- | ---: | ---: |
| 074 pre-KD | Two full stages per owner | 35,371,008 B | 2.82206 |
| 081 | Three full stages per owner | 53,056,512 B | 4.23309 |
| 084 | Variable rank, aligned to 32, up to three full stages | 35,371,008 B cap | 2.82206 cap |

The budget covers signs and scales for 100,270,080 decoder projection weights.
084 reuses 074's pinned Unsloth weights, IMatrix, 256 × 2,048 compression tokens,
ADMM, layer and block tuning, refit, and Q8_0 embeddings. Global distillation
and rank retry are disabled. The minimum rank is one quarter of the uniform
rank, subject to measured coverage and alignment.

## KL pricing

Use the completed curated FP32 sweep of 160 × 256 windows, with 40,800 scored
positions and 1,890 trials across 126 projections, five noise levels, and
three seeds. For the curated corpus, the pinned Google teacher and Unsloth
compression tokenizer produce identical streams of 1,000,294 token IDs and
the same BOS ID. Their checkpoint weight files also have identical SHA-256.
The sweep and its sampling comparison are available under
[`Datasets/gemma-3-270m-it-fp32-sensitivity-curated-160x256`](../../Datasets/gemma-3-270m-it-fp32-sensitivity-curated-160x256)
and [`Results/083`](../../Results/083/270m-curated-sensitivity-sampling.md).

1. Audit 074's committed pre-KD run. Fit rank-response curves from a quarter
   stage through three full stages for its 90 physical owners across 18
   blocks. Record per-projection error for the three members of each shared
   QKV owner.
2. Measure 074 pre-KD's isolated KL on the curated teacher for each projection,
   keeping the teacher embeddings fixed. Use that measurement as the KL
   anchor. Scale it by the measured IMatrix error ratio and the projection's
   curated sensitivity exponent.
3. Price changes in estimated isolated KL per factor byte. Admit only rank
   options whose projected errors for every member lie within the 1–20%
   sensitivity range. The production allocator selects one rank per owner
   under the global factor budget. Save the rank-response curves and priced
   options as separate, content-addressed datasets.
4. Compress, export and benchmark 084. Compare joint decoder KL and full-model
   KL on the same curated windows. Require byte-identical Q8_0 embeddings
   across the exports. Compare WikiText-2 and six-task accuracy against both
   references. Compare five-shot MMLU against 081, which has existing MMLU results.

The allocator predicts a sum of isolated KL contributions. Joint decoder KL
tests whether the predicted savings carry over to simultaneous substitutions.
WikiText-2 and task accuracy test whether any measured gain transfers to the
benchmark inputs.

## Reproduction

The local sequence and console logs are under `outputs/084`. If interrupted,
resume at the last incomplete step. The rank sweep checkpoints after each
rank and the isolated scoring checkpoints after each projection. The
compression recipe validates the rank-option dataset, pins its content key,
and materializes the 074 calibration tokens under the 084 run identity.

```powershell
$python = '.\.venv\Scripts\python.exe'
$source = 'evidence/074/074-d2-uniform-control-gemma-3-270m-it'
$snapshot = Join-Path $HOME '.cache/huggingface/hub/models--unsloth--gemma-3-270m-it/snapshots/23cf460f6bb16954176b3ddcc8d4f250501458a9'
$sweep = 'Datasets/gemma-3-270m-it-fp32-sensitivity-curated-160x256/measurements/af8e37237c8fc3b6e07391253c01098138f5a3a6253abc9c68fb74b5a5f936cb/results.json'
$imatrix = 'Datasets/gemma-3-270m-it-imatrix-unsloth/imatrix_unsloth.gguf_file'
$control = 'Results/074/gemma-3-270m-it-nanoquant-pre-kd.gguf'
$rankData = 'Datasets/gemma-3-270m-it-staged-rank-response-074'
$klData = 'Datasets/gemma-3-270m-it-curated-kl-rank-options-084'
$attr = 'outputs/084/074-curated-attribution.json'
$env:HF_DATASETS_CACHE = (Resolve-Path 'outputs/imatrix-calibration-comparison-v1/hf-datasets').Path
& $python tools/validate_resident_run.py --run-output $source --expected-blocks 18 --require-complete --output outputs/084/074-source-audit.json
& $python tools/build_staged_rank_dataset.py --run-output $source --model (Join-Path $snapshot 'model.safetensors') --audit outputs/084/074-source-audit.json --output $rankData
& $python tools/predict_imatrix_weighted_kl.py $sweep --predict-only --snapshot $snapshot --imatrix $imatrix --gguf $control --output outputs/084/074-curated-attribution.md
& $python tools/prepare_staged_kl_reference.py $rankData --reference-label '074 pre-KD' --output outputs/084/reference-plan.json
& $python tools/validate_imatrix_attribution.py $attr --gguf $control --output outputs/084/074-curated-isolated.md
& $python tools/rerank_staged_kl_budget.py outputs/084/reference-plan.json $attr outputs/084/074-curated-isolated.json --dataset $rankData --output outputs/084/084-curated-kl-rerank.md --publish-kl-options $klData
& $python experiments/084-curated-kl-allocation-compress-and-benchmark-gemma-3-270m-it.py
& $python tools/compare_staged_rank_runs.py $attr 74 81 84 --model-slug gemma-3-270m-it --pre-kd 74 --output Results/084/084-quality-and-joint-kl-comparison.md
```
