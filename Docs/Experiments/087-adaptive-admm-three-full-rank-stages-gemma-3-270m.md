# Experiment 087 — Adaptive block ADMM on Gemma 3 270M IT

## Question

Does spending more ADMM iterations on blocks with high final output error improve model quality at the same ranks as Experiment 081?

## Compression and acceptance rule

Use 081’s pinned `unsloth/gemma-3-270m-it` revision `23cf460f6bb16954176b3ddcc8d4f250501458a9`, input-column imatrix, calibration protocol, and local tuning. Fit three independently scaled rank-640 residual stages to each of the five physical owners in each of 18 decoder blocks. The attention Q, K and V projections share an owner. Global distillation is disabled.

Start **every block** at 3,200 outer ADMM iterations per residual stage, with two inner iterations and the cubic penalty schedule. After tuning, refitting and freezing the whole block, forward the compressed inputs through it. Divide the output-importance-weighted MSE against the teacher by the teacher’s weighted mean-square output. This measures cumulative error at that block boundary, including error arriving from previous blocks.

Accept when the ratio is strictly below 0.01. Otherwise refit the same block from its source weights and the same incoming activations at 6,400, 12,800, 25,600 and finally 51,200 iterations per stage. Each retry restarts its ADMM schedule. Accept the 51,200-iteration attempt even if it misses the target. Only the accepted block propagates activations to its successor. Each attempt has a separate journal and a decision receipt under the resident run’s `state/block-admm` directory.

## Benchmark and comparison

After compression, validate the frozen run, export the packed model and GGUF, and benchmark the dense replay of the packed candidate against BF16 using 081’s protocol. Report WikiText-2 perplexity on 64 × 128 tokens, PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande and BoolQ on up to 1,000 examples each, and five-shot MMLU on all 14,042 questions.

081 recorded WikiText-2 perplexity of 218.1163 versus 194.0538 for BF16, and MMLU accuracy of 26.29% versus 26.39% for BF16. Its GGUF occupied 238,158,624 bytes. Compare the same metrics, storage and per-block decisions after 087 finishes. A lower block error alone does not establish a downstream quality improvement.

## Launch

From the repository root in PowerShell:

```powershell
$env:HF_DATASETS_CACHE = (Resolve-Path outputs/imatrix-calibration-comparison-v1/hf-datasets).Path
.\.venv\Scripts\python.exe experiments/087-adaptive-admm-three-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py
```

The resident run lives under `evidence/087`. The workflow writes the final GGUF and quality reports to `Results/087`.

## Launch record

Started locally on the RTX 4000 Ada on November 23, 2026 at 7:41 p.m. Pacific. The 256 × 2,048 fitting tokens match 081’s verified calibration fingerprint `sha256:edd62d43b7ec77efcdcf8be890d46c32d478c9aa42dfcc04824147256e387833`. The local tokenizer matches the 081 benchmark hash and both pinned MMLU splits are present. The launch log is `outputs/087/087-launch.log`.
