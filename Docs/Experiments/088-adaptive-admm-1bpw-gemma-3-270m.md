# Experiment 088 — Gemma 3 270M IT at a 1 BPW projection budget

088 measures how much quality the model retains at a 1.0 bit-per-weight target for its decoder projections. It uses the resolved configuration of 087, removes the wildcard that forced all factor owners to three full ranks, and changes the uniform allocation target to 1.0. The calibration tokens, imatrix, stacked QKV, optimizer, tuning and evaluation settings match 087.

## Planned storage

The planner charges packed signs and BF16 scales to each owner’s budget and rounds ranks down to multiples of 32. In each of 18 blocks, the MLP gate, up and down owners each get rank 448, attention output gets 352, and stacked QKV gets 416. The initial plan totals 95,901,696 bits over 100,270,080 projection weights, or **0.95643 BPW**. The inherited rank-retry allowance can add at most 1,002,700 bits across the model, keeping projection storage below 1.0 BPW. The complete GGUF also contains embeddings, norms and metadata.

## Block retries and quality

Each block starts at 3,200 ADMM outer iterations. A final frozen weighted output MSE of at least 1% of teacher output power triggers a new fit at 6,400, 12,800, 25,600 and finally 51,200 iterations. The final attempt is accepted at the cap. The budget resets for the next block.

After compression and export, the workflow evaluates WikiText-2 (64 × 128 tokens), six zero-shot tasks (up to 1,000 rows each), and full five-shot MMLU (14,042 questions) against BF16. The 087 compressed baseline scored 200.1203 WikiText-2 perplexity and 26.41% MMLU accuracy. Compare measured projection BPW, complete GGUF size, capped blocks and benchmark scores after 088 finishes.

## Launch

```powershell
$env:HF_DATASETS_CACHE = (Resolve-Path outputs/imatrix-calibration-comparison-v1/hf-datasets).Path
.\.venv\Scripts\python.exe experiments/088-adaptive-admm-1bpw-compress-and-benchmark-gemma-3-270m-it.py
```

The resident journal lives under `evidence/088`. The launch log lives under `outputs/088` and the final GGUF and quality reports go to `Results/088`.
