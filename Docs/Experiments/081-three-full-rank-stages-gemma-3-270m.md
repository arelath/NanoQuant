# Experiment 081: Three full-rank stages on Gemma 3 270M IT

Prepared 2026-11-16. The experiment has not started.

081 uses the same pinned `unsloth/gemma-3-270m-it` checkpoint, Unsloth imatrix,
fitting tokens, and compression speed settings as 074. The rank ceiling changes
from two to three times physical rank. Global KL distillation is disabled so the
comparison uses 074's pre-KD snapshot. Per-layer tuning, block tuning and refitting
remain part of the recipe. Quality includes WikiText-2, the six legacy tasks at
1,000 examples each, and full five-shot MMLU. Comparing MMLU against 074 pre-KD
will require scoring that snapshot on the same MMLU protocol.

## Rank and storage plan

The existing additive representation fits each new term to the residual after
all previous terms, gives each term its own pre, mid and post scales, and tunes
the terms together. The maximum-rank policy promotes every physical owner to
its configured ceiling after ordinary allocation. The inherited 2.0 BPW
planning target does not trim those ranks.

| Owner per decoder block | Shape | Planned stage ranks |
| --- | ---: | --- |
| MLP gate and up | 2,048 × 640 | 640 + 640 + 640 each |
| MLP down | 640 × 2,048 | 640 + 640 + 640 |
| Attention output | 640 × 1,024 | 640 + 640 + 640 |
| Stacked QKV | 1,536 × 640 | 640 + 640 + 640 |

Across 18 blocks, all 90 physical owners have three terms, for 270 terms over
126 projections. The planned signs and scales cost 424,452,096 bits over
100,270,080 projection weights, or **4.23309 BPW**. 074's equivalent two-stage
cost is 2.82206 BPW. These numbers exclude the embedding, any output head,
norms and GGUF overhead.

## Setup checks and later launch

The plan and dry run are saved in `outputs/081`. An inventory check verified
all 90 owner shapes and the imatrix coverage of 126 projections. 074's retained
256 × 2,048 fitting tokens were validated and materialized under 081's own
run identity. The full MMLU inputs and all legacy task splits are locally available.
No 081 model, factorization, benchmark or GGUF export was run during setup.

The CPU integration test exercises actual three-full-rank fitting, shared QKV,
tuning, resume, accounting, packed replay and the llama.cpp converter checkpoint.
The measured tuning batch, compiled CUDA execution, full GGUF and quality are
future run results. The original 0.75 GiB WDDM shared-memory guard applies.

```powershell
.\.venv\Scripts\python.exe experiments/081-three-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --dry-run
```

When ready to launch, check that the GPU is free, set `HF_DATASETS_CACHE` to the
absolute `outputs/imatrix-calibration-comparison-v1/hf-datasets` path for the
verified legacy evaluation cache, and remove `--dry-run`.
