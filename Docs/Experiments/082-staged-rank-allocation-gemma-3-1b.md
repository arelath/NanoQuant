# Experiment 082: staged rank allocation for Gemma 3 1B IT

Experiment 082 tests whether redistributing Experiment 080's decoder factor
budget reduces IMatrix-weighted reconstruction error and improves model quality.
080 uses two full residual stages everywhere. 082 shares that budget across
130 physical units, with Q, K and V planned as a single unit in each block.
It uses the pinned Gemma 3 1B IT checkpoint, Unsloth IMatrix, and staged
rank-response dataset described in [the measurement protocol](080-staged-rank-response.md).

The decoder factor budget is 1,754,947,584 bits (219,368,448 bytes, or
2.51510989010989 bits per decoder weight). This is the two-stage factor payload
including BF16 scales. The embedding policy and other export settings match
080. Report the final GGUF size separately from the factor budget.

The reconstruction allocator uses the dataset's monotone piecewise log-linear
fits through the 17 measured knots. It starts at a quarter of each unit's
uniform budget rank, rounds up to rank 32, then assigns rank in steps of 32 by
predicted reduction in summed per-unit objective NMSE per bit. Each unit is
capped at three full stages. Sensitivity weighting, protected rank floors and
rank retries are disabled. This greedy solver searches the aligned ranks under
the shared budget. It can choose ranks between measured knots.

Compression follows 080's ADMM, per-layer tuning and block refit recipe.
Use the same validated 256 × 2,048 calibration tokens under a new run-local
receipt. Global KD is disabled. The dataset curves describe reconstruction
before activation tuning. Compare predicted error and rank distribution with
the two-stage plan, then compare the exported model with 080 using the same
dense WikiText-2 protocol, six tasks, and full MMLU. Check actual factor bits and
embedding equality on the exports before interpreting a quality difference.

The launcher uses the standard complete compression and quality workflow. Start
it after the dataset artifact is published and the CUDA device is free. For the
first launch, use the Hugging Face datasets cache from 080 and materialize 080's
calibration tokens with 082's config hash as the preparation ID. The preflight
receipt at `outputs/082/preflight.json` records the dataset key, calibration
fingerprint, resolved config and projected ranks.

The published dataset contains 2,210 measurements. CPU preflight projected a
3.35% reduction in summed objective NMSE (3.462853 to 3.346973), with ranks
1,696–2,816 and 6,336 decoder factor bytes left in the budget.

Launched on 2026-11-16 after dataset publication and a CUDA lease check.
Startup reached calibration with a measured tuning batch size of eight.
The launch logs are under `outputs/082/`.

```powershell
.\.venv\Scripts\python.exe experiments/082-staged-rank-allocation-compress-and-benchmark-gemma-3-1b-it.py
```
