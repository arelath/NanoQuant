Yes. The strongest opportunity is not more ADMM or longer tuning—it is fixing or bypassing global KD, then reallocating the existing serialized bytes more intelligently.

### What Experiment 003 says

[Experiment 003](D:/dev/research/NanoQuantRewrite/evidence/m9/003-gemma-3-1b-it-quality.json:2) is a smoke protocol: 16 WikiText windows and 25 examples per task. Its `passed: true` means evaluation completed, not that quality passed:

- WikiText PPL: 94.80 BF16 → 396.57 NanoQuant
- PIQA: 0.72 → 0.64
- ARC-Easy: 0.52 → 0.32
- BoolQ: 0.76 → 0.60

The broader [200-example benchmark](D:/dev/research/NanoQuantRewrite/evidence/m9/002-gemma-3-1b-it-quality-benchmark.md) confirms the degradation: PPL 96.90 → 453.57 and substantial losses on all six tasks.

However, the rewrite is only 2.08% behind contemporary legacy NanoQuant, with a 0.75% mean block-trajectory difference. That indicates this is primarily a limitation of the current ~1-BPW recipe, not a major rewrite correctness defect. See the [full parity summary](D:/dev/research/NanoQuantRewrite/evidence/m4/gemma-pageable-v28-four-block-canary/full-parity-summary.json).

### Recommended experiments, in order

1. Disable or validation-gate global KD.

The immutable pre-KD model scores PPL 415.16; KD worsens it to 453.57—9.25% worse—even though the cached KD objective decreases from 2.3988 to 2.1484.

There is a concrete objective problem in [`topk_distillation_loss`](D:/dev/research/NanoQuantRewrite/src/nanoquant/application/distillation.py:188): both teacher and student probabilities are normalized only among the teacher’s top-64 tokens. Student probability assigned outside those 64 tokens is invisible to the loss.

I would:

- Immediately evaluate the pre-KD artifact on the full task suite.
- Include “no KD” as checkpoint zero and restore it unless held-out next-token loss/PPL improves.
- Replace the loss with a top-k-plus-tail KL: cache teacher full log-normalizer and tail mass; compute the student full-vocabulary denominator chunkwise.
- Optionally blend hard-label next-token CE.
- Use separate held-out sequences rather than selecting on the reused KD cache.

Although `FULL_KL` exists in configuration, the resident workflow currently rejects it as unimplemented at [`resident_workflow.py`](D:/dev/research/NanoQuantRewrite/src/nanoquant/resident_workflow.py:228).

2. Reclaim bytes that currently add no information, then spend them on rank.

The logical report says 0.9963 BPW, but the exported NanoQuant tensors occupy 88,742,840 bytes, or 1.01746 physical BPW. The main discrepancy is that 921,216 scale values are charged as 16-bit but serialized as F32. The frozen values are already BF16, so widening them adds no precision.

Storing those scale sidecars as BF16 would save 1,842,432 bytes. The raw byte arithmetic is enough to raise every current `k_proj` and `v_proj` to their maximum rank 256—approximately 980 KB—while remaining below the current serialized size. This requires converter/kernel support for BF16 sidecars, but the kernel can still upcast during computation.

This is especially promising because the [reconstruction table](D:/dev/research/NanoQuantRewrite/evidence/m4/gemma-pageable-v28-four-block-canary/artifacts/ed/sha256-ed7a3ab022aae1460af83d3bdef20b269eaffcd23102a7c9fb5aa4d2aeb1b340/reconstruction.md) shows:

- `v_proj`: mean weighted error 0.425, mostly rank 160
- `k_proj`: mean 0.276, mostly rank 128
- `q_proj`: mean 0.202, mostly rank 448

So the highest residual error is concentrated in comparatively cheap attention projections.

3. Replace the scalar rank allocator with measured marginal rate–distortion allocation.

The current allocator:

- Assigns outliers as a uniform input-column fraction.
- Reduces sensitivity to `mean(input) × mean(output)`.
- Reuses one fixed utility value for every additional rank chunk, with no diminishing-return curve.

See [`application/planning.py`](D:/dev/research/NanoQuantRewrite/src/nanoquant/application/planning.py:67) and [`domain/planning.py`](D:/dev/research/NanoQuantRewrite/src/nanoquant/domain/planning.py:90).

Instead, measure each layer at `r−32`, `r`, and `r+32`, then greedily select the best actual loss reduction per serialized byte. Probe block-boundary loss rather than weight error alone. A useful initial constrained swap is two `q/o` rank chunks for three `v/k` chunks, followed by downstream replay.

4. Use more INT8 salient columns at the same byte cost.

Current outliers consume about 2.14 MB in BF16. Existing code already supports per-column INT8 storage in [`outliers.py`](D:/dev/research/NanoQuantRewrite/src/nanoquant/domain/outliers.py:47). The same columns would need about 1.07 MB including scales, allowing roughly twice as many salient values—or a mixture of extra columns and rank—within the existing size.

Allocate columns globally by residual reduction per serialized byte, rather than giving every layer 0.1%. An outlier column in a 6912-output MLP projection is much more expensive than one in a 256-output attention projection.

Before doing this, fix the accounting in [`outlier_bit_cost`](D:/dev/research/NanoQuantRewrite/src/nanoquant/domain/planning.py:25): it derives index width from `out_features` even though indices identify input columns, while the GGUF actually stores I32 indices. INT8 scale sidecars are also currently omitted from the estimate.

5. Preserve tuned binary latents and try global binary QAT.

Global tuning currently reconstructs hard ±1 factors and sets `immutable_binary_factors=True`; only scales, outliers, biases, and norms are optimized in [`global_distillation.py`](D:/dev/research/NanoQuantRewrite/src/nanoquant/global_distillation.py:176).

A higher-potential version would preserve the final pre-sign latent margins as training-only artifacts, then globally tune them with:

- A much smaller binary-factor LR than scale/outlier LR
- Gradient clipping or a trust region
- Held-out rollback
- The corrected KD/CE objective

This changes training artifacts, not deployment size. Binary QAT combined with distillation has performed well in other ultra-low-bit work such as [BitDistiller](https://arxiv.org/abs/2402.10631) and [OneBit](https://arxiv.org/abs/2402.11295), though the local held-out gate should remain authoritative.

### If “model size” means the entire GGUF

There is an even larger lever: the 699.9 MB GGUF contains:

- 604.0 MB BF16 tied token embedding/output head
- 88.7 MB NanoQuant decoder tensors
- About 7.1 MB everything else

Quantizing the tied embedding to Q8 and reallocating the roughly 280 MB saved to decoder rank would likely dominate all smaller optimizations. It must be tested carefully because the same tensor is also the output head. I would treat this as a separate fixed-total-file experiment, not as an apples-to-apples 1-BPW decoder comparison.

### What I would run next

1. Full task evaluation of the existing pre-KD artifact.
2. Corrected top-k-plus-tail KD, selected by held-out PPL with pre-KD rollback.
3. BF16 serialized scales plus exact serialized-byte rank reallocation, starting with full-rank `k_proj`/`v_proj`.
4. Cost-aware INT8 outlier allocation.
5. Only then global binary QAT.

I would not prioritize extra least-squares passes, ADMM multistart, or a uniform 32-epoch schedule: retained experiments already found LS saturation, negligible multistart benefit, and catastrophic later-block divergence from the longer schedule in [the optimization evidence](D:/dev/research/NanoQuantRewrite/Docs/16-behavior-preserving-optimizations.md:863). Hessian-aware objectives and incoherence rotations remain reasonable longer-term research tracks—the [NanoQuant paper](https://arxiv.org/abs/2602.06694) emphasizes Hessian-aware factorization, while [QuIP#](https://arxiv.org/abs/2402.04396) and [QuaRot](https://arxiv.org/abs/2404.00456) demonstrate the value of incoherence/rotations—but the measured KD and byte-allocation issues are much higher-confidence first moves.