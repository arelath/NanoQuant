# Residual stage block 12 functional gate

Date: 2026-11-10

## Result and decision

The rank-704 residual factor improves model-logit KL and NLL relative to the equal-budget INT8 residual-column pilot. It does **not** beat the unchanged Experiment 060 layer on KL. Do not promote this candidate to the resident recipe or deployment format.

This completes the first model-level selection screen in [the experiment plan](109-residual-stage-quality-experiment-plan.md). It tests one matrix and one fitting seed in the complete recovered 060 model context. Multi-block fits, additional seeds, and independent confirmation remain open.

| Block-12 down projection | C4 NLL | Teacher KL, nats/token |
|---|---:|---:|
| Unchanged 060 reference | 3.900233 | **0.701268** |
| Pilot first stage + INT8 residual columns | 3.903633 | 0.711173 |
| Pilot first stage + partial sign plane | 3.901628 | 0.711904 |
| Pilot first stage + rank-704 residual factor | **3.897066** | 0.707412 |
| Original BF16 teacher | 3.650550 | 0 |

The three pilot replacements share a 2.0-BPW cap, including the diagnostic container overhead and FP32 scales described in Doc 109. Their common first stage is the retained Experiment 067 fit under the Experiment 041 diagonal-Fisher objective. The unchanged 060 layer is a separate, lower-byte reference; it is not the same first-stage initialization or an equal-budget reconstruction control.

## Paired comparisons

Negative differences favor the residual candidate. Confidence intervals use 10,000 paired sequence bootstrap resamples, seed 0. The primary comparison was residual factor versus INT8 residual columns. Other comparisons are exploratory and have no multiplicity adjustment.

| Comparison | KL difference | KL paired 95% interval | NLL difference | NLL paired 95% interval |
|---|---:|---|---:|---|
| Residual factor minus INT8 pilot | -0.003761 (-0.529%) | [-0.005032, -0.002484] | -0.006567 | [-0.008743, -0.004399] |
| Residual factor minus sign plane | -0.004492 | [-0.006254, -0.002791] | -0.004562 | [-0.006404, -0.002629] |
| Residual factor minus unchanged 060 | +0.006143 (+0.876%) | [+0.003837, +0.008399] | -0.003167 | [-0.006514, +0.000331] |
| Sign plane minus INT8 pilot | +0.000731 | [-0.000893, +0.002456] | -0.002005 | [-0.004427, +0.000288] |

The predeclared primary screen required the KL interval's upper endpoint below zero and the NLL interval's upper endpoint no greater than +0.01 nats/token. The residual factor passes this controlled screen. However, the INT8 pilot itself regresses KL against unchanged 060 by +0.009904, with interval [+0.007126, +0.012571]. Improving that weaker control is not evidence of beating our incumbent.

The sign plane's much lower weight error did not yield a supported KL improvement over INT8. The residual factor's approximately 46% reconstruction-error reduction yielded only a 0.53% KL reduction for this one-layer replacement. This reinforces the need for model-level evaluation rather than extrapolating weight error into capability gains.

## Protocol and reconstruction audit

- Source model: `google/gemma-3-1b-it`, revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
- Student baseline: `Results/060/gemma-3-1b-it-nanoquant.gguf`, SHA-256 `5e445cab6c93f03492ca0c9cea4ce7977fc780b9dbd75ad60578aef609dad02c`.
- Restored all 1,198 GGUF tensors into 182 dense projection matrices, the six norms per block, final norm, and tied Q8 embedding/output weights. Shared QKV factors are reconstructed together and split in Q/K/V row order.
- Gemma norm conversion is `HF weight = GGUF effective gamma - 1`. Factor pre/mid/post scale tensors in 060 are **BF16**; norm tensors and salient-column scales are F32. Doc 108's earlier claim that factor scales were F32 was corrected.
- Both models execute through the same Transformers BF16 dense path. This is a GGUF-derived mathematical reconstruction, not a native llama.cpp runtime parity result or a rerun of 060's historical benchmark.
- Teacher remains the original BF16 model, including its original embedding. Every student arm shares the recovered Q8 embedding, all norms, and every non-target weight.
- Dataset: pinned C4 validation shard `en/c4-validation.00000-of-00008.json.gz`, revision `f998d2cd8b92435980789e3ecb2f89b4c68bfe1e`; file SHA-256 `bc35d7c1b1d14b90cd3a394cccbcbe191935edd04bf42ee965379c6e2987a5f0`.
- Tokens: the existing contiguous-stream protocol over the first 1,100 documents; offset 472, 48 sequences of 512 tokens, 24,528 scored next-token targets. No overlap with prior registered C4 intervals.
- Slice `residual-stage-c4-validation472-48x512` was permanently retired before scoring. Token hash: `sha256:a0f51fb416497af542c0a04f3ee1c4cec761a0acc75ad54a693cad85de97f4c8`.
- Full-vocabulary causal KL/NLL uses the existing evaluator kernel, token chunks of 32, and one sequence at a time. Teacher logits are shared across all student arms for each sequence.
- Peak PyTorch CUDA allocation: 4,868,748,288 bytes (about 4.53 GiB), excluding other applications and driver reservations.

The first load attempt stopped before scoring because it assumed independent Q/K/V units. The corrected run handles the shared group and checks that every GGUF tensor is consumed. The original attempt's protocol is retained; the successful output uses a separate `block12-functional-v2` directory.

## Evidence and reproduction

Evidence root: `outputs/residual-stage-gate-20261110/`.

- `c4-selection/`: token tensors, receipt, and registry snapshots before/after retirement.
- `block12-functional-v2/protocol.json`: immutable input, source, payload, and execution identities and the predeclared gate.
- `block12-functional-v2/restoration.json`: complete tensor reconstruction inventory.
- `block12-functional-v2/progress.json`: all per-sequence measurements.
- `block12-functional-v2/summary.json`: aggregate metrics and paired confidence intervals.
- `block12-functional-v2/executed-driver.py`: exact driver used for the completed evaluation, before adding a source-layer argument guard and Python 3.10-compatible streaming hashing.

The checked-in runner is `tools/probe_residual_stage_functional.py`. It verifies candidate hashes and byte caps, source-layer identity, token hashes and retirement, and baseline identity. It rejects a changed protocol on resume. A diagnostic replay of this now-retired selection slice is not fresh confirmation.

```powershell
.\.venv\Scripts\python.exe tools/probe_residual_stage_functional.py --snapshot C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752 --pilot outputs/residual-stage-gate-20261110/block12-seed1 --tokens outputs/residual-stage-gate-20261110/c4-selection/tokens.safetensors --output outputs/residual-stage-gate-20261110/block12-functional-replay
```

## Next gate

Keep the current 060 layer. Continue with fresh fitting objectives and matched functional tuning for blocks 0, 12, and 25 before drawing conclusions about the residual representation. The current single-matrix pilot uses an older importance objective and does not reproduce 060's tuning. Test multiple fitting seeds, compose any winners, and use a separate confirmation slice. Runtime/schema work remains deferred until a candidate improves the incumbent under the final gate.

## Validation

The full CPU suite passed with 1,525 tests and 49 deselected. After adding the shared-QKV regression check, all ten focused residual-budget and functional tests passed. Repository Ruff, mypy (217 source files), and the evaluation-slice registry validator passed. The archived executed driver matches its protocol hash. The evaluation process exited successfully and the CUDA lease was verified available afterward.
