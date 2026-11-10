# Residual-stage multi-block gate

Date: 2026-11-10

## Result and decision

The frozen three-block residual-factor composition passes both selection and independent confirmation against unchanged 060. On 96 confirmation sequences, it lowers teacher KL by **2.47%** and NLL by **0.01232 nats/token**, with paired 95% intervals excluding zero for both. It also beats the matched tuned INT8 and sign-plane compositions on both endpoints in confirmation.

Retain this as a successful dense research candidate and continue to the controlled distillation comparison in Doc 109. It changes three projections, costs an additional 279,240 diagnostic payload bytes versus their 060 tensor payloads, and is not a complete compressed-model or native-runtime result. Do not replace the production recipe or claim downstream capability gains from this gate.

## Predeclared protocol

Continue Doc 109 with blocks 0, 12, and 25 `mlp.down_proj`, seeds 0, 1, and 2, and 800 production ADMM iterations followed by two-pass scale fitting. This is a dense diagnostic, not a complete compression/export run or a resident algorithm change.

Use 16 new contiguous C4 validation sequences of 512 tokens at offsets 520–535 for fresh original-teacher online Fisher calibration and functional tuning. Normalize each importance vector to mean one and floor at 1e-6. Choose seven protected columns per block by weighted original-target column energy and preserve their original BF16 values identically in every arm. These objectives replace the unavailable 060 calibration; they do not reproduce it.

For every block and seed, fit one full-rank first stage and freeze it as the common initialization for INT8 residual columns, a partial sign plane, and an independently scaled residual factor. Retain all seeds and select each arm's lowest weighted NMSE seed before inspecting model-level metrics. Charge the existing 2.0-BPW diagnostic cap, FP32 scales, protected values, indices, 4,096-byte header and 32-byte alignment; round-trip actual payloads before evaluation.

Compose the three chosen blocks for each representation in the GGUF-recovered 060 model. Keep unchanged 060 as the incumbent reference. Test untuned and matched full-model KL-tuned variants. Tuning is restricted to the common first-stage pre/mid/post scale roles; signs, corrections, protected values, norms and all other layers stay frozen. Use 32 sequence updates (two passes through the same 16 fit sequences), Adam at 1e-4, gradient norm cap 1, no weight decay, and a fixed final endpoint. Bound each log-scale multiplier to [-0.1, +0.1]. There is no local-output proxy or selection-data early stopping. Charge updated scales in the payload; no optimizer state is part of inference storage.

The selection slice is a separate 48-by-512 C4 window at offsets 536–583. Primary comparison: composed tuned residual factors versus unchanged 060; also compare with the matched tuned INT8 and sign controls and untuned variants. Use sequence-paired 10,000-resample bootstrap intervals, seed 0. Require the KL difference's upper 95% endpoint below zero and the NLL difference's upper endpoint no greater than +0.01 nats/token. Exploratory comparisons are not multiplicity-adjusted. A selected winner must pass again on an independent 96-sequence confirmation window (offsets 584–679), consumed only after the candidate is frozen. If no candidate beats 060, retain 060 and do not consume confirmation data or implement runtime support.

The first screen does not establish whole-model recipe quality, native GGUF runtime parity, or downstream capability. Those remain later gates.

## Checklist

- [x] Fresh calibration and all nine first/residual seed fits complete.
- [x] Byte-accounted representatives selected without selection-data access.
- [x] Matched full-model scale tuning complete and persisted.
- [x] Composed functional selection complete.
- [x] Independent confirmation if the primary gate passes.
- [x] Report measured results, limitations and next decision; validate and commit tools/docs.

## Reconstruction screen

The fresh calibration completed on the original teacher. All nine first-stage fits and all nine residual-stage fits completed at 800 ADMM steps, using the existing production factorization and two-pass scale fitter. Every candidate was serialized and reconstructed before measuring NMSE. The residual stage has rank 704; the first stage has rank 1,152.

| Block | INT8 NMSE (seed) | Sign-plane NMSE (seed) | Residual-factor NMSE (seed) |
|---:|---:|---:|---:|
| 0 | 0.062788 (2) | **0.036991 (1)** | 0.037894 (2) |
| 12 | 0.149175 (2) | 0.087702 (1) | **0.084290 (1)** |
| 25 | 0.066556 (2) | 0.040321 (0) | **0.038509 (2)** |

Each INT8 payload occupies 1,989,664 bytes (1.999003 BPW); each sign-plane payload 1,990,560 bytes (1.999904 BPW); each residual-factor payload 1,979,168 bytes (1.988458 BPW). The shared cap is 1,990,656 bytes. The unchanged 060 projection payload is a lower-byte reference at 1,886,088 bytes per block.

All arms within a given seed share the first stage. Independently choosing each arm's best seed can choose different initial stages across the final representatives, so the table compares the matched selection procedures, rather than isolating representation alone at one initialization. All per-seed states remain available. These fresh objective values also cannot be numerically compared with Doc 109's older Fisher objective.

Evidence: `outputs/residual-multiblock-gate-20261110/fits/`. Its `protocol.json`, `calibration.json`, per-block fixtures, per-seed first stages and packed pilot payloads preserve input and output identities. `selection.json` fixes the representatives before model-level scoring.

## Selection result and frozen confirmation candidate

The primary selection gate passed on all 48 sequences (24,528 next-token targets). Each tuned arm updated exactly 27,648 common first-stage scale parameters. Updated payloads retain the original byte sizes. Full-vocabulary KL gradients were computed in 32-token chunks at the final hidden states, then propagated through the complete student model. The pinned model has no final logit softcap; the tool rejects a model with one.

| Three-block composition | NLL | Teacher KL |
|---|---:|---:|
| Unchanged 060 | 4.313930 | 0.737356 |
| Untuned INT8 | 4.323345 | 0.745500 |
| Tuned INT8 | 4.323189 | 0.741903 |
| Untuned sign plane | 4.316842 | 0.734435 |
| Tuned sign plane | 4.313832 | 0.729896 |
| Untuned residual factor | 4.303478 | 0.729720 |
| **Tuned residual factor** | **4.301216** | **0.725998** |
| Original BF16 teacher | 4.026002 | 0 |

The tuned residual candidate minus unchanged 060 has KL difference -0.011357 (-1.540%), paired 95% interval [-0.018456, -0.004024], and NLL difference -0.012714, interval [-0.024975, -0.000893]. Versus tuned INT8, its KL difference is -0.015905 (-2.144%), interval [-0.022770, -0.009863]. Its KL difference versus tuned sign planes is not significant: -0.003898, interval [-0.008329, +0.000716]. The sign-plane NLL interval versus 060 exceeds the predeclared +0.01 margin slightly, so it does not pass that joint gate on selection.

Freeze the tuned residual-factor candidate from `functional/tuned_residual_factor/receipt.json`; confirmation uses `--trained-from` and must not retrain. The composed candidate changes only the three down projections in the recovered 060 context. Each projection's diagnostic payload is 93,080 bytes larger than the incumbent's NanoQuant tensor payload, a total of 279,240 bytes. This is a common-cap comparison between the new representations, not exact byte parity with 060.

Selection evidence is under `outputs/residual-multiblock-gate-20261110/functional/`, with all per-sequence rows, paired intervals, updated payloads and receipts. Peak PyTorch CUDA allocation was 5,822,808,576 bytes (about 5.42 GiB), excluding other applications and driver reservations. The independent confirmation window was not consumed until after this selection decision.

## Independent confirmation

All 96 sequences completed, covering 49,056 next-token targets. Confirmation loaded the exact payloads frozen after selection, verified their hashes, and performed no training. The primary residual-factor comparison was specified before either evaluation window was scored. Other arms provide supporting comparisons; their intervals are exploratory and not multiplicity-adjusted.

| Three-block composition | NLL | Teacher KL |
|---|---:|---:|
| Unchanged 060 | 4.360860 | 0.686709 |
| Untuned INT8 | 4.370231 | 0.689870 |
| Tuned INT8 | 4.367240 | 0.684391 |
| Untuned sign plane | 4.360833 | 0.679978 |
| Tuned sign plane | 4.359938 | 0.676049 |
| Untuned residual factor | 4.350671 | 0.674309 |
| **Frozen tuned residual factor** | **4.348544** | **0.669739** |
| Original BF16 teacher | 4.081602 | 0 |

| Confirmation comparison | KL difference | KL paired 95% interval | NLL difference | NLL paired 95% interval |
|---|---:|---|---:|---|
| Tuned residual minus unchanged 060 | -0.016969 (-2.471%) | [-0.024707, -0.010551] | -0.012317 | [-0.023061, -0.002957] |
| Tuned residual minus tuned INT8 | -0.014652 (-2.141%) | [-0.017711, -0.011687] | -0.018696 | [-0.024820, -0.012783] |
| Tuned residual minus tuned sign | -0.006310 (-0.933%) | [-0.009024, -0.003434] | -0.011395 | [-0.017598, -0.005072] |
| Tuned residual minus untuned residual | -0.004570 (-0.678%) | [-0.005156, -0.003965] | -0.002127 | [-0.003079, -0.001220] |

The untuned residual composition also improves KL versus 060 by 1.81%, with difference -0.012399 and interval [-0.019989, -0.006002]. Thus the observed incumbent gain does not depend entirely on the added scale updates. Matched tuning improves all representations' confirmation KL; it does not make the INT8 control significantly better than 060. The sign-plane composition improves KL, while its NLL interval includes zero. Its confirmation comparisons remain secondary because it did not pass the joint selection gate.

Confirmation evidence is under `outputs/residual-multiblock-gate-20261110/confirmation/`. `protocol.json` records the frozen training-receipt hashes, and `summary.json`/`progress.json` preserve all aggregate and sequence-level results. Peak confirmation CUDA allocation was 4,851,971,072 bytes (about 4.52 GiB).

## Provenance, reproduction and limits

The source is the pinned `google/gemma-3-1b-it` revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`; source weight SHA-256 is `3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6`. The complete student context is restored from the same hash-verified 060 GGUF as Doc 110, including shared QKV, norm gamma conversion, and the tied Q8 embedding/output tensor.

All data uses the same pinned C4 validation shard and contiguous-stream tokenizer protocol as Doc 110. The three disjoint slices were permanently retired before use:

| Role | Sequence offsets | Token hash |
|---|---|---|
| Fisher fit and tuning | 520–535 | `sha256:1449374870ab4a3fb68b641806dc92068ec8b5d1482c23ef6bfc7263e09c80ad` |
| Selection | 536–583 | `sha256:99b9855097fd62f330b0b8ecc5c5c27f22523f7b3226baf6f24d50b8c642b010` |
| Confirmation | 584–679 | `sha256:31c30087cc8390f34c4e57e7e73b11bef34e02151db9802d74747ebb61e5a1cb` |

Root evidence is `outputs/residual-multiblock-gate-20261110/`. Token files have receipts and registry snapshots; `tool-hashes.json` identifies the helpers. The exact executed fit, budget-helper and functional scripts are archived alongside their protocols, and their hashes were verified. Replaying a retired window is a diagnostic replay, not new confirmation.

Compact protocol snapshots and full aggregate/per-sequence selection and confirmation results are also committed under `evidence/residual-multiblock-gate-20261110/`; the large fitting fixtures and packed payloads remain in the output directory.

```powershell
# Use a new output directory for a diagnostic replay; these inputs are already retired.
.\.venv\Scripts\python.exe tools/probe_residual_multiblock_fit.py --snapshot C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752 --tokens outputs/residual-multiblock-gate-20261110/c4-fit/tokens.safetensors --output outputs/residual-multiblock-gate-20261110/fits-replay
.\.venv\Scripts\python.exe tools/probe_residual_multiblock_functional.py --snapshot C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752 --fit-tokens outputs/residual-multiblock-gate-20261110/c4-fit/tokens.safetensors --fits outputs/residual-multiblock-gate-20261110/fits --tokens outputs/residual-multiblock-gate-20261110/c4-confirmation/tokens.safetensors --output outputs/residual-multiblock-gate-20261110/confirmation-replay --role confirmation --trained-from outputs/residual-multiblock-gate-20261110/functional
```

This improves only three `down_proj` matrices in an otherwise unchanged 060 model. The formats share a byte cap but use different exact byte totals; the incumbent uses fewer bytes. Fresh calibration, different protected columns, multiple seeds and three-block composition distinguish this experiment from the failed older-fixture pilot, so that reversal does not isolate one causal change. C4 fit and evaluation windows are disjoint but come from the same shard; this is not out-of-domain or task-suite acceptance. The short scale-only tuning is also not the planned full distillation objective ablation.

Next, preserve a common starting state for the current-objective/full-KL/full-KL-plus-hidden-anchor comparison. The old pre-distillation checkpoint is unavailable; if it cannot be recovered, explicitly define a new matched continuation experiment from the hash-verified 060 reconstruction rather than claiming a historical rerun. Keep the residual candidate frozen, compare against equally re-tuned 060, then extend any winning representation to a complete model through the mandatory compression/export workflow and task/runtime gates.

## Validation

Thirteen focused residual tests passed. The full suite passed with 1,529 tests and 49 deselected in 184.54 seconds. Ruff passed across the repository and mypy passed for all 217 source files. New CPU tests verify the chunked full-vocabulary KL gradient against direct autograd, protected-column invariance, frozen correction/sign persistence and unchanged payload size after scale updates. The permanent slice registry validates with 24 retired slices and no reserved slices. Both selection and confirmation workers exited successfully; no experiment remains running.
