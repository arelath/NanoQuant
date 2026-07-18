# Gemma 3 1B parity completion audit

## Decision

The NanoQuant rewrite has measured behavioral and quality parity with the contemporary legacy implementation on
the pinned `google/gemma-3-1b-it` workload. The retained candidate passed the ordered replay, quick, standard, and
full evaluation campaign with outcome `full-promotion`.

This decision closes the 1B legacy-parity objective. It does not claim that every release, distributed-execution,
70B-scaling, packaging, or operations task in the implementation checklist is complete.

## Pinned identity

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Local snapshot:
  `C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Model config hash: `sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130`
- Resident config hash: `sha256:ebe7e831e9a1f4bec5df3bb783359cafd0ddb316856e5f71df5e3b3d379f63db`
- Plan artifact: `sha256-8656b828a48e72f07799726b92c8a70b257f775c77f6c4bd5787686b524e8a9d`
- Candidate run: `evidence/m4/gemma-pageable-v28-four-block-canary`

## Requirement audit

| Requirement | Authoritative evidence | Result |
| --- | --- | --- |
| Real pinned model and calibration | The resolved Experiment 001 input is the pinned local snapshot and a run-local calibration tensor of 256 by 2,048 tokens generated with that snapshot's tokenizer. `evidence/m3` retains historical calibration statistics/objective artifacts. | Proven |
| Calibration behavior | M3 calibration parity, batch-partition invariance, and cached/uncached equivalence tests pass. The complete run freshly validates one calibration-statistics and one calibration-tensors artifact. | Proven |
| Allocation and BPW | All 182 ranks match contemporary legacy, with rank sum 105,856 and zero mismatches. Effective BPW is 0.9963181446312268 over 697,761,792 quantized parameters. | Proven |
| Factorization and scales | `evidence/m2/gemma-admm-factorization-parity.json` proves exact transposed legacy replay for latent/binary factors, scales, reconstruction, objective, and RNG state. The production native orientation is accepted by the complete composed trajectory. | Proven |
| Layer/block tuning | All 26 post-refit block boundaries are retained; mean absolute legacy delta is 0.7507% and maximum is 4.2188%, with 16 of 26 rewrite boundaries lower. The evaluation policy promoted the result. | Proven |
| Model-level KD | Eight epochs and 2,048 steps selected 885 parameters. Final objective is 2.148416 versus contemporary legacy 2.1430, a +0.2527% delta. | Proven |
| Held-out quality | Exact serial WikiText-2 uses 64 by 128 tokens and 8,128 scored targets with a pinned token hash. Tuned PPL is 453.570986 versus legacy 444.332773, +2.0791%, inside the accepted environment-matched band. | Proven |
| Bounded memory | Peak CUDA reservation is 7,631,536,128 bytes; peak host working set is 12,786,270,208 bytes. WDDM shared memory peaks at 622,854,144 bytes and returns to 83,886,080 bytes after all 26 block releases. | Proven |
| Resume and artifact integrity | The resumed run has one identity, a contiguous 208-record journal, 26 block commits, 182 layer commits, and 979 freshly hash-validated reachable artifacts totaling 11,742,170,282 bytes. | Proven |
| GGUF compatibility | The 699,863,936-byte GGUF has SHA-256 `4b3131f65f3c7d73afdb2c5809f87b860356418dae6c78873a3b2e95aa2daad3`. Fresh inspection matches all 1,274 NanoQuant tensors and 22,719,854 elements exactly and retains 158 model-shell tensors. | Proven |
| Runtime behavior | The full campaign records exact long-context output, zero unexpected fallbacks, 1,592,178,176 peak device bytes, and 160.74 tokens/s versus llama.cpp 184.5 tokens/s (87.12%). The accepted 32-token output and llama.cpp prefix are exact. | Proven |
| Ordered promotion | `evidence/m8/gemma-pageable-v28-evaluation-campaign-v2/results/campaign.json` records promotion at replay, quick, standard, and full tiers and final outcome `full-promotion`. | Proven |

## Fresh completion validation

The following commands were rerun on 2026-07-15 rather than trusting validation caches:

```powershell
.\.venv\Scripts\python.exe tools\validate_resident_run.py `
  --run-output evidence\m4\gemma-pageable-v28-four-block-canary `
  --expected-blocks 26 --require-complete

.\.venv\Scripts\python.exe tools\validate_llamacpp_gguf.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --gguf evidence\m6\gemma-pageable-v28-nanoquant.gguf `
  --reference-root D:\dev\research\llama.cpp
```

The resident validator re-hashed all 979 reachable artifacts and reported `complete: true`. The GGUF validator
reported `exact: true`, 182 NanoQuant layers, 1,274 NanoQuant arrays, and 22,719,854 matching elements.

The source tree also passes Ruff, mypy over all 150 source files, and 637 tests.

## Remaining work outside this parity decision

The implementation checklist deliberately remains open for large-model streaming/70B evidence, distributed
execution, final CLI/API breadth, scheduled performance CI, release packaging/security qualification, migrations,
and operational sign-off. These are real project tasks, but they do not contradict the measured 1B legacy-parity
result above.
