# Milestone 6 logical runtime evidence

## Pinned Gemma v28 frozen-to-logical export

The accepted 26-block `google/gemma-3-1b-it` resident run was exported from its atomically active globally tuned
state into the deployment-owned `nanoquant-v1` logical artifact. Export and validation were CPU-only and streamed
one block at a time; neither operation loaded the Hugging Face model or used CUDA.

The large reproducible artifact is retained locally at `gemma-pageable-v28-logical-runtime/` and remains ignored by
Git. `gemma-pageable-v28-logical-runtime-validation.json` is the committed compact evidence record. Validation
freshly hashed the source artifact graph and all output shards, then compared every logical specification and tensor
role with the selected source state.

```powershell
.\.venv\Scripts\python.exe tools\export_logical_runtime.py `
  --run-output evidence\m4\gemma-pageable-v28-four-block-canary `
  --output evidence\m6\gemma-pageable-v28-logical-runtime `
  --expected-blocks 26 `
  --source google/gemma-3-1b-it `
  --revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 `
  --family gemma3 `
  --config-hash sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130 `
  --tokenizer-hash sha256:b0f4e47731cc0550b068931303374ba73754d1086ab993d8ea528f7fffeb4611

.\.venv\Scripts\python.exe tools\validate_logical_runtime.py `
  --run-output evidence\m4\gemma-pageable-v28-four-block-canary `
  --artifact evidence\m6\gemma-pageable-v28-logical-runtime `
  --expected-blocks 26

.\.venv\Scripts\python.exe tools\validate_logical_reference_parity.py `
  --artifact evidence\m6\gemma-pageable-v28-logical-runtime `
  --absolute-tolerance 0.03125
```

This is evidence for exact frozen-to-logical conversion and bounded block sharding. It is not evidence that the
still-open CUDA-packed layout, complete model shell, or generation runtime is implemented.
