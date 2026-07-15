# Milestone 6 runtime artifact evidence

## Pinned Gemma v28 frozen-to-logical export

The accepted 26-block `google/gemma-3-1b-it` resident run was exported from its atomically active globally tuned
state into the deployment-owned `nanoquant-v1` logical artifact. Export and validation were CPU-only and streamed
one block at a time; neither operation loaded the Hugging Face model or used CUDA.

The large reproducible artifact was generated at `gemma-pageable-v28-logical-runtime/` and remained ignored by Git.
After packed conversion and downstream parity validation, it was removed with
`tools/cleanup_logical_artifact.py --apply` guarded by its exact descriptor SHA-256. The commands below reproduce it.
`gemma-pageable-v28-logical-runtime-validation.json` is the committed compact evidence record. Validation freshly
hashed the source artifact graph and all output shards, then compared every logical specification and tensor role
with the selected source state.

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

This is evidence for exact frozen-to-logical conversion and bounded block sharding.

## Pinned Gemma v28 logical-to-packed conversion

The logical artifact was converted CPU-only to packed descriptor schema 1 and layout
`llama.cpp-i32-lsb-v1`. The converter validated the source, packed one transformer block at a time, and wrote 26
atomic safetensors shards. The large reproducible packed artifact is ignored at
`gemma-pageable-v28-packed-runtime/`; `gemma-pageable-v28-packed-runtime-validation.json` is the compact evidence.

```powershell
.\.venv\Scripts\python.exe tools\convert_logical_to_packed.py `
  --logical-artifact evidence\m6\gemma-pageable-v28-logical-runtime `
  --output evidence\m6\gemma-pageable-v28-packed-runtime

.\.venv\Scripts\python.exe tools\validate_packed_runtime.py `
  --logical-artifact evidence\m6\gemma-pageable-v28-logical-runtime `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --absolute-tolerance 0
```

All 1,274 tensors across 182 layers reconstructed exactly. The logical factorized and unpack-once packed reference
backends matched exactly across 459,264 output elements. Packed shard bytes are 87,072,592, or 3.2764% of the
logical shard bytes. The descriptor embeds the exact modified llama.cpp commit, tracked dirty-diff object, converter,
loader, CPU operation, documentation, and CUDA-kernel hashes recorded in
`Docs/19-nanoquant-packed-layout-v1.md`. This proves the first packed format and offline conversion; it is not
by itself evidence for model-family GGUF conversion parity, the non-quantized model shell, native CUDA backend,
tokenizer/config package, or generation. The independent checks below cover the Gemma GGUF conversion boundary.

## Pinned Gemma v28 modified llama.cpp/GGUF compatibility

The packed artifact was streamed into 26 legacy-compatible checkpoint shards using canonical
`model.layers.<block>.<path>` prefixes and the exact `U_packed`, `V_packed`, shape, scale, and salient fields consumed
by the pinned modified llama.cpp converter. The generated checkpoint and 699,863,936-byte GGUF remain ignored;
`gemma-pageable-v28-llamacpp-validation.json` is the committed compact record.

```powershell
.\.venv\Scripts\python.exe tools\export_llamacpp_checkpoint.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --output evidence\m6\gemma-pageable-v28-llamacpp-checkpoint

.\.venv\Scripts\python.exe tools\validate_llamacpp_checkpoint.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --checkpoint evidence\m6\gemma-pageable-v28-llamacpp-checkpoint `
  --reference-root D:\dev\research\llama.cpp

.\.venv\Scripts\python.exe tools\validate_llamacpp_converter.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --checkpoint evidence\m6\gemma-pageable-v28-llamacpp-checkpoint `
  --llama-root D:\dev\research\llama.cpp `
  --model C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752

.\.venv\Scripts\python.exe D:\dev\research\llama.cpp\convert_nanoquant_to_gguf.py `
  C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752 `
  --nanoquant-checkpoint evidence\m6\gemma-pageable-v28-llamacpp-checkpoint `
  --outfile evidence\m6\gemma-pageable-v28-nanoquant.gguf `
  --outtype bf16 --no-lazy

.\.venv\Scripts\python.exe tools\validate_llamacpp_gguf.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --gguf evidence\m6\gemma-pageable-v28-nanoquant.gguf `
  --reference-root D:\dev\research\llama.cpp

D:\dev\research\llama.cpp\build-nanoquant-cpu\bin\Release\llama-cli.exe `
  -m evidence\m6\gemma-pageable-v28-nanoquant.gguf `
  -p Hello -n 1 --temp 0 --seed 1 -ngl 0 `
  --single-turn --simple-io --no-display-prompt --no-warmup --no-perf
```

The pinned converter selected `Gemma3ForCausalLM`, accepted all 182 sidecar groups, and mapped 1,274 GGUF tensors.
All sign words and F32-normalized scales were exact. Its required BF16-to-F16 salient normalization changed 512
elements with a maximum absolute change of `2.9802322387695312e-08`; every final normalized value was exact. Direct
GGUF inspection then matched all 22,719,854 NanoQuant elements and found the expected 158 non-quantized model-shell
tensors. The pinned CPU llama.cpp build loaded the GGUF, generated `Okay` for prompt `Hello`, and exited cleanly in
single-turn mode. This proves M6.11 conversion compatibility, not the native rewrite backend or a clean runtime-only
generation package.
