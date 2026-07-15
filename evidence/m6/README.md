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
single-turn mode. This proves M6.11 conversion compatibility; native rewrite execution is validated separately
below, while a clean runtime-only generation package remains open.

## Pinned Gemma v28 native CUDA packed execution

The version-1 `cuda-packed-triton` backend was validated directly against the complete packed artifact under the
cross-process CUDA lease. The validator hashes and inspects the packed descriptor/shards, runs the backend twice for
exact deterministic replay, compares every output with the F32 mathematical operation, records all real
shape/rank/outlier inventories, and measures incremental PyTorch CUDA allocation. Triton's compiler cache was placed
in a temporary directory outside the evidence tree.

```powershell
.\.venv\Scripts\python.exe tools\validate_cuda_packed_runtime.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --device cuda:0 --input-dtype bfloat16 --tokens 1 `
  --triton-cache $env:TEMP\nanoquant-triton-cache-m612 `
  --output evidence\m6\gemma-pageable-v28-cuda-packed-validation.json

.\.venv\Scripts\python.exe tools\validate_cuda_packed_runtime.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --device cuda:0 --input-dtype bfloat16 --tokens 4 `
  --triton-cache $env:TEMP\nanoquant-triton-cache-m612 `
  --output evidence\m6\gemma-pageable-v28-cuda-packed-prefill-validation.json
```

Both passes covered 26 blocks, 182 layers, all 18 real shape/rank combinations, BF16 scales, and the real salient
counts 2 and 7. Decode compared 459,264 outputs with maximum absolute error `1.9073486328125e-06` and 1,177,088
peak incremental allocated CUDA bytes. Four-token prefill compared 1,837,056 outputs with maximum absolute error
`3.814697265625e-06` and 1,370,112 peak incremental allocated CUDA bytes. Both deterministic replays were bit-exact.
Fixture coverage additionally exercises all declared input/scale/floating-salient dtypes, scaled-I8 salient values,
optional bias, tail words, and more than one salient tile. This completes M6.12 only; it does not establish
model-shell generation, KV-cache correctness, or performance parity. M6.13 subsequently added independently selected
prefill/decode plans that share identical prepared packed weights and validate workload geometry before dispatch.

The modified llama.cpp CUDA target was then rebuilt from the exact tracked dirty source in the Visual Studio x64
developer environment. Its `ggml-cuda.dll` therefore contains kernel SHA-256
`5c87336c2b6b8fb33805c6ee6a8752d4bd364beed63fd4cca03c2b36be966619`. The same fresh
`b9916-5c6ae7981` CUDA-enabled executable generated 16 tokens once with `-ngl 0` and once with `-ngl 99`:

```powershell
D:\dev\research\llama.cpp\build-local-cuda-ninja\bin\llama-cli.exe `
  -m evidence\m6\gemma-pageable-v28-nanoquant.gguf `
  -p "Write a short paragraph about quantization." `
  -n 16 --temp 0 --seed 1 -c 256 -b 64 -ub 64 `
  --single-turn --simple-io --no-display-prompt --no-warmup `
  -ngl 99
```

Both paths produced `Okay, here’s a draft of a short paragraph about quantum physics, and` exactly. CPU-forced
prompt/decode rates were 1.4/1.2 tokens/s; CUDA rates were 243.4/150.2 tokens/s. This short sample is a correctness
smoke, not the still-open stable reference benchmark. The separately configured CPU-only target compiled from the
same current source but aborts during model loading at `ggml-backend.cpp:1242`; it was not used for current-source
parity. `gemma-pageable-v28-llamacpp-cuda-smoke.json` records the exact binary and source hashes.

## Pinned Gemma v28 paired execution plans

The M6.13 validator prepares the complete packed artifact under one CUDA lease, plans prefill and decode separately,
and executes both regimes for every layer:

```powershell
.\.venv\Scripts\python.exe tools\validate_runtime_execution_plans.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --device cuda:0 --input-dtype bfloat16 `
  --batch-size 1 --prefill-tokens 4 `
  --triton-cache $env:TEMP\nanoquant-triton-cache-m613 `
  --output evidence\m6\gemma-pageable-v28-execution-plans-validation.json
```

Both plans selected `cuda-packed-triton` for all 182 layers with zero fallback. Every prefill dispatch shared the
same `PreparedLayer` object as its decode counterpart, so all packed weights occupied 87,087,616 incremental CUDA
bytes rather than being duplicated. The pass produced 1,837,056 prefill and 459,264 decode outputs across all 26
blocks; transient execution peaked only 342,528 bytes above prepared weight memory. This proves workload-plan
selection, shared preparation, geometry enforcement, and both dispatch paths. It does not prove attention metadata,
KV-cache behavior, sampling, or generation-engine correctness.
