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
single-turn mode. This proves M6.11 conversion compatibility; native rewrite execution and the clean runtime-only
generation package are validated separately below.

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

## Pinned Gemma v28 generation-engine validation

The M6.14--M6.16 validator loads the locally pinned Hugging Face Gemma shell, prepares both execution plans, replaces
the exact 182 decoder linears with parameter-free packed dispatch modules, releases the displaced source weights,
and performs allocator cleanup before entering the generation loop:

```powershell
$model = 'C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752'
$env:TRITON_CACHE_DIR = "$env:TEMP\nanoquant-triton-cache-m614"
.\.venv\Scripts\python.exe tools\validate_runtime_generation.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --model $model --device cuda:0 --input-dtype bfloat16 `
  --max-new-tokens 4 --wait-for-device-seconds 30 `
  --output evidence\m6\gemma-pageable-v28-generation-validation.json
```

Two prompts with token lengths 2 and 8 were left-padded to width 8 and generated through one prefill plus three
decode forwards. Both plans selected `cuda-packed-triton` for all 182 linears with zero fallback. A fresh bounded
`HybridCache` declared maximum length 12, exact replay produced identical token and stopping results, peak allocated
CUDA memory was 702,635,520 bytes, and retained allocation after the first and second passes differed by 512 bytes.
The compact record is `gemma-pageable-v28-generation-validation.json`. This proves prompt batching, positions,
attention metadata, inactive-row behavior at fixture level, bounded cache construction, prepared model-shell
binding, and deterministic real-model execution. It is not the configured-sampling, long-generation memory-growth,
performance, or clean-install gate.

### Seeded device sampling

The same validator exercises device-side configured sampling without changing the prepared model shell:

```powershell
.\.venv\Scripts\python.exe tools\validate_runtime_generation.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --model $model --device cuda:0 --input-dtype bfloat16 `
  --max-new-tokens 8 --sampling sample --temperature 0.8 `
  --top-k 64 --top-p 0.95 --seed 20260715 `
  --stopping-check-interval 4 --wait-for-device-seconds 30 `
  --output evidence\m6\gemma-pageable-v28-sampling-validation.json
```

Both full passes returned the same eight tokens for both prompts. Temperature scaling, top-k filtering, top-p
filtering, and multinomial selection remained on CUDA; the generator was seeded once before the loop. The run
recorded one stopping-check synchronization and one terminal metadata synchronization, all 182 linears used
`cuda-packed-triton` with zero fallback, and peak allocated memory was 727,145,472 bytes. The compact record is
`gemma-pageable-v28-sampling-validation.json`. Fixture tests independently prove top-k exclusion, top-p nucleus
collapse, seeded replay, invalid-config rejection, and delayed inactive-row stopping checks. This closes M6.17, not
the long-generation M6.20 gate.

## CUDA packed capability matrix

`tests/unit/test_runtime_cuda_matrix.py` exhaustively crosses every finite capability dimension declared by
`cuda-packed-triton`: F16/BF16/F32 input, F16/BF16/F32 source factors, F16/BF16/F32 scales, absent or
F16/BF16/F32/scaled-I8 salient values, bias absent/present, and prefill/decode. The fixed 35-input, 17-output,
rank-33 geometry deliberately exercises tails in both packed sign-word dimensions. The resulting 540 cases all
matched an independent F32 operation within `rtol=2e-5, atol=2e-4` and every repeated execution was bit-exact:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_runtime_cuda_matrix.py
# 1 passed in 24.43s
```

This finite capability matrix complements, rather than replaces, the two complete packed-artifact validators above.
Those cover all 182 layers, all 18 real Gemma shape/rank combinations, real salient counts 2/7, and both one-token
decode and four-token prefill. Together they close M6.19. They do not establish long-generation composed-model
parity or memory growth, which remains M6.20.

## Long generation, cache, and llama.cpp output parity

The short llama.cpp smoke and rewrite initially appeared to disagree because they used different shell/prompt
protocols. The retained llama CLI applies Gemma's instruction chat template and its NanoQuant op returns F32 into an
F32 shell. Raw-text BF16 Transformers execution is a useful diagnostic, but is not that parity protocol:

```powershell
.\.venv\Scripts\python.exe tools\validate_runtime_generation.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --model $model --device cuda:0 --input-dtype float32 `
  --prompt "Write a short paragraph about quantization." --chat-template `
  --max-new-tokens 128 --ignore-eos --stopping-check-interval 8 `
  --reference-output evidence\m6\gemma-pageable-v28-llamacpp-cuda-smoke.json `
  --wait-for-device-seconds 30 `
  --output evidence\m6\gemma-pageable-v28-long-f32-chat-generation-validation.json
```

The first 16 tokens exactly reproduce `Okay, here’s a draft of a short paragraph about quantum physics, and` from
both retained llama.cpp CPU and CUDA runs. All 128 tokens replayed exactly on the second rewrite pass, all 182 linears
used CUDA with zero fallback, maximum cache length was 144, and peak allocated CUDA memory was 1,313,887,232 bytes.
Retained allocation was 1,304,107,520 bytes after the first pass and 1,304,108,544 after the second, a 1,024-byte
delta rather than sequence-proportional growth. A separate 32-token unequal-prompt tiny Gemma test exactly matches
Transformers HybridCache generation and asserts fixed local/global cache shapes after sliding-window rollover.
Together these close M6.20.

## Layered packed-runtime benchmark baseline

The M6.21 command times kernel, prepared-layer, transformer-block, prefill, decode, time-to-first-token, and complete
generation scopes without mixing setup into the intended timed boundary. It retains every raw sample and emits
p10/p50/p90/p99 latency and throughput distributions, allocation peaks, fallback counts, deterministic output hashes,
and artifact/environment identity:

```powershell
$env:TRITON_CACHE_DIR = "$env:TEMP\nanoquant-triton-cache-m621"
.\.venv\Scripts\python.exe tools\benchmark_runtime.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --model $model --device cuda:0 --input-dtype float32 `
  --suite all --warmups 3 --repetitions 10 `
  --max-new-tokens 32 --stopping-check-interval 8 `
  --output evidence\m6\gemma-pageable-v28-runtime-benchmark.json
```

The retained run used the Gemma chat template, one 16-token prompt, forced-length greedy generation, and separate
block carrier passes so hook events did not contaminate full-model samples. All 182 linears used
`cuda-packed-triton` with zero fallback. Selected median results were:

| Scope | Median latency | Median throughput |
| --- | ---: | ---: |
| Gate-projection prefill kernel, 16 slots | 0.572 ms | 27,954.92 slots/s |
| Gate-projection decode kernel | 0.141 ms | 7,233.80 slots/s |
| Gate-projection prepared decode layer | 0.188 ms | 5,311.39 slots/s |
| Decoder block 0, decode | 4.692 ms | 213.14 tokens/s |
| Full model prefill, 16 tokens | 113.78 ms | 140.63 tokens/s |
| Full model single-token decode | 96.44 ms | 10.37 tokens/s |
| Time to first token | 117.52 ms | 8.51 tokens/s |
| Complete 32-token generation | 2.922 s | 10.95 tokens/s |

The compact JSON is 16,560 bytes with SHA-256
`fb6f28b041dc225f48ed4641bb028dbf41746f81f791fb5e259029acd54f36d1`. This closes command and JSON coverage,
not performance parity: the composed model is dramatically slower than the retained short llama.cpp CUDA smoke,
so Milestone 7 must profile the gap and reconcile the stable comparison protocol before accepting any throughput
claim.

## Self-contained runtime bundle and isolated installation

The M6.22 exporter removes the external source-snapshot dependency from normal loading. Its exact-inventory bundle
contains the packed descriptor/shards, copied config/tokenizer assets, 158 ordinary source checkpoint tensors, and
three derived non-persistent Gemma buffers. All 182 dense source linear weights are excluded. The derived embedding
scale and global/local RoPE frequencies are first-class bundle tensors because meta construction plus `to_empty()`
would otherwise leave them uninitialized:

```powershell
.\.venv\Scripts\python.exe tools\export_runtime_bundle.py `
  --packed-artifact evidence\m6\gemma-pageable-v28-packed-runtime `
  --model $model `
  --output evidence\m6\gemma-pageable-v28-runtime-bundle

.\.venv\Scripts\python.exe tools\validate_runtime_bundle.py `
  --bundle evidence\m6\gemma-pageable-v28-runtime-bundle `
  --device cuda:0 --input-dtype float32 --max-new-tokens 32 `
  --reference-output evidence\m6\gemma-pageable-v28-llamacpp-cuda-smoke.json `
  --output evidence\m6\gemma-pageable-v28-runtime-bundle-validation.json
```

The bundle has 36 hashed members and 731,007,650 member bytes. Its 49,104-byte descriptor SHA-256 is
`e5cef7236e2846a49129e991f8fc5efb660a9a8b8c71d9531590bed71739cc42`. Loading prepared the complete CUDA backend,
materialized only the ordinary shell around it, restored tied weights and all derived buffers, and rejected any meta
or uninitialized state. The 32-token run used every packed linear with zero fallback, bound all 157 fused RMSNorms,
26 decode-only RoPE sites, and 22 short-context sliding layers, reproduced the retained llama.cpp 16-token prefix,
and replayed exactly. The native BF16 tied embedding/output specialization is active once, retaining 708,224,000
bytes after the second pass; the 1,295,585,792-byte load/generation peak is nearly half the prior 2,503,545,344-byte
F32-expanded-table peak. All 26 eligible short-context decode attentions and all 26 compatible grouped decode-Q/K/V
paths are active. The 2,199-byte validation record has SHA-256
`c30ec1a6f0dc3ec5e55a992ef37673f62a1c514718ec8594285d800f51b1193d`.

The separate deployment distribution and isolated install proof are reproducible with:

```powershell
.\.venv\Scripts\python.exe -m pip wheel --no-build-isolation --no-deps `
  packaging\runtime --wheel-dir .tmp\runtime-wheel

.\.venv\Scripts\python.exe tools\validate_runtime_only_install.py `
  --wheel evidence\m6\nanoquant_runtime-0.1.0-py3-none-any.whl `
  --bundle evidence\m6\gemma-pageable-v28-runtime-bundle `
  --device cuda:0 --max-new-tokens 16 `
  --reference-output evidence\m6\gemma-pageable-v28-llamacpp-cuda-smoke.json `
  --triton-cache .triton-cache --work-root .tmp `
  --output evidence\m6\gemma-pageable-v28-runtime-only-install-validation.json
```

The retained wheel is 62,052 bytes with SHA-256
`caf271a0a5b099e6509f4ec7ac049c788a0a186680eb25486c941e04e96e138d`. Its exact 23-member inventory contains only
`nanoquant/__init__.py`, `nanoquant/runtime/*`, and distribution metadata. The isolated child resolved that installed
copy, imported zero research modules, loaded the bundle without a source-model path, selected CUDA for all 182
linears with zero fallback, bound all 157 Gemma3 RMSNorms, all 26 eligible decode-only RoPE sites, all 22 guarded
short-context sliding layers, the native BF16 tied-table specialization, and all 26 fused short-context decode
attentions plus all 26 grouped decode-Q/K/V paths, executed 330 prepared sliding-prefix updates, and generated
exactly `Okay, here’s a draft of a short paragraph about quantum physics, and`. Retained allocation is 706,519,552
bytes and peak allocation is 1,295,585,792 bytes. The 2,977-byte validation record has SHA-256
`5aa7d4e5ea35bf6cfcc69ece4697b214ed2afa3933f6769d5c8e2c6126dc64c4`. Runtime dependencies came from the pinned
host environment because the validation was offline; M10.9 retains the broader clean dependency-install matrix.
Together with full reference/CUDA numerical coverage, this closes M6.22 and the M6 correctness gate, not the
still-open Milestone 7 throughput gate.
