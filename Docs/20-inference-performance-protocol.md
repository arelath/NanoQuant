# Protocol-matched Gemma inference performance baseline

## Status

This document freezes Milestone 7's initial apples-to-apples artifact pair and workload. It separates correctness,
stage throughput, and process/model-load measurements so a faster number cannot silently substitute a different
prompt, cache precision, graph, or timing boundary.

The result is not performance parity. It proves that short-prompt prefill is already close to the modified llama.cpp
reference while single-token decode is the dominant remaining runtime gap.

## Artifact and implementation identities

| Item | Frozen identity |
| --- | --- |
| Model | `google/gemma-3-1b-it` at `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Rewrite packed descriptor | SHA-256 `b4f0c6270c4b59f8293c909ddeb21042ad1a2d7ee18601c77e4c57563c900487` |
| Rewrite runtime-bundle descriptor | SHA-256 `e5cef7236e2846a49129e991f8fc5efb660a9a8b8c71d9531590bed71739cc42` |
| Reference GGUF | 699,863,936 bytes; SHA-256 `4b3131f65f3c7d73afdb2c5809f87b860356418dae6c78873a3b2e95aa2daad3` |
| llama.cpp source | commit `5c6ae79816ee0f2b3d4bb8ec9061c294185d320b`, dirty diff object `cf463b9266db4e1f162ad8970e8ddcc1abfb5fbd` |
| `nanoquant.cu` | SHA-256 `5c87336c2b6b8fb33805c6ee6a8752d4bd364beed63fd4cca03c2b36be966619` |
| `llama-bench.exe` | SHA-256 `05c5ee7669a5c20ff5dea3737c153208675c288b36eb3b7b60e9c6f99284a496` |
| `llama-cli.exe` | SHA-256 `c091ac0bfe757eb1b6e534924d23c0de4b4f31d38841ae709d33ba7741499a4c` |
| Device | NVIDIA RTX 4000 Ada Generation Laptop GPU, 12,878,086,144 bytes |

The GGUF was exported from the same rewrite packed artifact and exactly validated across all 1,274 NanoQuant
tensors. Artifact quality and BPW therefore do not vary between the two runtime measurements.

## Shared workload

| Axis | Value |
| --- | --- |
| User prompt | `Write a short paragraph about quantization.` |
| Prompt protocol | Gemma instruction chat template |
| Prompt tokens | 16 |
| Generated tokens | 32, forced length for the rewrite; llama produced all 32 |
| Batch | 1 |
| Sampling | greedy / temperature 0; llama seed 1 |
| Context/cache bound | 64 in llama CLI; 48 physical slots in the rewrite |
| KV storage | F16 K and V |
| Model/linear boundary | F32 shell and F32 NanoQuant operation output |
| Attention | eager rewrite; llama flash attention disabled |
| llama batch / microbatch | 64 / 64 |
| CPU threads | 24 |
| GPU offload | all layers (`-ngl 99`) |
| Warm-up | enabled; rewrite uses three explicit warm-ups |
| Samples | 10 measured repetitions |

Transformers eager attention cannot directly multiply F32 queries by F16 cached keys. The rewrite's explicit
`PromotingHybridCache` stores K/V in F16, as llama.cpp does, and promotes the returned attention view to the query
dtype. This preserves the same 32 generated tokens as the accepted F32-cache correctness run. The promotion is
measured overhead, not an unreported optimization.

## Timing boundaries

`llama-bench` supplies device-resident prefill and steady decode samples without process/model-load time. It uses
synthetic token values at the same 16/32 geometry, which is valid for shape-level throughput but not output parity.
`llama-cli` separately runs the exact prompt and generation protocol ten times; its prompt/generation rates exclude
model load, while process wall is retained separately. The generated text is identical across repetitions and starts
with the accepted correctness reference.

The rewrite benchmark measures:

- model prefill with a fresh bounded cache;
- one decode forward after untimed cache prefill;
- synchronized time to first token;
- synchronized 32-token end-to-end generation including greedy selection and explicit stopping checks;
- kernel, prepared-layer, and transformer-block subscopes in separate passes.

All preparation, packed transfers, model binding, source-weight release, and allocator cleanup occur before these
timed regions. Both rewrite plans select `cuda-packed-triton` for all 182 linears with zero fallback.

## Baseline results

| Scope | Rewrite median | llama.cpp median | Rewrite/reference |
| --- | ---: | ---: | ---: |
| Exact-prompt prefill | 139.31 tokens/s | 144.20 tokens/s (`llama-cli`) | 96.61% |
| Synthetic 16-token prefill | 139.31 tokens/s | 513.26 tokens/s (`llama-bench`) | 27.14% |
| Single-token / steady decode | 10.21 tokens/s | 184.50 tokens/s (`llama-cli`) | 5.53% |
| Single-token / synthetic decode | 10.21 tokens/s | 216.15 tokens/s (`llama-bench`) | 4.72% |
| Rewrite time to first token | 113.06 ms | 110.96 ms prompt time (`llama-cli`) | 101.89% latency |
| Rewrite 32-token end to end | 3.268 s / 9.79 tokens/s | 173.44 ms decode stage / 184.50 tokens/s | different aggregate boundary |

The exact-prompt comparison is the promotion/gating result. The synthetic llama-bench numbers remain useful upper
bounds and explain why older isolated reference reports were higher than CLI generation. End-to-end wall values are
not divided directly because llama's printed decode stage excludes prompt and model load while the rewrite aggregate
includes prompt, sampling, stopping, and terminal synchronization.

## Evidence

- `evidence/m7/gemma-pageable-v28-rewrite-f16kv-benchmark.json`
  - 16,583 bytes; SHA-256 `fe96fe5129963aedfc082c70614757d6d61361f650481e94a59d103e3d743acc`.
- `evidence/m7/gemma-pageable-v28-llamacpp-benchmark.json`
  - 7,005 bytes; SHA-256 `0eb01eabb502efa7b6f39be18e704388c22dfff09ce25beeea27f5d23744b73e`.
- `evidence/m7/gemma-pageable-v28-llamacpp-generation-benchmark.json`
  - exact prompt and text, ten process repetitions; retained hash is recorded in `evidence/m7/README.md`.

## Promotion rule

This pair remains the M7 baseline until a deliberate protocol revision records new identities for both sides. An
optimization is promotable only when it preserves the accepted generation output/quality gates, has zero unexpected
fallback, and improves a repeated matched scope. The initial target is at least 70% of reference steady decode;
5.53% is a measured failure that requires profiling and implementation work, not an accepted exception.

## Decode profile and accounting

The M7.3 profile uses the same bundle, prompt, F32 shell, F16 cache storage, eager attention, batch, and cache bound.
It records ten samples in each of three independent CUDA-event passes:

- a sparse model/top-level pass (model, embedding/other shell work, 26 blocks, final norm, and head);
- a block/component pass (blocks plus attention, MLP, and the four Gemma norms);
- a block/prepared-linear pass (blocks plus all 182 NanoQuant linears).

Separating the passes prevents the 182 linear event pairs from inflating the attention/MLP measurements that contain
them. The sparse top-level pass accounts for **97.77% p50** of synchronized wall time, with a p10–p90 range of
**97.56–97.82%**. Model CUDA time accounts for **99.85% p50** of wall. This exceeds the 90% M7.3 gate without
double-counting nested regions.

The profile is diagnostic rather than a replacement throughput benchmark: its sparse pass has a 113.59 ms median
versus the uninstrumented 97.96 ms decode baseline because even CUDA event recording perturbs this launch-heavy path.
Within independent passes, eager self-attention is about 51% of profiled model time, MLP is about 13%, and all 182
prepared linears are about 29%. The four per-block norms collectively account for roughly 21%. These shares establish
two facts relevant to the next work:

1. Python/Transformers attention-cache and small-operation launch structure is material, including the required F16
   cache-to-F32 attention view promotion.
2. The packed linears alone take about 32.8 ms under linear instrumentation, already far above llama.cpp's complete
   approximately 5.4 ms/token decode budget; dispatch cleanup alone cannot close the gap, and the two Triton kernels
   require decode-specific work.

The profile selected all 182 CUDA linears with zero fallback, returned the same second generated token in all 30
measured passes, peaked at 1,310,700,032 allocated CUDA bytes, and added only 5,312,000 allocated bytes over the
post-warm-up baseline. The retained 549,537-byte record has SHA-256
`086cbdc871edc3f8f3baa8598239989e07e86100f82985f3f87d11514f4746d4`.

### Kernel launch census

A fourth, separately traced decode preserves the CUDA-event passes above and uses Kineto only to count and identify
one warmed token's CPU operators and CUDA kernels. It records **2,558 CUDA kernels**: 182 `_nanoquant_stage1` plus
182 `_nanoquant_stage2` kernels, and **2,194 non-NanoQuant kernels**. The NanoQuant stages account for 5.17 ms of
12.92 ms non-nested device kernel time (40.0%), while the eager shell accounts for 85.8% of launches. The CPU trace
also records 2,555 launch API calls and 6,985 ATen calls. Trace wall time is not a benchmark because Kineto adds
substantial overhead; call counts and non-nested device kernel times are the selection evidence.

The 591,740-byte record is
`evidence/m7/gemma-pageable-v28-rewrite-f16kv-kernel-profile.json`, SHA-256
`697c6eea7916198585c1d8a13e17389d865cff3b8be04f916474477a73496094`.

## First promoted optimization: fused Gemma3 RMSNorm

The trace showed 157 RMSNorm invocations per token. The pinned Transformers expression expands each into seven CUDA
kernels; PyTorch's native weighted `rms_norm` produces one kernel and was bit-exact for both real F32 dimensions.
The runtime now binds immutable `1 + weight` scales after shell loading and uses the native operation only for F32;
F16/BF16 retain the original cast, reduction, multiply, and cast-back expression.

The candidate trace replaced all 157 norms, reduced total kernels from 2,558 to 1,616, reduced non-NanoQuant kernels
from 2,194 to 1,252, and reduced non-nested device kernel time from 12.92 to 9.40 ms. Its output token, all 30 event
pass tokens, 182-linears/zero-fallback dispatch, and allocation behavior matched the control.
The 590,033-byte candidate profile has SHA-256
`7af492fb5649c94d6d0e15abad254715c687c469078ce739b9a28850b45562a5`.

Candidate/control/candidate generation then retained the exact same TTFT and 32-token output hashes. Candidate
32-token medians were 2.467 s and 2.440 s; the frozen control baseline is 3.268 s. Averaging the two candidates gives
a **24.9% latency reduction** and **33.2% throughput gain**. The paired candidate decode medians average 81.63 ms
versus the frozen 97.96 ms baseline. Current end-to-end throughput is about 13.04 tokens/s, still only **7.07%** of
the 184.50 tokens/s llama.cpp reference, so M7 remains decisively open.

## Static compilation feasibility

Two bounded `torch.compile(fullgraph=False, dynamic=False, mode="default")` probes used the same accepted fused-norm
runtime and exact second-token check. Compiling prefill and decode together produced 58 graphs and 47 graph breaks at
the packed dispatch workload `ContextVar`. Compiling only decode after eager prefill removed the prefill/decode shape
transition but still produced 49 graphs and 40 ContextVar breaks, plus layer-index/cache specialization. Its eager
median was 41.95 ms while the stabilized last-five compiled median was 134.22 ms (3.20x slower). Both returned token
236764 exactly. The direct compiled paths are therefore rejected; M7.15 requires a traceable fixed-workload packed
operation and stable cache specialization before a full promotion run.
The 2,501-byte diagnostic has SHA-256
`1e532e17d88f69d5edc64539d546137d977c249d4f71c9baef3f92a16d8db7f7`.

## Rejected broad compiled decode

A bounded M7.15 probe compiled only the fixed-shape one-token model forward with
`torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)`; equivalent cache prefill remained eager.
The exact second token and zero-fallback dispatch were preserved, but the current runtime boundary is structurally
hostile to broad Dynamo capture: it produced 49 graphs, 40 breaks at the workload `ContextVar`, per-layer
specialization/recompile-limit failures, and CUDA-graph skips for mutating HybridCache inputs.

Compilation took 32.98 s on the first call. Ten warmed samples measured 890.47 ms p50 compiled versus 80.85 ms eager,
an 11.0x regression. This path was rejected without production changes. The 2,050-byte record is
`evidence/m7/gemma-pageable-v28-rewrite-compiled-fixed-decode-probe.json`, SHA-256
`d6f40afff3c5ea757853a9ab39461e61b4c6fa4a74febd745ba295dc3ab437db`.

## Second promoted optimization: decode-only fused RoPE

After fused RMSNorm, the eager Q/K rotary helper still issued ten kernels per block: two negations, two concatenation
copies, four multiplies, and two adds. The accepted specialization handles only the pinned batch-one, F32,
one-token, four-query-head/one-key-head, 256-dimension geometry. One Triton program computes Q and K together;
prefill and any other dtype or geometry call the original Transformers helper unchanged.

The first prototype exposed CUDA FMA contraction differences of at most 4.77e-7. The promoted kernel instead emits
explicit `mul.rn.f32` followed by `add.rn.f32`, making both Q and K bit-identical to the eager expression. The direct
real-shape probe measured 43.01 microseconds p50 versus 156.69 microseconds eager. The complete profile bound all 26
attentions, reduced total launches from 1,616 to 1,382 and non-NanoQuant launches from 1,252 to 1,018, preserved token
236764 across every event and Kineto pass, and reduced non-nested CUDA kernel self time from 9.40 to 7.58 ms. The
588,150-byte exact-kernel profile has SHA-256
`3a327819399fb30c67a8c0fd0d2b179af9962d81a8a67a973f057ac022c6853c`.

WDDM interference was large enough that the full alternating sequence must not be averaged as a clean speed ratio:
initial candidate/control/candidate 32-token medians were 2.467, 2.719, and 1.281 s. Both candidates beat that control,
and a subsequent adjacent control/candidate pair under the faster device state measured 1.391 versus 1.208 s, a
13.1% latency reduction. All TTFT and generation hashes, peak allocation, 182 packed-linears/zero-fallback dispatch,
and cache bounds matched. The accepted path is about 26.48 tokens/s in the final retained candidate, still only
**14.35%** of the 184.50 tokens/s llama.cpp result. The remaining mask/cache, attention, vocabulary projection, and
host-launch gap remains the dominant M7 work.
