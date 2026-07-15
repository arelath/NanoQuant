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

## Third promoted optimization: short-context sliding masks

Gemma3's eager decoder independently constructs a sliding-window mask in every sliding layer. Under this protocol,
the 48-token maximum cache is smaller than the 512-token window, so `tril(ones, diagonal=-512)` is entirely false,
`where` returns the existing causal mask, and the offset slice starts at zero. A pinned decoder-layer binding now
elides that identity transformation only while both mask dimensions and the last cache position fit inside the
window. It executes the original Transformers construction for longer contexts.

The full exact-kernel trace bound all 22 sliding layers, retained token 236764 in every pass with zero packed-linear
fallback, and reduced the census from 1,382 to 1,294 launches/token. The 586,020-byte record is
`evidence/m7/gemma-pageable-v28-rewrite-short-sliding-mask-profile.json`, SHA-256
`3c9ec4a7e960deba1728b5b5b70cf424ba9e05b9c478fe276d7fd5bbc10a2fce`.

Candidate/control/candidate single-token model medians were 73.23, 82.49, and 77.20 ms, so both candidates beat the
control and their average is 8.8% lower. Exact 32-token generation medians were 2.317, 2.429, and 2.452 s; WDDM noise
reversed the second pair, but the candidate average remains 1.8% lower. TTFT and complete-generation hashes, peak
allocation, all 182 CUDA linears, and zero fallback were unchanged. The records' SHA-256 values are respectively
`68b6cce4146c7f855f4f18c0bc7bba0e9ff53387090191ca5ba4b6b72ee7975f`,
`30f1ce1f4c06abaa503c58742561af8440122f1345359b3b120503d0f175851a`, and
`3c943db45ecf4189871b4308fce68d4dbff5140ee70d6959b867b73b999aaf7d`.

## Fourth promoted optimization: pre-rollover sliding-cache updates

Before the cache rolls over, Transformers still constructs identity rotation indices, gathers the complete local K/V
cache, writes the new token, zeroes the backing tensors, and adds the gathered copy back. The generation engine
already knows the position start and token count as host integers. It now passes that metadata to a prepared
HybridCache, which performs the identical indexed F16 prefix write without a device scalar read. When the next write
would reach rollover, the prepared cache calls the original Transformers implementation unchanged. A 32-token tiny
Gemma test crosses that boundary and matches standard HybridCache generation exactly.

With the already accepted short-mask path enabled, the pinned real-model profile preserves token 236764 in all event
and Kineto passes while reducing kernels from 1,382 to 964, ATen calls from 5,261 to 4,271, launch APIs from 1,379 to
961, and non-nested device kernel self time from 7.58 to 7.11 ms. Top-level synchronized decode p50 falls from 42.04
to 34.98 ms. The 577,512-byte record has SHA-256
`ffed6f6a52910cb40a2fda6842da476c54fa465958bdb5c6439346834008b752`.

The first candidate/control/candidate generation sequence is retained but not averaged because its first candidate
contains large WDDM stalls: 1.517, 1.281, and 0.975 s. A subsequent stable adjacent control/candidate pair measures
1.141 versus 1.000 s for 32 exact tokens (12.4% lower) and 37.80 versus 32.95 ms for isolated decode (12.8% lower).
The final candidate produces 32.01 tokens/s, **17.35%** of the 184.50 tokens/s llama.cpp reference and 3.27x the
frozen 9.79 tokens/s rewrite baseline. Exact hashes, allocation, 182 CUDA linears, and zero fallback are unchanged.

## Fifth promoted optimization: fused cache conversion, prefix update, and attention views

The prepared prefix update still issued separate F32-to-F16 conversions for K and V, separate indexed writes into
the F16 backing caches, and separate full-cache F16-to-F32 promotions for eager attention. A guarded Triton kernel
now performs all six logical operations in one launch per layer. It explicitly rounds new values to F16 before both
the backing write and F32 output, so the result matches the prior representation rather than bypassing its precision
boundary. Direct CUDA tests compare both backing tensors and both promoted views with `torch.equal` for one-token and
multi-token/multi-batch geometries. Only contiguous CUDA F32 states with F16 backing storage strictly before rollover
use the kernel; all other cases retain the existing PyTorch or Transformers path. A tiny CUDA generation regression
crosses sliding rollover and exactly matches the fused-off F16-cache control.

The pinned profile executes 26 fused cache kernels/token, preserves token 236764 in every pass, and retains all 182
packed linears with zero fallback. Kernels fall from 964 to 834, launch APIs from 961 to 831, and ATen calls from
4,271 to 3,595; non-nested device kernel self time is 6.82 ms and synchronized top-level wall p50 is 31.07 ms. The
576,004-byte record has SHA-256
`2a97c8677bbbcd22d038e695db2c157784be8e0432bb56c02b025a3931482811`.

Candidate/control/candidate isolated-decode medians are 28.77, 30.88, and 29.38 ms, a 5.8% candidate-average
reduction. Exact 32-token generation medians are 0.851, 0.914, and 0.893 s, a 4.6% candidate-average reduction. All
three runs preserve output hash `d91549bc797d2ff5a31e3b1e224347fac211fd34aa2077e01e521a888d24de3f`
and identical peak allocation. Their SHA-256 values are respectively
`14d30f2fd41b8beccd11aa1bd1f4856a7a865c13571fe323ab7fa97685fb2a5c`,
`4d0beab14f2f542016469b496d0fc6cd28b557f4c20ce56c9c4b6c0499705f15`, and
`2aad1333ccc7de1c8a3a1ac9249db945aff9471e63913247a80ca8eb2cbab5ff`. The final candidate reaches 35.82 tokens/s,
**19.42%** of the 184.50 tokens/s reference and 3.66x the frozen rewrite baseline. The specialization is default for
supported inputs; `--no-fused-cache-prefix` retains the matched control.

## Sixth promoted optimization: native BF16 tied embedding/output table

The runtime bundle and converted GGUF both retain Gemma's shared 262144x1152 embedding/output tensor in its source
BF16 representation. The rewrite loader nevertheless expanded that tensor to F32, retaining roughly 1.21 GB and
reading it through a 2.95 ms full-vocabulary F32 GEMV on every token. The guarded CUDA/F32 runtime path now keeps one
shared BF16 parameter. A fused embedding kernel combines lookup, BF16-to-F32 promotion, and the existing F32 scale,
and is bit-identical to the prior embedding output. A mixed BF16-weight/F32-input output kernel accumulates in F32.
Unsupported devices or runtime input dtypes retain the original shell modules.

For the pinned prompt, comparing the mixed output kernel with the same BF16 table expanded to F32 measures maximum
absolute logit error 3.70e-6, RMSE 4.80e-7, and a 4.65 reference top-1 margin. Argmax, token 236764, and the complete
generation hash remain exact. Output-head kernel time falls from 2.945 to 1.481 ms, total non-nested device kernel
self time from 6.82 to 5.35 ms, and the census from 834 to 833 kernels/token. The 574,552-byte profile has SHA-256
`7642783948258de14823d38f7839c83c0ae68523246fa5082d6dad5a36e5e2d2`.

Candidate/control/candidate isolated-decode medians are 28.60, 36.92, and 29.18 ms; complete-generation medians are
0.999, 1.045, and 1.010 s. The candidate average is 21.7% lower for isolated decode and 3.8% lower end to end. All
runs preserve hash `d91549bc797d2ff5a31e3b1e224347fac211fd34aa2077e01e521a888d24de3f`.
Candidate/control/candidate SHA-256 values are
`ba32cb59c42f231b503ff35b81ee5ea1eee4d3839d5016d2acd2dff9c6d50686`,
`0cf739d601001a2ec21e497785d2674cee1f7914209ee7335273aeface8be99b`, and
`53f2f4f61b1c90b7b36110b37146146e6b5621057168f004f06e21e53e30ebff`. Matched retained/peak allocation drops
from 1.215/1.218 GB to 0.653/0.655 GB. The production bundle validator also replays all 32 exact tokens with zero
fallback while cutting its prior 2.504 GB load/generation peak to 1.296 GB. The guarded specialization is default;
`--no-native-bfloat16-tied-projection` retains the control.

## Seventh promoted optimization: fixed short-context decode attention

After RoPE and cache update, eager attention still launched grouped-query score matmul, scaling, causal-mask add,
softmax, and value matmul independently in every layer. The promoted Triton kernel computes the pinned batch-one,
four-query/one-KV-head, width-256 F32 operation in one launch while physical cache length is at most 64. It reads the
existing F32 causal mask, returns the already transposed contiguous output, and does not expose attention weights.
Training, requested attention weights, softcapping, non-contiguous/other geometries, and longer caches execute the
unchanged eager path. Direct 16- and 48-position CUDA tests match eager within 2e-5.

All 26 real layers bind, token 236764 remains exact in every pass, and complete generation retains hash
`d91549bc797d2ff5a31e3b1e224347fac211fd34aa2077e01e521a888d24de3f`. Kernels fall from 833 to 729, launch APIs
from 830 to 726, and ATen calls from 3,581 to 2,411. The 26 fused attention kernels total 0.104 ms/token and total
non-nested device kernel self time falls from 5.35 to 5.27 ms. The 572,184-byte profile has SHA-256
`2fe3efedee59cb089de644c12cf6c8f37ca416ac09010e32a528f1fdeb817601`.

Candidate/control/candidate isolated-decode medians are 31.90, 38.78, and 33.11 ms, a 16.2% candidate-average
reduction. Complete-generation medians are 1.067, 1.198, and 1.229 s; their candidate average is 4.1% lower, although
the second candidate is 2.6% slower than control and is retained as measured WDDM variance rather than omitted.
Peak allocation and hashes are identical. Candidate/control/candidate SHA-256 values are
`c18b40b9536c2f4fc66ed75c8495296cf5c33a668d4ab1744311a70424fc87d0`,
`f980ad14bfb072c4112d66d33c71ae22ccfb77f6bdb740f801e8740105616a96`, and
`a03a287b17c746d9da1d743aba13efa42e852ea5bed6f475125f8b47817eaa84`. Production bundle validation binds all 26
fused attentions and exactly replays 32 tokens with zero packed fallback. The guarded specialization is default;
`--no-fused-decode-attention` retains the eager control.

## Eighth promoted optimization: grouped decode Q/K/V projections

Each pinned attention block previously invoked the NanoQuant first-stage and reconstruction/outlier kernels
separately for Q, K, and V. All 26 blocks share the same input width and salient-column count, use eight-aligned
ranks and output widths, and have no bias or outlier scales. The promoted binding concatenates immutable right
factors/scales, pads and concatenates left factors, and executes the three first stages in one Triton launch followed
by the three second stages in one launch. It is decode-only. Prefill and any unsupported device, dtype, shape, or
payload contract execute the original prepared linears independently.

Changing the program mapping changes F32 reduction order: the direct CUDA fixture observed maximum per-projection
errors of 3.05e-5, 1.19e-5, and 2.29e-5. A 5e-5 regression ceiling records that intentional numerical surface. The
real Gemma profile and all candidate/control/candidate runs nevertheless preserve complete hash
`d91549bc797d2ff5a31e3b1e224347fac211fd34aa2077e01e521a888d24de3f`. All 26 groups bind with zero packed fallback;
kernels fall from 729 to 625, launch APIs from 726 to 622, ATen calls from 2,411 to 2,255, and non-nested device
kernel self time from 5.269 to 4.887 ms. The 473,645-byte profile is
`evidence/m7/gemma-pageable-v28-rewrite-grouped-qkv-probe.json`, SHA-256
`5449e368da942af2dd4f88fa9453a846003fd4e187a02458d3533356340284e4`.

Candidate/control/candidate isolated-decode medians are 45.279, 49.984, and 43.754 ms; their candidate average is
10.9% lower. Complete 32-token medians are 1.220, 1.440, and 1.360 s; their candidate average is 10.4% lower. The
records' SHA-256 values are respectively
`6851c6512a613d862ee755f4c90472e554b6c4c6c730814687a4569fcfe3c063`,
`73fca99108da50551f0d8a2a62e8fa1ad143013a1a9fb3c385f03bac17824972`, and
`467a5044aaab21af6bb0d78c9af721746fb98d8694b368aaf65fc81175cc2d95`. The supported specialization is now the
CUDA/F32 default; `--no-group-decode-qkv` retains the separate-launch control.
