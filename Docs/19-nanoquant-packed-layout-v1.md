# NanoQuant packed layout v1

## Status and scope

The first rewrite packed layout is `llama.cpp-i32-lsb-v1`. It is the canonical, portable sign-word layout consumed
by the modified llama.cpp NanoQuant CPU and CUDA operations. It is not the legacy extension's optional `half2`
packing. A Gemma3 adapter now emits an exactly validated complete GGUF from it; the rewrite packed artifact itself
remains independent of the architecture-specific model shell.

This document freezes sign encoding and padding; tensor shapes, dtypes, roles, and names; alignment requirements;
scale, salient-outlier, and bias semantics; the rewrite packed descriptor and block-shard contract; and the verified
adapter boundary used to emit the model-family-correct pinned Gemma GGUF.

## Inspected reference identity

The mapping was checked against `D:\dev\research\llama.cpp` in its intended dirty NanoQuant state, not against
upstream llama.cpp alone:

| Item | Identity |
| --- | --- |
| Git HEAD | `5c6ae79816ee0f2b3d4bb8ec9061c294185d320b` |
| Binary dirty-diff Git object | `cf463b9266db4e1f162ad8970e8ddcc1abfb5fbd` |
| `ggml/src/ggml-cuda/nanoquant.cu` SHA-256 | `5c87336c2b6b8fb33805c6ee6a8752d4bd364beed63fd4cca03c2b36be966619` |
| `convert_nanoquant_to_gguf.py` SHA-256 | `92b0d31c1ce83d0fe3668bbb20cee6a4da24ec3e9476f6699890d01540241e4d` |
| `docs/development/nanoquant.md` SHA-256 | `12c46863a480a04b1ba449bb7bcb2f637419b677678b60f93522c531bd3f9ac8` |
| `src/llama-model.cpp` SHA-256 | `11175fca67ecd8b97f6d6ffa7d2e8b848839d768669d63f7ec629a69d8d704aa` |
| `ggml/src/ggml-cpu/ops.cpp` SHA-256 | `f51195610b4c533e4f606f984c7083bb542e4c3b3c8e740fdc647f8a5b0eff1c` |

The dirty-diff hash identifies the whole tracked patch at inspection time. The individual hashes make the relevant
converter, loader, CPU reference, and CUDA kernel independently auditable. Packed descriptor schema 1 embeds this
complete provenance record under `layout.reference`; a reader rejects any different value for this layout version.

## Mathematical contract

For input `x[..., in]`, right factor `V[rank, in]`, and left factor `U[out, rank]`, the packed operation is:

```text
pre_x[..., i] = x[..., i] * scale_pre[i]
latent[..., r] = sum_i(pre_x[..., i] * V[r, i]) * scale_mid[r]
base[..., o] = sum_r(latent[..., r] * U[o, r]) * scale_post[o]
salient[..., o] = sum_s(x[..., salient_idx[s]] * salient_weight[o, s]
                         * optional_salient_scale[s])
y = base + salient + optional_bias
```

`U` and `V` contain only `-1` and `+1`. Accumulation details belong to a backend capability/version; they do not
change the stored sign-word meaning.

The rewrite logical reference masks `scale_pre` at salient indices. llama.cpp does not perform that mask inside its
operation, so this packed layout requires those stored `scale_pre` entries to be exactly zero. Conversion rejects a
layer that violates this rule instead of silently changing it. All 182 quantized layers in the accepted pinned Gemma
v28 logical artifact satisfy the invariant.

## Logical, rewrite-packed, and GGUF mapping

`<layer>` is the canonical rewrite layer name. `<base>` is the model-adapter-mapped GGUF weight base name after
removing `.weight`. PyTorch shapes are row-major. GGML/GGUF displays dimension 0 first, so a PyTorch `[rows, cols]`
array is described as GGML `[cols, rows]`.

| Logical role | Rewrite packed role and PyTorch shape | GGUF tensor and GGML shape | Allowed dtype |
| --- | --- | --- | --- |
| `factor_right` / `V` | `factor_right_words`, `[rank, ceil(in/32)]` | `<base>.nq_v`, `[ceil(in/32), rank]` | `I32` |
| `factor_left` / `U` | `factor_left_words`, `[out, ceil(rank/32)]` | `<base>.nq_u`, `[ceil(rank/32), out]` | `I32` |
| `scale_pre` | `scale_pre`, `[in]` | `<base>.nq_scale_pre`, `[in]` | `F16`, `BF16`, or `F32` |
| `scale_mid` | `scale_mid`, `[rank]` | `<base>.nq_scale_mid`, `[rank]` | `F16`, `BF16`, or `F32` |
| `scale_post` | `scale_post`, `[out]` | `<base>.nq_scale_post`, `[out]` | `F16`, `BF16`, or `F32` |
| `outlier_indices` | `outlier_indices`, `[k]` | `<base>.nq_salient_idx`, `[k]` | `I32` |
| `outlier_values` | `outlier_values`, `[out, k]` | `<base>.nq_salient_weight`, `[k, out]` | `F16`, `BF16`, `F32`, or `I8` |
| `outlier_scales` | `outlier_scales`, `[k]` | `<base>.nq_salient_scale`, `[k]` | `F16`, `BF16`, or `F32` |
| `bias` | `bias`, `[out]` | normal model tensor `<base>.bias`, not an `nq_*` sidecar | layout scale dtype |

Packed safetensors keys are namespaced as:

```text
layouts.llama.cpp-i32-lsb-v1.<layer>.<packed-role>
```

The descriptor records each complete key, shape, dtype, role, and layer specification. Packed tensors therefore do
not masquerade as backend-independent logical tensors.

GGUF base-name mapping is an adapter responsibility. In particular, the modified converter may undo the attention
Q/K output-row permutation. It must apply the same row order to packed `U`, `scale_post`, and salient weights. The
rewrite packed artifact remains in the source model's canonical row order until that adapter transform is performed.
The pinned `Gemma3ForCausalLM` converter declares `undo_permute = False`; validation rejects any model class that
requests the currently unimplemented row transform rather than silently emitting misordered factors.

## Sign words, padding, and alignment

Each row is divided into groups of 32 logical signs:

- word index is `column // 32`;
- bit index is `column % 32`;
- bit zero is the least-significant bit;
- cleared bit means `+1`;
- set bit means `-1`;
- unused tail bits are cleared and therefore mean `+1`;
- serialized words are signed `int32`; kernels interpret the same bits as `uint32`.

No extra row padding is added after `ceil(columns / 32)` words. Original dimensions and rank come from the layer
specification rather than from padding bits.

The minimum word alignment is four bytes. The modified CUDA kernel optionally loads four words with `uint4` only
when both the tensor base and row stride are 16-byte aligned. A contiguous tensor naturally has row stride
`ceil(columns / 32) * 4` bytes. The runtime exposes the same predicate; lack of 16-byte row alignment selects the
scalar-word path and is valid, not a format error.

## Salient outliers and bias

- Indices and values are absent together or present together.
- Indices are `int32`, strictly increasing in the rewrite format, in range, and have shape `[k]`.
- Floating salient weights have no salient scale tensor.
- `I8` salient weights require exactly one scale per salient column; a scale tensor is invalid for floating weights.
- The salient term is added after the post-scaled binary-factor product.
- Bias is not part of the llama.cpp NanoQuant sidecar group. The rewrite keeps it in the packed layer shard so its
  runtime math is self-contained; a GGUF exporter must map it to the architecture's ordinary bias tensor and the
  model graph must add it exactly once.

Pinned Gemma v28 uses floating salient values without salient scales and has no biases in its 182 target linears, so
the real artifact does not by itself cover the `I8` salient-scale or bias branches. Fixture tests retain those cases.

## Packed artifact schema 1

`nanoquant-packed-model.json` declares:

- artifact format `nanoquant-packed-model`;
- packed descriptor schema `1`;
- layout metadata and version `llama.cpp-i32-lsb-v1`;
- source model/config/tokenizer identity;
- SHA-256 of the complete logical source descriptor;
- contiguous block entries and exact layer/tensor inventories;
- shard paths, byte sizes, and SHA-256 hashes;
- total layer and serialized weight bytes.

Weights use one safetensors shard per source transformer block. Conversion validates the complete logical source,
packs and writes one block at a time, clears the block state after writing, commits atomically, and refuses to
overwrite an existing output. Inspection verifies descriptor bounds, paths, file sizes, hashes, tensor inventories,
shapes, and dtypes without loading every payload. Loading one layer opens only its containing block shard.

## Modified llama.cpp checkpoint bridge

The Gemma3 adapter exports one safetensors checkpoint shard per transformer block. Canonical rewrite names such as
`blocks.12.self_attn.q_proj` become converter names such as `model.layers.12.self_attn.q_proj`. Each group contains
packed `U` and `V`, explicit `U_shape` and `V_shape`, the three scales, and optional salient tensors. Its descriptor
binds the checkpoint to the exact packed descriptor hash and embeds the pinned modified llama.cpp provenance.
Export is atomic, streams one block at a time, refuses overwrite, and rejects unsupported families and layer bias.

Explicit factor shapes also work around a defect in the pinned reference converter without changing that reference:
when both factors are packed and `U_shape` is absent, its `U_packed` branch reads `scale_mid` before assigning that
local. The bridge always supplies authoritative shapes, so the faulty inference branch is never entered. The
converter normalizes scales to F32 and floating salient values to F16. On Gemma, the latter intentionally changed
512 source BF16 values with maximum absolute difference `2.9802322387695312e-08`; all converter and GGUF values were
exact after that declared normalization.

## Verification and compatibility boundary

Unit coverage includes bit-boundary widths, exact LSB ordering, tail padding, aligned and unaligned row strides,
all supported factor dtypes, bias, floating and scaled-I8 salient paths, incompatible-state rejection, exact tensor
round trips, reference execution, descriptor corruption, future schemas, lazy loading, and overwrite rejection.

The checkpoint bridge maps the packed artifact into the pinned converter's legacy-compatible Hugging Face sidecar
names, one shard per transformer block. On the accepted Gemma artifact, the exact pinned converter accepted all 182
groups and emitted a 699,863,936-byte GGUF whose 1,274 NanoQuant tensors and 22,719,854 normalized elements matched
the packed source exactly. The GGUF also contained 158 ordinary model-shell tensors and loaded through the pinned CPU
llama.cpp build. This completes conversion compatibility (M6.11); it is not a CUDA backend in the rewrite, a
runtime-owned model shell/tokenizer package, or clean runtime-only generation proof. Those remain M6.12-M6.22 work.
