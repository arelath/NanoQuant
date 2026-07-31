# Mixed-V packed overlay v1

## Decision

The accepted selective mixed-V experiment now has a deployment-runtime contract. It does not change packed layout
`llama.cpp-i32-lsb-v1`, packed artifact schema 1, or the modified llama.cpp/GGUF bridge. Instead, a
`nanoquant-mixed-v-overlay` is bound by SHA-256 to one existing packed descriptor and replaces only named layers.
Every other layer resolves lazily from that exact base artifact.

This separation is intentional:

- existing packed artifacts remain readable and byte-identical;
- GGUF export continues to consume only the established canonical layout;
- a mixed-V layer is expanded once to canonical right-factor words during backend preparation;
- the timed linear path and both packed CUDA kernels remain unchanged;
- rejecting or changing this experimental representation does not create ambiguity inside schema 1.

The current accepted selection policy remains the one recorded in
[55-selective-mixed-v-down-projection.md](55-selective-mixed-v-down-projection.md) and
[56-mixed-v-seed-runtime-acceptance.md](56-mixed-v-seed-runtime-acceptance.md): Gemma `mlp.down_proj`, rank 1344,
256 free right-factor rows, and the seed-stable blocks selected by the 0.95% primary-seed weighted-RMSE gate. The
runtime format itself permits other positive dimensions so fixtures and future model families can use the same
codec; selection is an upstream factorization policy, not a loader decision.

## Compact right-factor contract

For each canonical 32-bit sign word in the coded tail:

```text
record = codebook_index | (correction_pair_id << 10)
decoded_word = codebook[codebook_index] XOR correction_mask(correction_pair_id)
```

The record has exactly 19 logical bits:

- 10-bit unsigned index into exactly 1024 `int32` codebook words;
- 9-bit ID for two distinct corrected bit positions;
- valid pair IDs 0 through 495 enumerate the 496 unordered pairs lexicographically;
- IDs 496 through 511 are invalid and rejected.

Records are row-major and tightly packed least-significant-bit first into the minimum
`ceil(record_count * 19 / 32)` contiguous `int32` words. The final unused bits must be zero. No sentinel word and no
persisted correction table are allowed. The fixed pair table is reconstructed algorithmically.

The complete compact layer stores:

- canonical packed `U`;
- canonical packed free `V` prefix;
- the minimal 19-bit coded-tail stream;
- one 1024-word codebook;
- the three scale tensors and the existing optional bias, salient, and low-rank-patch sidecars.

Construction validates the compact payload by expanding it and passing the result through the existing
`PackedLayerState` invariants. Consequently noncanonical sign padding, malformed sidecars, invalid scales, and
out-of-range pair IDs fail before execution.

For the accepted Gemma down projection (`in=6912`, rank 1344, free rows 256), `V` has 216 words per row:

| Component | Bytes |
| --- | ---: |
| Free 256 rows | 221,184 |
| 1088 coded rows at 19 bits per word | 558,144 |
| 1024-word codebook | 4,096 |
| Compact `V` total | 783,424 |
| Fully expanded rank-1344 `V` | 1,161,216 |
| Compact reduction before preparation | 377,792 |

The compact rank-1344 `V` is also 54,656 bytes smaller than an ordinary rank-970 `V`, leaving room for its larger
`U` and middle scale. Artifact accounting must still include all sidecars and shard overhead.

## Artifact and fallback behavior

`nanoquant-mixed-v-overlay.json` schema 1 declares the frozen codec metadata, exact base packed-descriptor hash,
selected source-block indexes, replacement layer specifications, tensor roles/shapes/dtypes, shard sizes and
SHA-256 hashes, replacement count, and overlay shard bytes. Each selected source block has one safetensors shard.
Inspection validates descriptor bounds, paths, hashes, complete tensor inventories, and the rule that a replacement
may differ from its base layer only in rank and rank-shaped tensors without eagerly reading payloads.

An opened overlay provides three explicit reads:

- `load_runtime_layer`: compact state for a replacement, ordinary `PackedLayerState` for fallback;
- `load_compact_layer`: selected compact state only;
- `load_packed_layer`: CPU-expanded canonical state for compatibility and validation.

The overlay is a transition and validation artifact. Retaining it beside a complete base artifact intentionally
duplicates the superseded selected-layer data on disk. A final self-contained bundle must omit those unreachable
base payloads or use a composite shard writer before claiming model-file savings.

The follow-up [projection-family screen](58-mixed-dominant-factor-projection-screen.md) fits tall `gate_proj` and
`up_proj` matrices in transposed orientation. Its coded right factor is therefore the source matrix's left factor.
Those research results do not fit this v1 right-factor-only schema and must not be written as mixed-V replacements.

## Runtime preparation

`torch-packed-reference` accepts the compact state and expands it once before preparing its dense correctness
payload. `cuda-packed-triton` version 2 copies the free words, payload, and codebook to the target device and launches
one Triton expansion kernel on the current stream. The resulting right-word tensor is the same payload used by the
existing stage-one linear kernel. Compact fields and the algorithmic pair table are not retained in the prepared
layer.

Tests cover minimal bit counts, arbitrary record boundaries, exact round trips, invalid pair IDs, nonzero final
padding, exact canonical reconstruction, base-bound persistence, lazy fallback, descriptor corruption, CPU
reference parity, architecture dependency direction, and leased CUDA equality against a separately expanded packed
state.
