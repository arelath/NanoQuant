# Milestone 8 evaluation evidence

## Packed Gemma long-context gate

All three results use the self-contained `gemma-pageable-v28-runtime-bundle` descriptor
`e5cef7236e2846a49129e991f8fc5efb660a9a8b8c71d9531590bed71739cc42`, the designated RTX 4000 Ada Laptop GPU,
F32 shell execution, greedy generation, and four forced output tokens.

| Evidence | Prompt | Oracle / candidate chunk | Exact | Candidate peak bytes | SHA-256 |
| --- | ---: | ---: | --- | ---: | --- |
| `gemma-pageable-v28-long-context-1024.json` | 1,025 | monolithic / 512 | yes | 856,781,312 | `75d5141123d7b8416a5e427874cc012e24f32ef1dab2245e5cddab616000b688` |
| `gemma-pageable-v28-long-context-4096.json` | 4,097 | monolithic / 512 | yes | 899,101,696 | `f63fc2025b4b10356a8a6a77b19e4c3593285b984475e234d5447dc5912ec581` |
| `gemma-pageable-v28-long-context-ceiling.json` | 32,761 | 256 / 512 | yes | 1,592,178,176 | `c17f2cf200fcd151d880c4389ae191a4c6dfb296628047abc950803b06db3cf4` |

The near-ceiling case totals 32,765 tokens against the model's declared 32,768-token limit. It observes 128 oracle
and 64 candidate prefill forwards, three decode forwards, the exact 32,765 cache bound, identical generated tokens
and stop reason, and zero execution-plan or generation fallbacks. The independently chunked oracle avoids recreating
the monolithic eager-attention VRAM hazard that bounded prefill is intended to prevent.
