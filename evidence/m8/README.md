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

## Replay-to-full campaign gate

`gemma-pageable-v28-evaluation-campaign-v2` is the self-contained M8.GATE directory. It retains seven copied inputs,
their hashes, one canonical campaign result, lifecycle/promotion events, the full resolved intent and environment,
resource/cost observations, and generated summary/comparison reports. All three tier policies promote and the report
contains no consistency warning or warning/error event.

- Campaign JSON: SHA-256 `ccacea00ee1b4e7678ae9f662827fd45cf5ec28071918c88ef67f799a0f35cc8`.
- Manifest: SHA-256 `31c71cdb5a501daffd54954a75f17149f7e9e09f9704a0c4647d0b3b4defb1b5`.
- Event stream: SHA-256 `f9ee74a2b9d8ebaae1db35132bf0fec87ecabbb948375c8a9707938e0e9a9adb`.
- Summary: SHA-256 `4dc6d6a259dd277d28ad0e9d6003f7f06e29209ee168b763ca5f6611c03e79f0`.
- Comparison: SHA-256 `e4132daf9240cddadbb2a6c9320c7f9c0f7cd5cabd32df6c2fd772f650a8454d`.
