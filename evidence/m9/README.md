# Milestone 9 migration evidence

## Experiment 011 generation-throughput migration

`011-generation-tps.json` is the canonical zero-argument migration result for legacy Experiment 011. It uses:

- pinned `google/gemma-3-1b-it` revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`;
- production v28 packed descriptor SHA-256
  `b4f0c6270c4b59f8293c909ddeb21042ad1a2d7ee18601c77e4c57563c900487`;
- the exact legacy raw prompt (12 tokens), BF16 input/cache, forced 128-token output, one warmup, and three timed
  repetitions;
- generation-only timing after model loading, packed preparation, tokenization, and warmup.

The run passed with all 182 linears prepared on `cuda-packed-triton`, zero prefill/decode fallback, and deterministic
output hash `06e2b19ec33bcd0c3822f82928b1bf5aa9e3f4af7e77f43561796f2cfc8aa955`. Complete-generation median throughput is
116.897 tokens/s; mean throughput is 110.178 tokens/s. The retained legacy GEMV result for the same named protocol is
22.499 mean tokens/s, so the migrated production runtime is 4.90x faster on that historical workload.

The result is 14,122 bytes with SHA-256
`e7933acba9014ae9adb9e2d456b9dd1c60a1e3bcd9ecf815192ce9c1327fe981`.
