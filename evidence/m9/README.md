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

## Experiment 003 base-versus-frozen quality migration

`003-gemma-3-1b-it-quality.json` retains the exact historical smoke dimensions: 16 WikiText windows of 128 tokens
and 25 zero-shot PIQA, ARC-Easy, and BoolQ examples. Dataset revisions, tokenizer content, ordered partition hashes,
the frozen commit identity, global-tuning artifact, raw example scores, timing, and memory peaks are included.

The base-model protocol agrees with retained legacy 003 evidence:

| Metric | Rewrite | Legacy |
| --- | ---: | ---: |
| WikiText PPL | 94.801033 | 94.798897 |
| PIQA `acc_norm` | 0.72 | 0.72 |
| ARC-Easy `acc_norm` | 0.52 | 0.52 |
| BoolQ `acc` | 0.76 | 0.76 |

The migration found that lm-eval 0.4.12's `acc_norm` divides by the original choice's character length, not its token
count, and that its Gemma path uses eager attention plus BF16 log-softmax accumulation. Those behaviors are now
explicit and regression-tested. The v28 frozen candidate measures PPL 396.572849 and primary task values 0.64, 0.32,
and 0.60 respectively; historical 003 used a different residual-outlier checkpoint, so those candidate values are
not presented as checkpoint-output parity.

The result is 142,854 bytes with SHA-256
`a90f880b90dd91b957bf7a179d9941f0cbc8bf55ca061f8e0915347d5d1ee604`.
