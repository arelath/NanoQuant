# Milestone 9 migration evidence

## Active Experiment 002 full quality benchmark

`002-gemma-3-1b-it-quality-benchmark.json` is the benchmark-only run requested after the active experiment
chronology was reset. It compares the pinned BF16 source model with the accepted v28 NanoQuant candidate over the
same common dimensions as legacy Experiment 007: 64 WikiText-2 windows of 128 tokens and the first 200 zero-shot
examples from PIQA, ARC Easy, ARC Challenge, HellaSwag, WinoGrande, and BoolQ. Model lifetimes are sequential.
`002-gemma-3-1b-it-quality-benchmark.md` is the human-readable report generated from the same retained payload;
future Experiment 002 runs write both files directly.

| Metric | BF16 | NanoQuant | Delta |
| --- | ---: | ---: | ---: |
| WikiText-2 PPL | 96.9012 | 453.5710 | +368.08% |
| PIQA `acc_norm` | 0.715 | 0.595 | -0.120 |
| ARC Easy `acc_norm` | 0.625 | 0.380 | -0.245 |
| ARC Challenge `acc_norm` | 0.400 | 0.225 | -0.175 |
| HellaSwag `acc_norm` | 0.585 | 0.380 | -0.205 |
| WinoGrande `acc` | 0.620 | 0.545 | -0.075 |
| BoolQ `acc` | 0.810 | 0.555 | -0.255 |

The quality loss versus BF16 is substantial and must not be hidden by the result's `passed` field, which means only
that every evaluator completed with finite metrics. Against the matching legacy 007 phase-1/no-Hessian task row,
the rewrite deltas are `+0.005`, `0.000`, `+0.020`, `-0.030`, `+0.015`, and `-0.025`; mean absolute delta is
0.0158 and maximum absolute delta is 0.030. Exact serial PPL remains compared with the contemporary legacy run:
453.5710 versus 444.3328 (+2.0791%). Thus the benchmark confirms old/new quantized quality parity while showing
the large absolute cost of approximately 1 BPW relative to BF16.

BF16 evaluation took 207.88 seconds and NanoQuant took 336.08 seconds, making the factorized evaluation path 61.7%
slower. Peak allocated CUDA was 4,148,166,656 and 4,357,881,856 bytes respectively. Total wall time was 555.35
seconds. The result is 1,883,677 bytes with SHA-256
`88434096021b8733e91b1c7f116d39f5b3d508457d61fc7ada2af583ade5abac`.

## Retained legacy Experiment 002 paired short-decode migration

`002-gemma-3-1b-it-short-decode.json` is the canonical three-case migration of the historical original/eager/GEMV
benchmark. It preserves the raw prompt, legacy non-special-token fill to 32 prompt tokens, 32 generated tokens,
top-k 32, temperature 0.8, seed zero, one warmup, three measurements, and the historical timer boundary after an
unsynchronized prefill. Each model is loaded, measured, and released before the next model, so source, logical, and
packed representations never overlap in VRAM.

| Current case | Aggregate decode tokens/s | Peak allocated bytes | Ratio vs current base |
| --- | ---: | ---: | ---: |
| Source Transformers | 14.325 | 2,081,723,392 | 1.000x throughput / 1.000x memory |
| Logical factorized reference | 7.938 | 1,993,371,648 | 0.554x / 0.958x |
| Immutable packed production | 12.444 | 720,707,072 | 0.869x / 0.346x |

Packed production prepares all 182 linears on the CUDA packed backend with zero prefill or decode fallback. Its peak
allocation is 65.4% below the current base and within 0.2% of legacy GEMV's retained 719,535,616 bytes. The packed
case is still 13.1% slower than the current BF16 base on this short context; this is retained as a performance gap,
not turned into a false parity claim. The legacy eager/GEMV rows used a different smoke checkpoint and mutable
runtime modes, so their values remain historical references rather than paired numerical-checkpoint evidence.

The result is 16,878 bytes with SHA-256
`a32f0ffc092d426842e50c97b61245f561fae60aa3884e79bfe4c5979d7feb7c`.

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
