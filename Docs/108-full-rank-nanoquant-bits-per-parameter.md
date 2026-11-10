# Full rank NanoQuant bits per parameter

Full-rank NanoQuant factorization requires approximately **1.28–1.40 bits per original parameter** for the transformer projection matrices in Gemma 3 1B, 4B, 12B, 27B, and Llama 3.3 70B, including FP32 scales. These are calculated packed-storage estimates, not measured compression or quality results.

## Scope and definition

For each weight matrix with `m` output rows and `n` input columns, full rank means setting the factor rank to `r = min(m, n)`. Each matrix is represented by two binary factors and three scale vectors.

The calculation includes seven separate matrices per language-model transformer block: attention Q, K, V, and O projections, plus MLP gate, up, and down projections. The denominator counts the original parameters in those matrices only.

Embeddings, output heads, vision components, normalization weights, outlier corrections, optional residual patches, metadata, and container overhead are excluded. Optional shared QKV factorization is not assumed.

## Results

| Model | Parameters in quantized matrices | Binary factors only, bits/parameter | With FP16 scales, bits/parameter | With FP32 scales, bits/parameter |
|---|---:|---:|---:|---:|
| Gemma 3 1B | 697,761,792 | 1.231380 | 1.253663 | **1.275946** |
| Gemma 3 4B | 3,208,642,560 | 1.319444 | 1.331076 | **1.342708** |
| Gemma 3 12B | 10,758,389,760 | 1.366374 | 1.374123 | **1.381871** |
| Gemma 3 27B | 25,598,361,600 | 1.311587 | 1.317123 | **1.322659** |
| Llama 3.3 70B | 68,451,041,280 | 1.394608 | 1.398438 | **1.402267** |

The FP16 column matches NanoQuant's default planning scale-bit assumption. The FP32 column reflects the scale precision observed in existing GGUF exports, while still excluding the additional payloads and container overhead listed above. Neither column is a whole-model file-size estimate.

## Calculation

For scale precision `s` bits, the packed factor and scale cost of one matrix is:

```text
binary factor bits = r × (m + n)
scale bits         = s × (m + n + r)
bits per parameter = [r × (m + n) + s × (m + n + r)] / (m × n)
r                  = min(m, n)
```

The model average is the sum of all included matrix costs divided by the sum of their original parameter counts. It is not an unweighted average of the seven projection ratios. All full ranks in this comparison are multiples of 32, so rank alignment does not change the results.

For a square `d × d` matrix, the binary factors alone cost 2 bits per original parameter. For a rectangular matrix with `m >= n`, the factor-only cost is `1 + n/m`. Wide MLP matrices therefore reduce the model average substantially below 2 bits per parameter. Scale overhead becomes smaller as matrix dimensions grow.

## Architecture inputs

| Model | Blocks | Hidden width | MLP width | Query heads | KV heads | Head dimension |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 1B | 26 | 1,152 | 6,912 | 4 | 1 | 256 |
| Gemma 3 4B | 34 | 2,560 | 10,240 | 8 | 4 | 256 |
| Gemma 3 12B | 48 | 3,840 | 15,360 | 16 | 8 | 256 |
| Gemma 3 27B | 62 | 5,376 | 21,504 | 32 | 16 | 128 |
| Llama 3.3 70B | 80 | 8,192 | 28,672 | 64 | 8 | 128 |

With hidden width `h`, MLP width `i`, query width `q = query_heads × head_dimension`, and KV width `k = kv_heads × head_dimension`, the seven matrix shapes are:

```text
Q:    q × h
K:    k × h
V:    k × h
O:    h × q
gate: i × h
up:   i × h
down: h × i
```

Architecture inputs were resolved from the Hugging Face model configurations, including configuration defaults for omitted fields. Configuration revisions used:

| Model | Configuration revision |
|---|---|
| Gemma 3 1B | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| Gemma 3 4B | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Gemma 3 12B | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| Gemma 3 27B | `005ad3404e59d6023443cb575daa05336842228a` |
| Llama 3.3 70B | `6f6073b423013f6a7d4d9f39144961bfbfbc386b` |

Reference configurations: [Google Gemma](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py) and [Meta Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/6f6073b423013f6a7d4d9f39144961bfbfbc386b/config.json).

## Interpretation

Full rank does **not** imply lossless reconstruction. Binary factors remain constrained even when their inner dimension reaches the original matrix's maximum possible rank. More binary components can improve the fit beyond that point; this is an overcomplete factorization.

Consequently, a 2-bit-per-parameter NanoQuant budget can exceed the full-rank binary-factor cost without guaranteeing an exact fit. Differences between the models above come primarily from their matrix aspect ratios and scale overhead, rather than total model size.

The implementation's factor-cost formula is in [`factor_bit_cost`](../src/nanoquant/domain/planning.py). Quantizable projection definitions are in [`model_adapters.py`](../src/nanoquant/infrastructure/model_adapters.py), and trainable factor and scale shapes are represented by [`TrainableFactorizedLinear`](../src/nanoquant/application/layers.py).
