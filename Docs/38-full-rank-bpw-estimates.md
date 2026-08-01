# Full-rank NanoQuant bits-per-parameter estimates

This document estimates the storage cost of applying NanoQuant's binary-factor representation at maximum physical
rank to every supported decoder linear. It covers representative dense text models from Gemma 3 270M through
Llama 3.3 70B and includes the common size classes between them.

These are capacity estimates, not measured compression results. In particular, maximum rank is the largest
representable factor rank, not a guarantee of exact reconstruction or acceptable model quality.

## Summary

| Model | Blocks | Text parameters | Quantized decoder parameters | Full-rank core BPW | Estimated whole-text-model BPW | Estimated payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 270M | 18 | 0.268B | 0.100B | 1.4397 | 5.8610 | 0.18 GiB |
| Qwen3 0.6B | 28 | 0.596B | 0.440B | 1.4969 | 3.3264 | 0.23 GiB |
| Gemma 3 1B | 26 | 1.000B | 0.698B | 1.2537 | 3.4442 | 0.40 GiB |
| Qwen3 1.7B | 28 | 1.721B | 1.409B | 1.4746 | 2.7462 | 0.55 GiB |
| Llama 3.2 3B | 28 | 3.213B | 2.819B | 1.5010 | 2.3601 | 0.88 GiB |
| Gemma 3 4B | 34 | 3.880B | 3.209B | 1.3311 | 2.5726 | 1.16 GiB |
| Qwen3 4B | 36 | 4.022B | 3.633B | 1.3569 | 2.0483 | 0.96 GiB |
| Llama 3 8B | 32 | 8.030B | 6.979B | 1.4019 | 2.3311 | 2.18 GiB |
| Qwen3 8B | 36 | 8.191B | 6.946B | 1.4538 | 2.5251 | 2.41 GiB |
| Gemma 3 12B | 48 | 11.766B | 10.758B | 1.3741 | 1.9849 | 2.72 GiB |
| Qwen3 14B | 40 | 14.768B | 13.212B | 1.4094 | 2.1568 | 3.71 GiB |
| Gemma 3 27B | 62 | 27.009B | 25.598B | 1.3171 | 1.6927 | 5.32 GiB |
| Qwen3 32B | 64 | 32.762B | 31.206B | 1.2783 | 1.6216 | 6.18 GiB |
| Llama 3.3 70B | 80 | 70.554B | 68.451B | 1.3984 | 1.6102 | 13.23 GiB |

`B` is decimal billions of unique parameters. `GiB` is binary gibibytes. Gemma multimodal checkpoints are counted
as text models here: their vision tower and multimodal projector are excluded.

The two BPW columns have different denominators:

- **Full-rank core BPW** is NanoQuant factor and scale bits divided by the parameters in the seven quantized decoder
  matrices per block.
- **Estimated whole-text-model BPW** adds Q8_0 token embeddings and, when untied, the output head, plus BF16 for
  remaining text parameters, then divides by all unique text-model parameters.

The second value is a logical payload estimate, not `artifact_bpw`. GGUF metadata, tensor alignment, container
overhead, and optional duplicate layouts require a real export before exact artifact BPW is known.

## Calculation

For a weight matrix with `m` output features, `n` input features, and factor rank `r`, the repository's
`factor_bit_cost` calculation is:

```text
factor_bits(m, n, r)
    = r(m + n)                  # one-bit left and right factors
    + 16(m + n + r)             # BF16 pre, mid, and post scales
    + packing_padding
```

Full rank means:

```text
r = min(m, n)
```

The calculation uses rank alignment 32, matching the packed runtime. Every maximum rank in this table is already a
multiple of 32, so packing padding is zero.

For model `M`, the two reported estimates are:

```text
core_bpw(M)
    = sum(factor_bits for all decoder matrices)
      / sum(source parameters in those matrices)

whole_model_bpw(M)
    = (factor bits
       + 8.5 * embedding-and-untied-head parameters
       + 16 * remaining text parameters)
      / unique text-model parameters
```

Q8_0 costs 8.5 logical bits per value: each block stores 32 signed 8-bit values and one 16-bit scale. This matches
the base export policy's Q8_0 treatment of `token_embd.weight` and an independent `output.weight`.

The estimate deliberately excludes:

- residual outliers, their indices, bias correction, and low-rank patches;
- retries or any rank beyond the physical maximum;
- GGUF keys, tensor descriptors, alignment, and container overhead;
- tokenizer and other non-weight files;
- Gemma 3 vision-tower and multimodal-projector parameters.

## Full-rank projection inventory

The adapter quantizes `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` in every
decoder block. Gate, up, and down projections have the same maximum rank for all models below, so they are shown as
one MLP-rank column.

| Model | Hidden | Intermediate | Q rank | K rank | V rank | O rank | MLP rank | Vocabulary | Tied embedding/head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Gemma 3 270M | 640 | 2,048 | 640 | 256 | 256 | 640 | 640 | 262,144 | yes |
| Qwen3 0.6B | 1,024 | 3,072 | 1,024 | 1,024 | 1,024 | 1,024 | 1,024 | 151,936 | yes |
| Gemma 3 1B | 1,152 | 6,912 | 1,024 | 256 | 256 | 1,024 | 1,152 | 262,144 | yes |
| Qwen3 1.7B | 2,048 | 6,144 | 2,048 | 1,024 | 1,024 | 2,048 | 2,048 | 151,936 | yes |
| Llama 3.2 3B | 3,072 | 8,192 | 3,072 | 1,024 | 1,024 | 3,072 | 3,072 | 128,256 | yes |
| Gemma 3 4B | 2,560 | 10,240 | 2,048 | 1,024 | 1,024 | 2,048 | 2,560 | 262,208 | yes |
| Qwen3 4B | 2,560 | 9,728 | 2,560 | 1,024 | 1,024 | 2,560 | 2,560 | 151,936 | yes |
| Llama 3 8B | 4,096 | 14,336 | 4,096 | 1,024 | 1,024 | 4,096 | 4,096 | 128,256 | no |
| Qwen3 8B | 4,096 | 12,288 | 4,096 | 1,024 | 1,024 | 4,096 | 4,096 | 151,936 | no |
| Gemma 3 12B | 3,840 | 15,360 | 3,840 | 2,048 | 2,048 | 3,840 | 3,840 | 262,208 | yes |
| Qwen3 14B | 5,120 | 17,408 | 5,120 | 1,024 | 1,024 | 5,120 | 5,120 | 151,936 | no |
| Gemma 3 27B | 5,376 | 21,504 | 4,096 | 2,048 | 2,048 | 4,096 | 5,376 | 262,208 | yes |
| Qwen3 32B | 5,120 | 25,600 | 5,120 | 1,024 | 1,024 | 5,120 | 5,120 | 151,936 | no |
| Llama 3.3 70B | 8,192 | 28,672 | 8,192 | 1,024 | 1,024 | 8,192 | 8,192 | 128,256 | no |

## Interpretation

Model size alone does not determine maximum-rank BPW. Matrix aspect ratios and grouped-query attention determine how
many one-bit factor values are shared by each source weight. This is why Gemma 3 1B has a 1.2537 core BPW estimate
while the similarly small Qwen3 0.6B and Llama 3.2 3B estimates are close to 1.50 BPW.

Embeddings explain the large gap between core and whole-model BPW at small sizes. Gemma 3 270M has approximately
168 million unique embedding parameters but only 100 million quantized decoder parameters. Even at Q8_0, that
embedding pushes the estimated complete text payload from 1.4397 core BPW to 5.8610 whole-model BPW. At 70B, the
embedding and head are a much smaller fraction, so the two numbers converge.

The table should therefore be used as follows:

- use **core BPW** to compare factorization cost or plan rank budgets;
- use **whole-text-model BPW** and **payload GiB** for early storage and download planning;
- use measured `effective_bpw`, serialized bytes, and `artifact_bpw` from a completed run for experiment claims.

## Configuration provenance

The dimensions were resolved from the model configurations used by the rewrite's supported Gemma 3, Qwen3, and
Llama adapters. The representative sources are:

- [Gemma 3 270M](https://huggingface.co/unsloth/gemma-3-270m-it),
  [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it),
  [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it),
  [Gemma 3 12B](https://huggingface.co/unsloth/gemma-3-12b-it), and
  [Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it);
- [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B),
  [Qwen3 1.7B](https://huggingface.co/Qwen/Qwen3-1.7B),
  [Qwen3 4B](https://huggingface.co/Qwen/Qwen3-4B),
  [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B),
  [Qwen3 14B](https://huggingface.co/Qwen/Qwen3-14B), and
  [Qwen3 32B](https://huggingface.co/Qwen/Qwen3-32B);
- [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct),
  [Meta-Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), and
  [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

The cost definition is authoritative in `src/nanoquant/domain/planning.py`; the seven-layer architecture contract is
authoritative in `src/nanoquant/infrastructure/model_adapters.py`; and the distinction between core and artifact BPW
is defined in `Docs/10-artifacts-and-compatibility.md`.
