# Experiment 2: Gemma 3 1B quality benchmark

- Status: `completed`
- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Candidate run: `D:\dev\research\NanoQuantRewrite\evidence\m4\gemma-pageable-v28-four-block-canary`
- Backend: `factorized`
- Wall time: 555.35 seconds

`completed` means all evaluators returned finite metrics; it is not a BF16-quality acceptance gate.

## Protocol

- WikiText-2: 64 samples × 128 tokens, batch 1
- WikiText token hash: `sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`
- Tasks: piqa, arc_easy, arc_challenge, hellaswag, winogrande, boolq; first 200 rows, batch 1
- Tokenizer hash: `sha256:19317db471b30f6cfa877d781ecac1db28de6628e44e3751df0c44344444a811`

## Quality results

| Benchmark | Metric | BF16 | NanoQuant | Delta | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| WikiText-2 | perplexity ↓ | 96.901169 | 453.570986 | +356.669816 (+368.08%) | 4.6808x |
| piqa | acc_norm ↑ | 0.7150 | 0.5950 | -0.1200 | 0.8322x |
| arc_easy | acc_norm ↑ | 0.6250 | 0.3800 | -0.2450 | 0.6080x |
| arc_challenge | acc_norm ↑ | 0.4000 | 0.2250 | -0.1750 | 0.5625x |
| hellaswag | acc_norm ↑ | 0.5850 | 0.3800 | -0.2050 | 0.6496x |
| winogrande | acc ↑ | 0.6200 | 0.5450 | -0.0750 | 0.8790x |
| boolq | acc ↑ | 0.8100 | 0.5550 | -0.2550 | 0.6852x |

## Runtime and memory

| Model | Elapsed seconds | Peak CUDA bytes | Peak host bytes |
| --- | ---: | ---: | ---: |
| BF16 | 207.88 | 4,148,166,656 | 2,886,381,568 |
| NanoQuant | 336.08 | 4,357,881,856 | 3,648,581,632 |

## Provenance

- Experiment config hash: `sha256:3bd08a348d0d9f1487c7da2c0a1291512b4fa5de3ececd6235f77fbb28052a97`
- Launcher: `experiments/002-benchmark-gemma-3-1b-it.py`
- Candidate identity: `{"config_hash":"sha256:ebe7e831e9a1f4bec5df3bb783359cafd0ddb316856e5f71df5e3b3d379f63db","model_hash":"sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130","plan_hash":"sha256-8656b828a48e72f07799726b92c8a70b257f775c77f6c4bd5787686b524e8a9d"}`
- Global tuning: `{"artifact_id":"sha256-edef5622c5b03e24b75d77ee05f389e064e24d73a3ff7087282d6c3761629669","artifact_type":"global-tuning-result","schema_version":1}`
