# unsloth/gemma-3-270m-it

Benchmark values are arithmetic means across matching runs. Each cell shows how many runs were averaged.
The variant header includes the recorded file size when available.

Largest recorded quantized file: 253,115,552 bytes.

| Benchmark | F16 (542,835,488 bytes) | Q4_K_M (253,115,552 bytes) | IQ4_NL (241,964,192 bytes) | Q2_K (237,079,712 bytes) | Source reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| WikiText-2 (perplexity, samples=64, limit=1000, WikiText samples=64, sequence=128, batch=8) — perplexity | 193.6071 (runs=1) | 205.6365 (runs=3) | 203.7868 (runs=3) | 269.0174 (runs=3) | 194.0538 (runs=10) |
| WikiText-2 (perplexity, samples=64, limit=200, WikiText samples=64, sequence=128, batch=1) — perplexity | — | — | — | — | 193.2281 (runs=1) |
| WikiText-2 (perplexity, samples=64, limit=200, WikiText samples=64, sequence=128, batch=8) — perplexity | — | — | — | — | 194.0538 (runs=6) |
| WikiText-2 (perplexity, samples=570, sequence=512, batch=512) — perplexity | 64.0052 (runs=1) | 66.9298 (runs=3) | 63.966 (runs=3) | 83.4318 (runs=3) | — |
| arc_challenge (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 26.20% (runs=1) | 26.03% (runs=3) | 26.17% (runs=3) | 23.23% (runs=3) | 26.50% (runs=10) |
| arc_challenge (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | 32.50% (runs=1) |
| arc_challenge (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 32.50% (runs=6) |
| arc_easy (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 52.10% (runs=1) | 49.13% (runs=3) | 49.23% (runs=3) | 44.00% (runs=3) | 51.40% (runs=10) |
| arc_easy (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | 48.00% (runs=1) |
| arc_easy (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 48.00% (runs=6) |
| boolq (acc, samples=1000, limit=1000, batch=4) — score_0_to_1 | 58.30% (runs=1) | 45.50% (runs=3) | 52.07% (runs=3) | 59.87% (runs=3) | 57.80% (runs=10) |
| boolq (acc, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | 56.50% (runs=1) |
| boolq (acc, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 56.50% (runs=6) |
| hellaswag (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 43.90% (runs=1) | 43.57% (runs=3) | 42.17% (runs=3) | 42.63% (runs=3) | 43.70% (runs=10) |
| hellaswag (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | 48.00% (runs=1) |
| hellaswag (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 49.00% (runs=6) |
| mmlu (acc, samples=14042, limit=1000, batch=4) — score_0_to_1 | — | — | — | — | 26.39% (runs=1) |
| piqa (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 67.70% (runs=1) | 66.27% (runs=3) | 66.37% (runs=3) | 65.33% (runs=3) | 67.80% (runs=10) |
| piqa (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | 70.00% (runs=1) |
| piqa (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 70.50% (runs=6) |
| winogrande (acc, samples=1000, limit=1000, batch=4) — score_0_to_1 | 53.00% (runs=1) | 51.87% (runs=3) | 51.97% (runs=3) | 52.13% (runs=3) | 52.40% (runs=10) |
| winogrande (acc, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | 54.50% (runs=1) |
| winogrande (acc, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 56.00% (runs=6) |

## Source reports

- **F16**; file size 542,835,488 bytes: `outputs/imatrix-calibration-comparison-v1/native/comparison.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/f16.json`
- **Q4_K_M**; file size 253,115,552 bytes: `outputs/imatrix-calibration-comparison-v1/native/comparison.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/curated-Q4_K_M.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/eaddario-full-Q4_K_M.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/eaddario-matched-Q4_K_M.json`
- **IQ4_NL**; file size 241,964,192 bytes: `outputs/imatrix-calibration-comparison-v1/native/comparison.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/curated-IQ4_NL.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/eaddario-full-IQ4_NL.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/eaddario-matched-IQ4_NL.json`
- **Q2_K**; file size 237,079,712 bytes: `outputs/imatrix-calibration-comparison-v1/native/comparison.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/curated-Q2_K.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/eaddario-full-Q2_K.json`, `outputs/imatrix-calibration-comparison-v1/native/quality/eaddario-matched-Q2_K.json`
- **Source reference**: `Results/007/007-gemma-3-270m-it-quality.json`, `Results/010/010-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/013/013-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/014/014-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/015/015-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/016/016-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/021/021-d2-kl-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/059/059-best-methods-2bpw-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/074/074-fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-pre-kd-quality.json`, `Results/074/074-fast-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/075/075-fast-row-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/076/076-fast-hybrid-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/077/077-curated-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-pre-kd-quality.json`, `Results/077/077-curated-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/078/078-matched-eaddario-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-pre-kd-quality.json`, `Results/078/078-matched-eaddario-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-quality.json`, `Results/081/081-three-full-rank-stages-compress-and-benchmark-gemma-3-270m-it-quality.json`
