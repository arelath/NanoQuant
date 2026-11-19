# google/gemma-3-1b-it

Benchmark values are arithmetic means across matching runs. Each cell shows how many runs were averaged.
The variant header includes the recorded file size when available.

Largest recorded quantized file: 806,058,272 bytes.

| Benchmark | Q4_K_M (806,058,272 bytes) | UD-Q3_K_XL (726,839,840 bytes) | Q4_0 (721,918,496 bytes) | UD-Q2_K_XL (693,858,080 bytes) | Q2_K (689,814,560 bytes) | IQ2_M (669,783,808 bytes) | UD-IQ1_S (557,014,304 bytes) | Source reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WikiText-2 (perplexity, samples=16, limit=25, WikiText samples=16, sequence=128, batch=1) — perplexity | — | — | — | — | — | — | — | 94.801 (runs=1) |
| WikiText-2 (perplexity, samples=2302, WikiText samples=2302, sequence=128, batch=8) — perplexity | — | — | — | — | 97.6878 (runs=1) | — | — | 83.2586 (runs=1) |
| WikiText-2 (perplexity, samples=64, limit=1000, WikiText samples=64, sequence=128, batch=8) — perplexity | 98.0934 (runs=1) | 102.3971 (runs=1) | 113.4178 (runs=1) | 105.2282 (runs=1) | — | 111.619 (runs=1) | 162.6418 (runs=1) | 96.4596 (runs=26) |
| WikiText-2 (perplexity, samples=64, limit=200, WikiText samples=64, sequence=128, batch=1) — perplexity | — | — | — | — | — | — | — | 96.9012 (runs=2) |
| WikiText-2 (perplexity, samples=64, limit=200, WikiText samples=64, sequence=128, batch=8) — perplexity | — | — | — | — | 109.3679 (runs=2) | — | — | 96.4596 (runs=20) |
| arc_challenge (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 36.30% (runs=1) | 39.10% (runs=1) | 32.70% (runs=1) | 34.70% (runs=1) | — | 36.10% (runs=1) | 30.10% (runs=1) | 37.60% (runs=26) |
| arc_challenge (acc_norm, samples=1172, batch=4) — score_0_to_1 | — | — | — | — | 35.07% (runs=1) | — | — | 37.88% (runs=1) |
| arc_challenge (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 40.00% (runs=2) |
| arc_challenge (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 37.50% (runs=2) | — | — | 39.50% (runs=20) |
| arc_easy (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 61.70% (runs=1) | 63.00% (runs=1) | 58.80% (runs=1) | 59.40% (runs=1) | — | 58.40% (runs=1) | 52.10% (runs=1) | 63.80% (runs=26) |
| arc_easy (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 62.50% (runs=2) |
| arc_easy (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 58.50% (runs=2) | — | — | 60.50% (runs=20) |
| arc_easy (acc_norm, samples=2376, batch=4) — score_0_to_1 | — | — | — | — | 58.12% (runs=1) | — | — | 63.17% (runs=1) |
| arc_easy (acc_norm, samples=25, limit=25, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 52.00% (runs=1) |
| boolq (acc, samples=1000, limit=1000, batch=4) — score_0_to_1 | 75.00% (runs=1) | 76.10% (runs=1) | 69.70% (runs=1) | 70.00% (runs=1) | — | 70.10% (runs=1) | 63.60% (runs=1) | 76.80% (runs=26) |
| boolq (acc, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 81.00% (runs=2) |
| boolq (acc, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 78.00% (runs=2) | — | — | 81.00% (runs=20) |
| boolq (acc, samples=25, limit=25, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 76.00% (runs=1) |
| boolq (acc, samples=3270, batch=4) — score_0_to_1 | — | — | — | — | 71.96% (runs=1) | — | — | 75.57% (runs=1) |
| hellaswag (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 52.60% (runs=1) | 52.80% (runs=1) | 51.70% (runs=1) | 50.30% (runs=1) | — | 49.60% (runs=1) | 47.20% (runs=1) | 52.80% (runs=26) |
| hellaswag (acc_norm, samples=10042, batch=4) — score_0_to_1 | — | — | — | — | 54.61% (runs=1) | — | — | 57.82% (runs=1) |
| hellaswag (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 58.50% (runs=2) |
| hellaswag (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 56.00% (runs=2) | — | — | 58.50% (runs=20) |
| mmlu (acc, samples=14042, limit=1000, batch=4) — score_0_to_1 | — | — | — | — | — | — | — | 39.77% (runs=1) |
| piqa (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 72.60% (runs=1) | 71.70% (runs=1) | 73.20% (runs=1) | 71.40% (runs=1) | — | 72.90% (runs=1) | 68.10% (runs=1) | 73.00% (runs=26) |
| piqa (acc_norm, samples=1838, batch=4) — score_0_to_1 | — | — | — | — | 70.67% (runs=1) | — | — | 72.20% (runs=1) |
| piqa (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 71.50% (runs=2) |
| piqa (acc_norm, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 72.00% (runs=2) | — | — | 72.00% (runs=20) |
| piqa (acc_norm, samples=25, limit=25, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 72.00% (runs=1) |
| winogrande (acc, samples=1000, limit=1000, batch=4) — score_0_to_1 | 59.60% (runs=1) | 57.50% (runs=1) | 59.20% (runs=1) | 59.30% (runs=1) | — | 59.20% (runs=1) | 55.90% (runs=1) | 59.40% (runs=26) |
| winogrande (acc, samples=1267, batch=4) — score_0_to_1 | — | — | — | — | 59.35% (runs=1) | — | — | 58.72% (runs=1) |
| winogrande (acc, samples=200, limit=200, batch=1) — score_0_to_1 | — | — | — | — | — | — | — | 62.00% (runs=2) |
| winogrande (acc, samples=200, limit=200, batch=4) — score_0_to_1 | — | — | — | — | 63.00% (runs=2) | — | — | 62.50% (runs=20) |

## Source reports

- **Q4_K_M**; file size 806,058,272 bytes: `outputs/073/gguf-comparison/unsloth-q4_k_m.json`
- **UD-Q3_K_XL**; file size 726,839,840 bytes: `outputs/073/gguf-comparison/unsloth-ud-q3_k_xl.json`
- **Q4_0**; file size 721,918,496 bytes: `evidence/quality/20260812-unsloth-gemma3-1b-q4_0-vs-experiment-058/unsloth-gemma-3-1b-it-Q4_0-quality.json`
- **UD-Q2_K_XL**; file size 693,858,080 bytes: `outputs/073/gguf-comparison/unsloth-ud-q2_k_xl.json`
- **Q2_K**; file size 689,814,560 bytes: `outputs/hf-quality/070-bf16-unsloth-full/unsloth-q2-k.json`, `outputs/hf-quality/20261111T154349925198Z/quality.json`, `outputs/hf-quality/20261111T155551345855Z/quality.json`
- **IQ2_M**; file size 669,783,808 bytes: `Results/069/external_iq2_m-quality.json`
- **UD-IQ1_S**; file size 557,014,304 bytes: `outputs/073/gguf-comparison/unsloth-ud-iq1_s.json`
- **Source reference**: `Results/006/006-gemma-3-1b-it-quality.json`, `Results/011/011-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/012/012-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/017/017-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/022/022-d2-kl-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/023/023-d2-kl-interaction-tuned-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/024/024-best-methods-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/032/032-raw-fisher-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/033/033-covariance-refined-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/034/034-post-refit-qkv-covariance-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/035/035-foldable-mlp-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/037/037-matched-tail-aware-d2-base-gemma-3-1b-it-quality.json`, `Results/040/040-low-pressure-correction-minimal-fold-gemma-3-1b-it-quality.json`, `Results/040/040-low-pressure-correction-minimal-fold-gemma-3-1b-it-tasklimit1000-quality.json`, `Results/042/042-low-pressure-correction-d2-compress-and-benchmark-gemma-3-1b-it-gguf-quality.json`, `Results/042/042-low-pressure-correction-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/044/044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it-gguf-quality.json`, `Results/044/044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/048/048-adaptive-capability-correction-d2-gemma-3-1b-it-gguf-quality.json`, `Results/048/048-adaptive-capability-correction-d2-gemma-3-1b-it-quality.json`, `Results/054/054-functional-binary-lr-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/055/055-overcomplete-rank-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/056/056-physical-cap-rank-redistribution-d2-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/057/057-matched-056-product-k16-codebook-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/058/058-matched-057-rate-matched-int8-outliers-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/060/060-best-methods-2bpw-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/069/external_iq2_m-quality.json`, `Results/070/070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/071/071-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/073/073-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it-pre-kd-quality.json`, `Results/073/073-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it-quality.json`, `Results/080/080-fast-imatrix-no-kd-compress-and-benchmark-gemma-3-1b-it-quality.json`, `evidence/037/experiment037-conditional-standard-quality.json`, `evidence/037/experiment037-conditional-tasklimit1000-quality.json`, `evidence/037/experiment037-tail-aware-folded1p06-standard-quality.json`, `evidence/037/experiment037-tail-aware-folded1p06-tasklimit1000-quality.json`, `evidence/037/experiment037-tail-aware-uncalibrated-standard-quality.json`, `evidence/037/experiment037-tail-aware-uncalibrated-tasklimit1000-quality.json`, `evidence/040/experiment040-weight2-epoch1-fold1p015-standard-quality.json`, `evidence/040/experiment040-weight2-epoch1-fold1p015-tasklimit1000-quality.json`, `evidence/042-analysis/experiment042-candidate-packed-tasklimit1000-quality.json`, `evidence/042-analysis/experiment042-conditional-tasklimit1000-quality.json`, `evidence/042/experiment042-conditional-epoch8-standard-quality.json`, `evidence/042/experiment042-conditional-epoch8-tasklimit1000-quality.json`, `evidence/042/experiment042-final-tasklimit1000-quality.json`, `evidence/044/experiment044-tail256-tasklimit1000-quality.json`, `evidence/m9/002-gemma-3-1b-it-quality-benchmark.json`, `evidence/m9/003-gemma-3-1b-it-quality.json`, `evidence/quality/20260812-unsloth-gemma3-1b-q4_0-vs-experiment-058/unsloth-gemma-3-1b-it-Q4_0-quality.json`, `outputs/hf-quality/070-bf16-unsloth-full/baseline.json`
