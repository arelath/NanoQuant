# google/gemma-3-4b-it

Benchmark values are arithmetic means across matching runs. Each cell shows how many runs were averaged.
The variant header includes the recorded file size when available.

No quantized file size was recorded for this model.

| Benchmark | Source reference |
| --- | ---: |
| WikiText-2 (perplexity, samples=64, limit=1000, WikiText samples=64, sequence=128, batch=8) — perplexity | 51.7556 (runs=1) |
| WikiText-2 (perplexity, samples=64, limit=200, WikiText samples=64, sequence=128, batch=1) — perplexity | 51.6386 (runs=3) |
| arc_challenge (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 57.30% (runs=1) |
| arc_challenge (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | 52.00% (runs=3) |
| arc_easy (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 77.70% (runs=1) |
| arc_easy (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | 76.50% (runs=3) |
| boolq (acc, samples=1000, limit=1000, batch=4) — score_0_to_1 | 84.20% (runs=1) |
| boolq (acc, samples=200, limit=200, batch=1) — score_0_to_1 | 87.00% (runs=3) |
| hellaswag (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 65.10% (runs=1) |
| hellaswag (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | 64.50% (runs=3) |
| piqa (acc_norm, samples=1000, limit=1000, batch=4) — score_0_to_1 | 77.20% (runs=1) |
| piqa (acc_norm, samples=200, limit=200, batch=1) — score_0_to_1 | 76.50% (runs=3) |
| winogrande (acc, samples=1000, limit=1000, batch=4) — score_0_to_1 | 69.20% (runs=1) |
| winogrande (acc, samples=200, limit=200, batch=1) — score_0_to_1 | 66.00% (runs=3) |

## Source reports

- **Source reference**: `Results/072/072-two-full-rank-stages-compress-and-benchmark-gemma-3-4b-it-quality.json`, `evidence/m10/003-gemma-3-4b-it-quality.json`, `evidence/m11/004-gemma-3-4b-it-vproj-plus30-quality.json`, `evidence/m12/005-gemma-3-4b-it-vproj-maxrank-quality.json`
