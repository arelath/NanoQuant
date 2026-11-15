# Gemma 3 270M imatrix task-quality comparison

Measured on 2026-11-15. Curated calibration's mean gains over the equal-budget
control are **+0.25 percentage points for Q2_K, +1.07 for Q4_K_M, and +0.07 for
IQ4_NL**. The Q4_K_M result reverses the earlier perplexity ranking. Per-task
changes are mixed, so the mean should be read alongside the individual scores.

All ten existing GGUFs were scored through the same llama.cpp runtime on the same prepared token IDs.
Each task uses the first 1,000 examples of its pinned split, the established prompts and primary metrics.
PIQA, ARC Easy, ARC Challenge and HellaSwag use acc_norm; WinoGrande and BoolQ use acc.
Higher is better. Scores below are percentages; the macro mean weights the six tasks equally.
The standard evaluator also computes WikiText diagnostically; it is not used to rank this comparison.

| Model / imatrix | PIQA | ARC Easy | ARC Challenge | HellaSwag | WinoGrande | BoolQ | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f16 | 67.7 | 52.1 | 26.2 | 43.9 | 53.0 | 58.3 | 50.20 |
| curated-Q2_K | 65.1 | 43.8 | 22.8 | 42.8 | 51.5 | 61.5 | 47.92 |
| eaddario-matched-Q2_K | 65.6 | 44.0 | 23.7 | 42.2 | 52.2 | 58.3 | 47.67 |
| eaddario-full-Q2_K | 65.3 | 44.2 | 23.2 | 42.9 | 52.7 | 59.8 | 48.02 |
| curated-Q4_K_M | 66.3 | 50.0 | 26.6 | 44.1 | 51.6 | 48.0 | 47.77 |
| eaddario-matched-Q4_K_M | 66.3 | 48.7 | 25.6 | 43.4 | 52.4 | 43.8 | 46.70 |
| eaddario-full-Q4_K_M | 66.2 | 48.7 | 25.9 | 43.2 | 51.6 | 44.7 | 46.72 |
| curated-IQ4_NL | 67.7 | 49.4 | 26.1 | 43.0 | 52.6 | 50.3 | 48.18 |
| eaddario-matched-IQ4_NL | 65.4 | 49.8 | 26.8 | 41.8 | 51.8 | 53.1 | 48.12 |
| eaddario-full-IQ4_NL | 66.0 | 48.5 | 25.6 | 41.7 | 51.5 | 52.8 | 47.68 |

## Curated minus control

Differences are percentage points, not relative percentages.

| Format | Control | PIQA | ARC Easy | ARC Challenge | HellaSwag | WinoGrande | BoolQ | Mean delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2_K | eaddario-matched | -0.5 | -0.2 | -0.9 | +0.6 | -0.7 | +3.2 | +0.25 |
| Q2_K | eaddario-full | -0.2 | -0.4 | -0.4 | -0.1 | -1.2 | +1.7 | -0.10 |
| Q4_K_M | eaddario-matched | +0.0 | +1.3 | +1.0 | +0.7 | -0.8 | +4.2 | +1.07 |
| Q4_K_M | eaddario-full | +0.1 | +1.3 | +0.7 | +0.9 | +0.0 | +3.3 | +1.05 |
| IQ4_NL | eaddario-matched | +2.3 | -0.4 | -0.7 | +1.2 | +0.8 | -2.8 | +0.07 |
| IQ4_NL | eaddario-full | +1.7 | +0.9 | +0.5 | +1.3 | +1.1 | -2.5 | +0.50 |

## Reproducibility and limits

- Evidence directory: `outputs/imatrix-calibration-comparison-v1/native/quality/`.
- Validation recomputed all 60,000 primary task outcomes from saved per-example predictions and checked paired sample identities.
- Fixed native settings: four parallel sequences, eight threads and batch threads, 99 GPU layers.
- F16 was freshly scored with the same runtime; no historical PyTorch baseline was substituted.
- All model hashes match the earlier native comparison. Input, runtime, runner and protocol hashes match across all ten runs.
- Individual JSON files retain per-example predictions, task identities, runtime binary hashes and model hashes.
- `comparison.json` in the evidence directory also records paired correct/incorrect changes for each task and control.
- The existing full Eaddario matrix has a larger collection budget; matched Eaddario is the controlled comparison.
- The original Eaddario data was not retroactively decontaminated. These are retained diagnostic task slices, not an untouched final acceptance set.
- One corpus selection and one deterministic evaluation per model: small differences do not establish a general improvement.
- Native quantization uses input-column importance; this does not test NanoQuant output-row weighting or KL distillation.
- No compression or distillation experiments were launched for this benchmark.

Reproduce from the repository root with `.venv/Scripts/python.exe outputs/imatrix-calibration-comparison-v1/native/quality/run_quality.py`.
Completed evaluations are reused only if their full evaluator identity matches.
