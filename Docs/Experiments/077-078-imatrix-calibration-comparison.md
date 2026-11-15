# Gemma 3 270M calibration comparison

## Scope and artifacts

The curated corpus is `Datasets/gemma-3-imatrix-calibration-v1/calibration.txt`.
Both new importance matrices use `unsloth/gemma-3-270m-it` revision
`23cf460f6bb16954176b3ddcc8d4f250501458a9`, the same F16 GGUF and
`llama-imatrix` executable, 512-token contexts, 1,953 chunks, and special-token
parsing. Each projection has exactly 999,936 observations on both axes.
Production inventory validation covers all 126 input and output vectors across
18 blocks, with 112,896 input channels and 124,416 output rows.

| Matrix | Local directory under `Datasets/` | SHA-256 |
| --- | --- | --- |
| Curated | `gemma-3-270m-it-imatrix-calibration-v1-rows` | `e1e271f6a1b35dea9a372713e7ab403094ee848b174648b5dd679a70a44b7283` |
| Equal-budget Eaddario | `gemma-3-270m-it-imatrix-eaddario-matched-v1-rows` | `763e0f0ae837362155da1864eb18230cc9a4f0a6eebcec069ac59014ec1e7931` |
| Existing full Eaddario | `gemma-3-270m-it-imatrix-eaddario-huge-rows` | `3deceafea3aadd5fb6b75e22dfa0123bb99f22e0be95816354b374e80ebcc0c9` |

Each directory contains `imatrix.gguf`, `source.json`, and `validation.json`.
The equal-budget Eaddario input is the unchanged first 20,000 complete lines
of the original corpus; only its first 1,953 contexts are consumed. The existing
full matrix used 13,301 contexts (6,810,112 token slots). It is a useful retained
baseline but has a larger collection budget. This is a single deterministic
corpus selection, not a multi-seed study.

The new corpus has explicit Gemma boundaries. llama.cpp adds its normal leading
BOS, so its warning about a doubled initial BOS is retained in the collection
log. The observations above are the actual GGUF counts, not estimated HF token
counts. The same native executable and parse-special setting are used in both
fresh collections.

## Native GGUF comparison

The native comparison measures Q2_K, Q4_K_M, and IQ4_NL quantization from the
same F16 source, with eight quantization threads and no additional tensor-type
overrides. All arms must report 126 importance entries in use. The native
quantizer consumes input-column importance; it ignores the output-row fields.
This comparison does not measure NanoQuant's row-weighted factorization or KD.

### Primary quality comparison: six task benchmarks

On 2026-11-15, all nine existing quantized GGUFs and the F16 reference were
freshly evaluated using the established PIQA, ARC Easy, ARC Challenge,
HellaSwag, WinoGrande, and BoolQ protocol: the first 1,000 examples per task,
with the pinned prompts, splits, and primary accuracy metrics. Every model
used the same native runtime and prepared token IDs. No compression or
distillation experiment was launched for this evaluation.

The table shows the unweighted mean of the six task scores, in percent.
F16 scored **50.20%**. Higher is better; individual task results are in the
[full task-quality report](gemma-3-270m-imatrix-task-quality.md).

| Native format | Curated corpus | Equal-budget Eaddario | Existing full Eaddario | Curated minus equal-budget |
| --- | ---: | ---: | ---: | ---: |
| Q2_K | 47.92% | 47.67% | 48.02% | +0.25 pp |
| Q4_K_M | 47.77% | 46.70% | 46.72% | +1.07 pp |
| IQ4_NL | 48.18% | 48.12% | 47.68% | +0.07 pp |

Task accuracy gives a different ranking from perplexity. The clearest mean
gain from curated calibration is Q4_K_M, including gains on four tasks against
the equal-budget control, one tie, and a WinoGrande decline. Q2_K's small mean
gain over equal-budget Eaddario is driven mainly by BoolQ; four tasks decline.
IQ4_NL's equal-budget mean is nearly unchanged, with gains on three tasks and
losses on three. These fixed diagnostic slices and one corpus selection do not
establish a universal calibration improvement.

All 60,000 per-example task outcomes were checked against the reported scores.
Model hashes, input hashes, runtime hashes, and protocol hashes were verified;
each model used four parallel sequences, eight threads and batch threads, and
99 GPU layers. Per-model JSON, paired outcome counts, commands, evaluator
source hashes, and the resumable runner are under
`outputs/imatrix-calibration-comparison-v1/native/quality/`.

### Historical perplexity diagnostic

The measurements below are retained diagnostic evidence. Use the task results
above for the quality comparison rather than inferring task performance from PPL.

Evaluation uses the complete cached WikiText-2 raw test split at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, joining all 4,358 text rows with
double newlines. UTF-8 text SHA-256:
`696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`.
All models use native llama.cpp perplexity, context/batch/microbatch 512,
eight threads, 99 GPU layers, seed 0, default perplexity stride 0, and all
complete contexts. There are 570 contexts and 145,350 scored next-token targets
(255 per context); the final partial context is discarded.

Detailed commands, executable and DLL hashes, model hashes, per-context
cumulative NLL, timings, and comparison results are retained in
`outputs/imatrix-calibration-comparison-v1/native/`. This native protocol is
different from the shorter NanoQuant quality protocol; their absolute PPL
numbers must not be compared directly. WikiText is retained diagnostic evidence,
not an untouched final acceptance set. No task-accuracy claim follows from PPL.

#### Perplexity results on 2026-11-15

Lower perplexity is better. The unquantized F16 reference scored **64.0052**.

| Native format | Curated corpus | Equal-budget Eaddario | Existing full Eaddario | New vs equal-budget | New vs existing full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2_K | 78.6100 | 85.3799 | 86.3055 | -7.93% | -8.92% |
| Q4_K_M | 67.4555 | 66.7665 | 66.5674 | +1.03% | +1.33% |
| IQ4_NL | 65.9993 | 62.7732 | 63.1254 | +5.14% | +4.55% |

The new corpus improves Q2_K perplexity but worsens perplexity for both four-bit
presets. The subsequent task benchmarks above show why this alone is not a
quality conclusion. Corpus mix and formatting changed together; this experiment
does not isolate the cause. No multiple-seed uncertainty was measured. The
original Eaddario control was
preserved as-is; it was not retroactively decontaminated using the new corpus's
evaluation exclusion filter.

Q2_K and Q4_K_M each require fallback types for 90 of 236 stored tensors because
640-wide projections do not fit 256-element K-quant blocks. IQ4_NL's main
32-element format fits these widths, but its preset still requests K-quant
formats for the embedding and 18 value projections, causing 19 fallbacks
(embedding to Q8_0 and value projections to Q5_1). The presets retain
higher-precision tensors, so their names are not whole-model bit rates:
effective stored tensor BPW is 6.880, 7.358, and 7.025 respectively.
The local JSON records actual storage types and bytes. The full reference report is
`outputs/imatrix-calibration-comparison-v1/native/comparison.md`, with machine-readable
measurements and provenance in the adjacent `comparison.json`.

## NanoQuant Experiments 077 and 078

The two launchers retain the frozen Experiment 075 recipe. Their numerical
configs differ only in the imatrix path and SHA-256; output names and publication
paths differ as expected. Both use output-row importance. The supported
`materialize_pinned_calibration` function validated and materialized 075's exact
256-by-2,048 fitting tokens into each run's own artifact store and receipt.
Calibration fingerprint:
`sha256:edd62d43b7ec77efcdcf8be890d46c32d478c9aa42dfcc04824147256e387833`.

The first restricted-environment attempt stopped before block processing with
`VRAM002 WDDM shared-memory meter is unavailable after CUDA initialization`.
The required Windows GPU Process Memory counter could not be opened:
`PdhAddEnglishCounterW` returned `0xC0000BC8`; an independent `Get-Counter`
probe also found no available counter set. `nvidia-smi` cannot replace this
per-process shared-memory measurement. After access changed, a real CUDA
preflight successfully measured shared memory with the unchanged 0.75 GiB guard.
077 was then started, but the user clarified that this request was **setup only**.
Its parent and child workers were stopped immediately during online Fisher
calibration (last durable progress event: 107/256). There is no block journal or
compressed checkpoint; only the pinned fitting tokens and measured batch-size
selection (8) are reusable. 078 has not been launched. Neither experiment has a
compressed NanoQuant model or completed NanoQuant quality result. No experiment
was restarted after the setup-only instruction.

### Configured pre/post export and quality flow

Both numbered launchers now use `experiment_pre_post_callable_main`. A future
explicit launch performs the usual complete compression and post-KD benchmark,
then invokes the supported pre-KD backfill on the same immutable block commits.
This reconstructs the weights **before global KL distillation**; it does not
remove the recipe's earlier per-layer and per-block tuning.

Before distillation, the opted-in resident workflow runs the fresh
`validate_resident_run.py --require-complete` audit with the actual block count.
Audit failure prevents distillation. The pre-KD backfill independently performs
a fresh complete-run audit before export and evaluation. These are operational
checks; the numerical config and fitting data remain unchanged.

| Experiment | Pre-KD GGUF | Post-KD GGUF |
| --- | --- | --- |
| 077 | `Results/077/gemma-3-270m-it-nanoquant-pre-kd.gguf` | `Results/077/gemma-3-270m-it-nanoquant.gguf` |
| 078 | `Results/078/gemma-3-270m-it-nanoquant-pre-kd.gguf` | `Results/078/gemma-3-270m-it-nanoquant.gguf` |

Each run also gets separate `-pre-kd-quality.json`/`.md` and `-quality.json`/`.md`
reports. Pre-KD exports use distinct logical, packed, and checkpoint paths under
`outputs/NNN/pre-kd`. The pre-KD Markdown includes the pre/post comparison table.
Quality is the standard **dense packed-model reference backend**, not a native
GGUF runtime benchmark: WikiText-2 64 by 128 tokens (batch 8), and PIQA, ARC Easy,
ARC Challenge, HellaSwag, WinoGrande, BoolQ at 1,000 examples each (batch 4).
Pre-KD reuses its matching post-KD BF16 result and verifies protocol equality.

`--plan-only` and `--dry-run` include both output plans in one JSON object and do
not run models, export files, or benchmarks. Importing a launcher also runs
nothing. A post-KD failure prevents the pre-KD phase. A later retry validates and
reuses existing export artifacts and explicitly refreshes the pre-KD quality
report against the current post-KD report; stale scores are not silently reused.
The standalone backfill still refuses an existing report unless its explicit
`--refresh-quality` option is supplied.

For a later explicitly requested run, launch only where the counter works, after
checking the GPU and device lease. The validated fitting inputs are already
run-local. Set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and
`HF_DATASETS_CACHE=outputs/imatrix-calibration-comparison-v1/hf-datasets` (absolute
path) to use the verified writable copy of cached evaluation data. Use writable
workspace paths for `TEMP`, `TMP`, `TRITON_CACHE_DIR`, and
`TORCHINDUCTOR_CACHE_DIR`. Run 077 and 078 sequentially. Freshly rescore 075 with
the same current evaluator before comparing its older quality report: unrelated
evaluation-code changes were already present when this task started.

Safe setup previews (these were validated without model execution):

```powershell
.\.venv\Scripts\python.exe experiments/077-curated-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --plan-only
.\.venv\Scripts\python.exe experiments/078-matched-eaddario-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-270m-it.py --plan-only
```

Only when ready to execute, remove `--plan-only`. Each launcher then performs
both snapshot exports and both quality benchmarks; run the launchers sequentially.

Both launcher previews and their resolved config comparison passed.
All 23 corpus/imatrix builder tests passed; repository-wide Ruff and mypy passed.
The setup adds orchestration and an opt-in audit; quantization math is unchanged.

Pre/post setup validation: 50 focused tests passed, followed by the full CPU
suite with **1,820 passed, 2 skipped, 54 deselected**. CUDA was hidden during
testing; no real-model experiment ran. Repository-wide Ruff and mypy (228 files)
passed. The full log is
`outputs/imatrix-calibration-comparison-v1/pre-post-setup-tests.log`.
