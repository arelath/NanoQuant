# Curated sensitivity comparison after Experiment 083

Measure whether the sensitivity estimates and joint ranking change with a larger,
curated text sample. Use the existing curated Gemma IMatrix calibration corpus
at `Datasets/gemma-3-imatrix-calibration-v1/calibration.txt`, with SHA-256
`3d8fa1e99d56ca7ba8455a1aabd5b6d10a82ca1a7d502f7fb15450cba290e301`.
Its manifest records 1,000,294 tokens across 15,449 records. Tokenization through
the pinned Gemma tokenizer preserves the existing native turn and BOS/EOS tokens.

Sample 160 disjoint windows, each consisting of a BOS and 255 corpus tokens,
with seed zero. This gives 40,800 scored positions per perturbation, exactly
ten times the prior 4,080. Windows can cross record boundaries, matching the
existing stream-window protocol. The new teacher cache has its own dataset
identity and approximately 20 GiB of retained FP16 log probabilities.

First compare joint decoder KL for the existing 080, 082 and 083 exports on the
curated 160 windows and a nested 16-window subset selected with seed 83.
Compare their rankings with the saved original-corpus results.

Then run the full sensitivity sweep with all 182 individual projections,
relative Gaussian noise levels 0.01, 0.025, 0.05, 0.10, 0.20 and 0.35, and
seeds 0, 1 and 2. All 3,276 trials persist per-window KL. Fit the usual
power-law curves to the 160-window means and to the nested 16-window means.
This tests sample-size stability within the curated corpus and provides a
comparison with the original 16-window corpus. The sample-size and corpus
changes are both present when comparing the old sweep to the new full sweep.

The model, precision, noise generation, embeddings, arithmetic and 256-token
context length follow the prior sensitivity protocol. Corpus provenance and
coverage limitations are documented in [the corpus report](../gemma-calibration-corpus.md).

```powershell
.\.venv\Scripts\python.exe tools/build_gemma_3_1b_sensitivity_dataset.py --calibration-text Datasets/gemma-3-imatrix-calibration-v1/calibration.txt --samples 160 --sequence-length 256 --teacher-dataset Datasets/gemma-3-1b-it-fp16-sensitivity-curated-160x256 --local-files-only
```

Reuse this invocation to resume completed noise trials. The full sweep can take
many hours. Logs are under `outputs/curated-sensitivity-160x256`.

## Initial joint-model comparison

The curated teacher dataset completed on 2026-11-17. Joint decoder KL is measured
with clean embeddings on the same windows for all three existing exports.

| Experiment | Original corpus, 16 windows | Curated nested 16 | Curated 160 |
| --- | ---: | ---: | ---: |
| 080 | 1.369504 | 0.764553 | 0.747412 |
| 082 | 1.413458 | 0.756072 | 0.761960 |
| 083 | 1.379170 | 0.770809 | 0.786708 |

080 leads on the full curated sample. Its ranking relative to 082 reverses in
the nested 16-window subset. On 160 windows, 083 minus 080 is +0.039297
nats/token, with a paired window standard error of 0.007966. Absolute KL values
across the original and curated corpora reflect their different token distributions.
These windows provide within-corpus variation, not independent corpus replications.
The [result JSON](../../Results/083/083-curated-160-joint-kl.json) retains every window.
The 1B sweep was stopped at the user's request after 418 of 3,276 completed
trials. Its checkpoints and teacher dataset are preserved for resume.

## 270M first

The active replacement uses pinned `google/gemma-3-270m-it` revision
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`, the same curated text,
160 × 256 windows and seeds 0, 1, 2. Omit the 35% noise level. The five levels
1%, 2.5%, 5%, 10%, 20% give 1,890 trials across 126 individual projections.
Use the existing FP32 sensitivity recipe for 270M, which avoids its FP16
overflow and BF16 small-error rounding floor.

The 270M tokenizer recognizes the native Gemma control tokens and produces
1,000,294 corpus tokens. Input validation is retained under
`outputs/curated-sensitivity-270m-160x256/input-validation.json`.

```powershell
.\.venv\Scripts\python.exe tools/build_gemma_3_270m_sensitivity_dataset.py --calibration-text Datasets/gemma-3-imatrix-calibration-v1/calibration.txt --samples 160 --sequence-length 256 --calibration-seed 0 --teacher-dataset Datasets/gemma-3-270m-it-fp32-sensitivity-curated-160x256 --error-levels 0.01 0.025 0.05 0.10 0.20 --seeds 0 1 2 --local-files-only
```

Logs are under `outputs/curated-sensitivity-270m-160x256`. After the sweep,
the queued sampling analysis compares nested 16 versus all 160 curated
windows and the retained complete 270M FP32 sensitivity sweep.

## Scoring performance

The teacher cache occupies 20 GiB in host memory. The initial scorer transfers
each 128 MiB window to CUDA for every trial. An eight-window profile measured
0.462 seconds of forward computation, 0.366 seconds of teacher transfer and
0.155 seconds of KL reduction.

The active 270M sweep now uses `--teacher-gpu-cache-gib 6`. This retains
48 windows in CUDA memory and pins the remaining 112 host windows. A copy
stream prefetches one window while the current window executes. Pinned host
copies require approximately 14 GiB of additional host memory during setup.
Peak allocated CUDA memory in the validation was approximately 7.7 GiB.

Three paired full-160-window perturbed-model scores took 8.70/6.75,
15.96/8.83 and 16.56/11.31 seconds before/after caching. Total scoring time
fell 35%, a 1.53× speedup across these pairs. Every per-window KL, NLL and
aggregate score matched exactly, including the saved clean baseline. The
run resumes the existing measurement identity after 21 committed trials.
Detailed timing receipts are under
`outputs/curated-sensitivity-270m-160x256/scoring-benchmark-repeat.json`.
The active log is `sweep-gpu-cache.log` in that same directory.

The four-pass scorer (`--teacher-passes 4 --teacher-gpu-cache-gib 6`) holds
40 teacher windows (5 GiB) in CUDA at a time and evaluates every outstanding
perturbation before advancing to the next partition. Per-pass results checkpoint
under `measurements/<sweep-id>/passes-4`; the standard complete trials merge
after all partitions finish, using the original ordered per-window reductions.
An interruption resumes completed partial scores. Existing complete trials are
retained and skipped.

On three 270M perturbations, four passes took 6.3–6.7 seconds of scoring per full
160-window trial. Uploading all four partitions took 7.4 seconds, amortized
across the sweep. Full per-window records and aggregate scores matched the
saved batch-one measurements exactly. Peak allocated CUDA memory was 6.45 GiB.
The CPU regression also verifies uneven partitions, partial-pass interruption,
resume and exact combined results.

## Batched forwards

The active run uses four resident teacher partitions with
`--evaluation-batch-size 8`. A benchmark of batches 1, 2, 4 and 8 on a
40-window resident partition found batch 8 fastest at approximately 1.04 seconds,
versus 2.1–2.4 seconds for batch 1. Peak allocated CUDA memory was 9.2 GiB.
Measurements covered the clean model and 1% and 20% perturbations.

Batched FP32 matrix products introduce small rounding changes. The largest
aggregate KL shift in these tests was about 3.1e-8 nats/token. Consequently
batch 8 has a distinct measurement identity containing its evaluation batch size
and scorer version. Earlier batch-one complete and partial measurements remain
in their original checkpoint directory. The teacher dataset is shared.
Pass boundaries align to forward batches so partitioning preserves the same
batched forward shapes and window ordering.

The active log is
`outputs/curated-sensitivity-270m-160x256/sweep-batch8-four-pass.log`.
Resume with the same 270M command, appending
`--teacher-gpu-cache-gib 6 --teacher-passes 4 --evaluation-batch-size 8`.
