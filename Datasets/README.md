# Reusable planning datasets

## Gemma 3 calibration corpus candidate

`gemma-3-imatrix-calibration-v1/calibration.txt` contains 1,000,294 tokens across
15,449 deduplicated records, with a separate 65,933-token held-out subset.
It mixes prose, instructions, code, math, and multiscript text, uses Gemma's
native user-turn template for prompts, and matches the pinned 270M, 1B, and 4B
tokenizers exactly. Instruction data is prompt-only. Its downstream quality
depends on the quantization format and must be measured.

See [research, provenance, limitations, and usage](../Docs/gemma-calibration-corpus.md).
The directory's `manifest.json` records hashes, sources, mixture counts, and
cached-evaluation overlap filtering. Existing imatrix artifacts are unchanged.

The computed 270M row-bearing matrix is
`gemma-3-270m-it-imatrix-calibration-v1-rows/imatrix.gguf`. Its equal-budget old-corpus
control is `gemma-3-270m-it-imatrix-eaddario-matched-v1-rows/imatrix.gguf`.
Both contain 1,953 contexts of 512 tokens and complete input/output importance
for all 126 projections. See the [matched comparison and NanoQuant run status](../Docs/Experiments/077-078-imatrix-calibration-comparison.md).

## Allocation profiles

`tools/build_allocation_profile_dataset.py` extracts an existing run's measured rank responses and optional KL
sensitivity weights into a compact JSON dataset. It validates the source artifacts and copies the scalar evidence,
so the dataset can be reused after the source run is retired.

The prepared Experiment 070 dataset is `Datasets/gemma-3-1b-it-allocation-070`. It covers all 130 physical units,
including shared QKV groups, with three measured ranks per unit and the complete KL sensitivity profile.

```powershell
.\.venv\Scripts\python.exe tools\build_allocation_profile_dataset.py `
  --run-output evidence\070\070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it `
  --output Datasets\gemma-3-1b-it-allocation-070
```

The builder takes KL sensitivities from the source recipe. Use `--kl-profile PATH` to select a complete compatible
profile explicitly. A run's rank-probe profile must be complete before extraction; compression can still be in
progress. Repeating extraction reuses an identical dataset and rejects replacement with different evidence.

Self-measured D2 launchers and the 070 campaign runner accept the following option.

```powershell
--allocation-profile-dataset Datasets\gemma-3-1b-it-allocation-070
```

This selects approximate allocation guidance and bypasses the uniform control, KL sensitivity measurement and
rank-probe fits. The consumer creates a fresh rank plan under its own bit budget, rank alignment, residual-stage
accounting and charged sidecars. Calibration, production factorization, tuning and quality evaluation continue.
The dataset already contains the KL measurements it consumes. It can also be combined with
`--kl-teacher-dataset`: the workflow validates the teacher artifacts, model/protocol, token hash, dataset fingerprint
and baseline NLL against the packaged KL profile. Teacher inference and KL profiling are still skipped; the teacher
dataset serves as verified source provenance in this mode, rather than supplying global-distillation targets.
`--kl-profile` and `--kl-control-run` remain incompatible with the allocation dataset.

Dataset-backed launchers accept `--dry-run` to validate inputs and print the resolved recipe without compression,
tuning, export or evaluation. Experiment 071 selects both prepared datasets by default.
Launchers also accept `--plan-only` to inspect configuration before datasets exist; this does not validate dataset
contents. [Experiment 072](../Docs/Experiments/072-two-full-rank-stages-gemma-3-4b.md) configures the separate
`gemma-3-4b-it-kl-48x512` teacher dataset and starts with fresh 4B allocation measurements. Its teacher dataset
has not been generated during setup; the document includes the preparation command.

Dataset reuse requires the same pinned model, original projection tensors, matrix shapes and physical grouping.
It permits different calibration tokens, seeds, fitting/tuning settings and bit budgets. For a changed baseline
rank, the existing piecewise exponential curve is re-anchored in absolute rank units. Endpoint slopes extend beyond
the measured interval. Reports mark the input as approximate and list units whose allowed aligned rank range
extends beyond the donor's measurements. New recipes still need quality evaluation to establish whether these
estimates allocate capacity well.

The dataset's content identity participates in the run configuration hash. The resident planner imports the compact
artifact into its own store and records its identity, source recipe, predicted baseline error ratios and rank
decisions. It creates no new measured rank-probe records for dataset predictions. The strict exact-probe reuse path
remains available independently.

YAML/Python callers can select `allocation.reconstruction.response_source: dataset` with `allocation_dataset` and
`allocation_dataset_key`. D2 launchers resolve those fields and the packaged `kl_profile_artifact`/`kl_profile_key`
automatically. The numerical planner version is 67 for this execution path.

Validation replayed Experiment 070's saved allocation inputs in 6–12 seconds on CPU and reproduced all 130
planned ranks and the original 1,395,505,664-bit allocation. A separate tiny-model integration test changes the
calibration inputs, seed and bit budget, completes compression through the dataset path, and checks resume behavior.

## KL teacher targets

### Gemma 3 1B IT weight sensitivity

Run the following from the repository root to measure how much each projection changes
the output distribution when its weights receive random error.

```powershell
.\.venv\Scripts\python.exe tools\measure_block_kl_sensitivity.py
```

The default sweep uses the pinned Gemma 3 1B IT checkpoint loaded in FP16. It tests
each of the seven projections in each of the 26 blocks individually, for 182 targets
and 3,276 trials with three noise seeds at each of six relative Frobenius errors
(1%, 2.5%, 5%, 10%, 20%, and 35%). It samples 16 disjoint
256-token windows from the `content` column of the pinned
[imatrix calibration Parquet](https://huggingface.co/datasets/eaddario/imatrix-calibration/blob/e87ed55dcba9d9c3a3e41539f3e728e981b1daa4/combined_all_small.parquet).
Each trial scales Gaussian noise to the requested relative error in one projection matrix,
then restores its original weights. The rest of the model stays at its original FP16 values.
The results measure sensitivity to this noise distribution. NanoQuant fitting errors can
have a different structure and therefore produce a different KL response at the same norm.
Use `--granularity block` to perturb all seven projections in a block together. Use `--blocks 0 12 25`
to select a subset, and `--error-levels 0.01 0.05 0.10` to measure the response at several
noise levels. The same seed scales the same noise direction across levels for a given matrix.

The first run saves exact token IDs and full-vocabulary FP16 teacher log probabilities
under `Datasets/gemma-3-1b-it-fp16-sensitivity-16x256` (about 2 GiB by default). Add
`--teacher-only` to prepare those targets as a separate step. Use `--teacher-dataset` to
point to the same dataset from other sweeps. Reading either the default Parquet or a custom
`--calibration-parquet PATH --text-column COLUMN` requires PyArrow from the `evaluation` extra.
Choose a separate teacher directory when changing the corpus or sampling protocol.

The sweep checkpoints each trial to `measurements/<sweep-id>/results.json` inside the
teacher dataset directory. The sweep ID identifies the teacher artifact and measurement
settings, so multiple sweeps can share the same teacher. For later fitting, use each
trial's measured weight error after FP16 rounding and its full-vocabulary KL in nats
per next-token position. The dataset also retains the clean control and per-window scores.
Rerun the same command to resume from the last saved trial. A sweep that differs from an
earlier sweep of the same teacher only in its error levels imports that sweep's clean control
and its trials at the shared levels, recording their source in `imported_trials`, and measures
only the new levels. The earlier sweep directory is left unchanged.

On completion or a graceful interruption, the tool exports `results.json` and
`results.md` to `Results/gemma-3-1b-it-weight-sensitivity`. The Markdown ranks targets
by mean KL at each noise level. Pass a different `--output` directory when changing
the sweep settings; a change of error levels alone replaces the exported report.

Fit each projection's KL response from a completed sweep with
`tools/fit_kl_sensitivity.py PATH_TO_MEASUREMENTS_RESULTS_JSON --heatmap-directory PATH`.
The tool writes `kl-response-fits.json` beside the dataset's measurements. Each curve
uses a power law for KL above the clean control and records the seed means, fit residual
metrics, and predictions with each error level held out in turn. Each curve is then
inverted at fixed KL targets above the clean control: 0.001, 0.0025 and 0.005
nats/token by default, or repeat `--kl-target NATS` to choose others. One
`equal-kl-target-error-heatmap-<KL>-nats.png` is rendered per target, all on one colour
scale. Arrows mark extrapolations beyond the measured weight errors. The 1% and 2.5%
levels keep more of the low-KL targets within the measured range. The
targets predict isolated projection effects. Joint perturbations may interact.

### Gemma 3 270M IT weight sensitivity

This launcher builds the teacher dataset, measures each projection, fits the KL
response curves, and renders the measured KL heatmaps, response curves, and
equal-KL target heatmaps.

```powershell
.\.venv\Scripts\python.exe tools\build_gemma_3_270m_sensitivity_dataset.py
```

It uses the pinned `google/gemma-3-270m-it` revision
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` and the 1B calibration and
perturbation defaults. The 270M model runs in FP32 with TF32 disabled. The clean
model overflowed in FP16. In BF16, any weight change added about 0.004 nats/token of
rounding noise, even at 0.01% weight error, which hid the response below about 5% error.
In FP32, KL grows with the square of the weight error down to 0.01%. Cached teacher
log probabilities use FP16, and the clean control is about 3e-7 nats/token.
Its 18 blocks have 126 projection targets, for 2,268
trials at six error levels and three seeds. Install the `evaluation` and
`visualization` extras for Parquet and plotting.

The teacher and measurements go under `Datasets/gemma-3-270m-it-fp32-sensitivity-16x256`.
The earlier BF16 measurements remain under `Datasets/gemma-3-270m-it-bf16-sensitivity-16x256`
for reference; do not use their fits for low-KL targets.
Each `measurements/<sweep-id>/` directory contains the raw `results.json`
and `kl-response-fits.json` with fit quality. Its `reports/` folder contains the
JSON and Markdown reports and the PNGs. Rerun the same command to resume.
Use `--teacher-only` to build just the teacher, or `--local-files-only` when
the pinned model and calibration Parquet are already cached.

`tools/measure_joint_kl_targets.py` perturbs every projection at once to its equal-KL
target error and writes `joint-kl-targets.md` beside the sweep. `tools/measure_gguf_kl.py`
scores llama.cpp GGUF quantizations against the same teacher and tokens and writes
`gguf-kl.md`; NanoQuant GGUFs are rebuilt from their binary-factor sidecars. The compared
files are Experiment 074's `Results/074/gemma-3-270m-it-nanoquant.gguf` and its pre-KD
variant, plus, in `Datasets/gemma-3-270m-it-gguf-unsloth`, `UD-IQ2_XXS`, `Q2_K`, `Q3_K_M`,
`Q4_K_M` and `Q8_0` from `unsloth/gemma-3-270m-it-GGUF` at revision
`c90975dbd40c0c7b275fefaae758c3415c906238`, with sizes and SHA-256 in `source.json`.

### Gemma 3 4B IT weight sensitivity

When ready to generate it, run from the repository root:

```powershell
.\.venv\Scripts\python.exe tools\build_gemma_3_4b_sensitivity_dataset.py --local-files-only
```

This uses the pinned `google/gemma-3-4b-it` revision
`093f9f388b31de276ce2de164bdc2081324b9767` and the same FP16 Gaussian-noise
measurement protocol as the 1B launcher. It loads the text decoder from the
multimodal checkpoint and defaults to 34 blocks, 238 projections, and 4,284
trials (16 × 256-token windows, six error levels, three seeds). Install the
`evaluation` and `visualization` extras for Parquet and plotting. The full
sweep requires many forward passes, a device that fits the FP16 4B decoder,
and about 2 GiB of storage for the default teacher log probabilities alone.

Everything stays under `Datasets/gemma-3-4b-it-fp16-sensitivity-16x256` by
default. Rerun the same command to reuse the teacher and resume the sweep.
Use `--teacher-dataset PATH` for a different dataset directory.

```text
measurements/<sweep-id>/
  results.json
  kl-response-fits.json
  reports/
    results.json
    results.md
    kl-sensitivity-heatmaps.png
    kl-error-response.png
    equal-kl-target-error-heatmap-0.001-nats.png
    equal-kl-target-error-heatmap-0.0025-nats.png
    equal-kl-target-error-heatmap-0.005-nats.png
```

The measured heatmaps and response graph are rendered after all trials finish.
The equal-KL fit and heatmaps require a projection sweep with at least four
positive error levels and invertible response curves. `--teacher-only`
prepares the teacher targets without measurements or plots; `--no-plots`
keeps the measurements and text reports but skips chart generation.
This dataset is distinct from the WikiText KL teacher targets used by 072.

### KL allocation profiling

`tools/build_kl_teacher_dataset.py` prepares the original model's full-vocabulary log probabilities for KL allocation
profiling. Each dataset contains the exact WikiText input tokens, FP16 log probabilities, baseline negative log
likelihood, and the model, tokenizer and execution identities needed to validate reuse.

The default output is `Datasets/<model>-kl-<samples>x<sequence-length>` under the project root. Generated dataset
directories are ignored by Git. Keep separate directories for different revisions or protocols.

## Build once

From the repository root, build the pinned Gemma 3 1B dataset used by Experiment 070.

```powershell
.\.venv\Scripts\python.exe tools\build_kl_teacher_dataset.py `
  --source google/gemma-3-1b-it `
  --revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 `
  --wikitext-samples 48 --sequence-length 512 --batch-size 1 `
  --device cuda --local-files-only
```

Use `--snapshot PATH` to select an existing local model snapshot, `--output PATH` to choose the dataset directory,
and omit `--local-files-only` to allow downloading the pinned model and input dataset. The model ID and pinned
revision are parameters, so the same builder supports other models accepted by the KL evaluator.

To extract already calculated targets, add the existing campaign cache path.

```powershell
  --reuse-cache-root evidence\070\kl-teacher-cache
```

Extraction validates the cache against the requested model, inputs and execution settings. It hard-links immutable
artifacts on the same volume and copies them when linking is unavailable. The resulting dataset has its own
artifact references and remains usable after the original campaign is removed.

Rerunning a completed build validates and reuses it. A committed target cache survives an interruption before final
dataset publication. An interruption during initial teacher inference requires recomputing that uncommitted cache.
A dataset-directory lease prevents concurrent builders from modifying the same output.

## Use in experiments

Self-measured D2 experiment launchers accept the dataset as an execution input.

```powershell
.\.venv\Scripts\python.exe experiments\070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it.py `
  --kl-teacher-dataset Datasets\gemma-3-1b-it-kl-48x512
```

Experiment 070 retains its block-bounded execution and resume behavior. This option supplies the KL teacher when
the campaign reaches sensitivity profiling.

The campaign runner forwards the same option to every process slice.

```powershell
.\.venv\Scripts\python.exe tools\run_experiment070_campaign.py `
  --kl-teacher-dataset Datasets\gemma-3-1b-it-kl-48x512
```

For direct profiling, add `--teacher-dataset Datasets\gemma-3-1b-it-kl-48x512` to `tools/profile_kl_budget.py` or
`nanoquant kl-budget`, with the same `--wikitext-samples`, `--sequence-length`, `--batch-size` and device settings.
The dataset supplies the saved inputs directly, so consumption requires no WikiText download or tokenization.
The output receipt records the dataset artifact identity and target-cache reuse.

## Reuse contract and storage

The dataset uses version-1 metadata and content-addressed safetensors artifacts. The reader validates hashes,
token IDs, tensor shapes, FP16 target storage, model revision/configuration, tokenizer content, model dtype,
attention implementation, device and batching. `cuda` and `cuda:0` identify the same device. Incompatible datasets
raise an error before sensitivity evaluation. An explicitly selected dataset is never silently regenerated.

The 48-by-512 Gemma dataset has 12 GiB of target tensors because it retains all 262,144 vocabulary entries. The
same FP16 targets and baseline calculation are used by the existing KL evaluator, preserving its numerical
protocol. Compact top-k distillation targets and generated conversational responses have their own formats and
consumers.

Reuse removes teacher-target calculation from KL profiling. Control compression, compressed-model sensitivity
arms, block tuning and global distillation remain separate computations. The model shell is still loaded to
evaluate the spliced compressed projections.

The final `dataset.json` pointer is published after its referenced inputs and targets are committed. Keep the
complete dataset directory together when moving it between workspaces.

## External llama.cpp importance data

To generate local importance data from a Hugging Face model and Parquet corpus,
use `tools/build_imatrix_dataset.py`. The first default is Gemma 3 270M IT with
the complete `eaddario/imatrix-calibration/combined_all_huge.parquet` corpus,
stored in `Datasets/gemma-3-270m-it-imatrix-eaddario-huge/` beside a hash-pinned
`source.json` completion receipt. See [building imatrix datasets](../Docs/imatrix-datasets.md)
for preparation, generation, options and completion checks.

Experiment 073 uses `Datasets/gemma-3-1b-it-imatrix-unsloth/imatrix_unsloth.dat`,
downloaded from `unsloth/gemma-3-1b-it-GGUF` at revision
`f0b45be0aac41bd6a100a4b5734cad5f67255bfb`. Its SHA-256 is
`86a81192de49a6480f8dc19c5a7fd3f4377af59172fd4c7b8d93d296a0a18515`.
The neighboring `source.json` records the exact URL, revision, size and hash.
All 182 input-importance vectors match the pinned 1B model's projection inventory.
This is an external activation-importance input, separate from allocation profiles
and KL teacher targets. See [Experiment 073](../Docs/Experiments/073-imatrix-two-full-rank-stages-gemma-3-1b.md)
for the prepared recipe and launch command.

Experiment 074 uses `Datasets/gemma-3-270m-it-imatrix-unsloth/imatrix_unsloth.gguf_file`,
downloaded from `unsloth/gemma-3-270m-it-GGUF` at revision
`c90975dbd40c0c7b275fefaae758c3415c906238`. It is a GGUF-format imatrix with SHA-256
`2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3`; `source.json`
beside it records the URL, revision, size and hash. All 126 vectors match the pinned
`unsloth/gemma-3-270m-it` projection inventory. See
[Experiment 074](../Docs/Experiments/074-fast-imatrix-two-full-rank-stages-gemma-3-270m.md).
