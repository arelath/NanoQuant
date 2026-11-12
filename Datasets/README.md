# Reusable planning datasets

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
