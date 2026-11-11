# Reusable KL teacher datasets

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
