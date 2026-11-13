# Experiment 072: two full-rank stages on Gemma 3 4B IT

Prepared 2026-11-12. The uniform control reached 12 of 34 committed blocks before a CUDA OOM on resume on 2026-11-13.

## Recipe

Transfer Experiment 071 to `google/gemma-3-4b-it`, pinned at
`093f9f388b31de276ce2de164bdc2081324b9767`. The pinned checkpoint is already cached locally.
The parent configuration is frozen in `experiments/recipes/references/experiment071.json`, with semantic hash
`sha256:2ae172330b1e6d7db5b1ff878a18f3c5b0a169706426a02088c8f9c9fbeb25f5`.

Preserved from 071: exactly two independently scaled full-rank stages per quantization owner, stacked QKV,
600 outer × 2 inner ADMM iterations, optional PyTorch compilation enabled, calibration, tuning epochs and learning
rates, global distillation, no residual columns, and a 1,000-example task limit. Maximum-rank promotion follows
allocation and may exceed the nominal 2-BPW allocation target. Embeddings/output heads are not given these stages.

The established 4B template supplies the model/revision and bounded-memory batch settings: nonfactorized tuning 4,
factorized tuning 1, post-block refit 1, microbatch 1, and resident forward batch 4. Inline quality is disabled;
final quality uses the existing 4B dense backend. Other 071 recipe settings, including its retry thresholds, remain
inherited. These batch/backend changes mean this is not an identical-execution comparison with 071.

The text decoder has 34 blocks, with 170 quantization owners and 340 independent stages:

| Owner | Matrix shape | Primary rank | Residual rank |
| --- | --- | ---: | ---: |
| MLP gate / up | 10,240 × 2,560 | 2,560 | 2,560 |
| MLP down | 2,560 × 10,240 | 2,560 | 2,560 |
| Stacked QKV | 4,096 × 2,560 | 2,560 | 2,560 |
| Attention output | 2,560 × 2,048 | 2,048 | 2,048 |

These are planned ranks derived from the pinned text configuration, not completed fit results. The vision tower
is not quantized by this experiment. The standard complete export workflow also exports the separate validated
BF16 `mmproj-BF16.gguf` required by the multimodal source.

## Teacher and allocation preparation

072 selects `Datasets/gemma-3-4b-it-kl-48x512` by default. It holds full-vocabulary FP16 teacher log probabilities
for 48 WikiText sequences of length 512, batch size 1, on CUDA. The targets alone occupy approximately 12 GiB.
The source BF16 4B model is the teacher; no separately trained teacher is needed.

The 1B teacher and allocation datasets are incompatible. No 4B allocation dataset exists yet, so the first campaign
uses the established self-measured D2 sequence: a matched tuned control, KL sensitivity measurement using saved
teacher targets, measured rank-response probes, and final compression/export/quality evaluation. Both compression
passes use the two-full-rank policy. This is more preparation than 071's dataset-backed run. The allocation curves
cannot change the final promoted ranks; retaining this preparation produces reusable 4B allocation evidence without
introducing a new allocation bypass. The inherited rank-response probes remain 400×5; production fits are 600×2.

The 48×512 KL dataset does not replace global distillation's separate teacher protocol. That stage remains inherited.
The KL builder and consumer now resolve vocabulary defaults and decoder depth through the model adapter, including
Gemma 3's nested text configuration. Dataset identity still hashes the original source config.

## Commands (run later)

Inspect setup only, including before the teacher dataset exists:

```powershell
.\.venv\Scripts\python.exe tools\run_experiment072_campaign.py --plan-only
```

This validates configuration and prints paths; it does not load or validate dataset contents, load model weights,
generate targets, or start a campaign. The existing `--dry-run` is stricter and requires a prepared allocation dataset.

Build the separate teacher dataset when ready, from the repository root:

```powershell
.\.venv\Scripts\python.exe tools\build_kl_teacher_dataset.py `
  --source google/gemma-3-4b-it `
  --revision 093f9f388b31de276ce2de164bdc2081324b9767 `
  --output Datasets/gemma-3-4b-it-kl-48x512 `
  --wikitext-samples 48 --sequence-length 512 --batch-size 1 `
  --device cuda --local-files-only
```

Then start the campaign:

```powershell
.\.venv\Scripts\python.exe tools\run_experiment072_campaign.py
```

The runner permits 80 process slices by default, enough for two 34-block passes plus finalization. It preserves
one-new-block interruptions and durable resume; `--maximum-slices` can raise the ceiling after additional restarts.
`--kl-teacher-dataset` can override the teacher location. A later compatible 4B allocation dataset can be passed with
`--allocation-profile-dataset` to skip preparation. Missing or incompatible teacher targets fail before control fitting.

Both passes resume from the latest committed activation checkpoint with `restore_completed_blocks=False`.
This keeps previously fitted weights on disk during compression and preserves the existing commit identity.
The control OOM occurred while reloading the frozen prefix onto the 12 GiB GPU. Rerun the same campaign command
to continue from the saved checkpoint. The export and final quality workflows load the completed artifacts separately.

Once the measured rank-probe profile is complete, extract it for future reuse:

```powershell
.\.venv\Scripts\python.exe tools\build_allocation_profile_dataset.py `
  --run-output evidence/072/072-two-full-rank-stages-compress-and-benchmark-gemma-3-4b-it `
  --output Datasets/gemma-3-4b-it-allocation-072
```

Future artifacts belong under `evidence/072`, `outputs/072`, and `Results/072`. The experiment links the older 4B
Experiment 018 as historical context; use 072's own protocol-matched BF16 quality result for quantization loss.
018 has different settings and is not a controlled one-variable comparison. No quality or speedup is claimed yet.

## Setup validation

- The real `--plan-only` command passed; its output is `outputs/072-setup-plan.json`.
- CPU tests check recipe inheritance, two-stage planner promotion, no-compute setup behavior, and the 80-slice runner.
- Tiny CPU teacher tests cover both flat and nested configurations, dataset reuse, and matching KL consumer results.
- Model-size defaults were checked from the pinned local config without loading weights. A safetensors-header check
  verified all 170 owner shapes and 340 planned stages; see `outputs/072-rank-shape-check.json`.
- Full default test suite: **1,618 passed, 53 deselected** (CUDA/external-data/performance tests excluded).
  Repository-wide Ruff and mypy's 224-file source check passed.

Full-model memory use, compiled execution, compression quality and export/runtime acceptance remain unmeasured.
