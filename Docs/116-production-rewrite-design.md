# 116 — Production rewrite design: NanoQuant from Experiment 085

Status: design proposal, 2026-11-18. Nothing here has been implemented. The implementation checklist is
[117-production-rewrite-task-list.md](117-production-rewrite-task-list.md).

## 1. Summary

Experiment 085 is the first result worth turning into a product. It compresses Gemma 3 1B IT to a 547 MB
GGUF: 2.515 bits per decoder projection weight, with Q8_0 embeddings. Its scores:

- joint decoder KL: 0.726
- WikiText-2 PPL: 124.6
- six-task mean: 56.1%
- MMLU: 31.8%

This repository produced that result. It also holds two years of unadopted alternatives, and the
machinery built to keep those alternatives reproducible.

Measured today:

| What | Size |
| --- | ---: |
| `src/nanoquant` | 225 files, 80,587 lines |
| `tests` | 356 files, 51,667 lines |
| `tools` | 228 scripts, 67,450 lines |
| `Docs` | 266 markdown files |
| Disk under `Datasets/ evidence/ outputs/ Results/` | ≈ 450 GB (91 + 291 + 45 + 21) |
| One 085 run directory (`evidence/085/...`) | 34 GB, 17 GB of it content-addressed artifacts |
| Intermediate export formats for one model | 4 (logical 6.9 GB → packed → llama.cpp checkpoint → GGUF) |
| The inputs 085's allocation actually consumes | < 15 MB of JSON plus a 1.4 MB imatrix |
| Block commit phase in the 085 profile | 1,043 s of 13,799 s (7.6%) |

The proposal is a new, small package. It implements exactly the 085 method, under these rules:

- no content hashing;
- plain named directories;
- one compressed checkpoint format;
- recompute activations instead of storing them;
- one CLI.

The target size is about 10k source lines and 3k test lines. The current repository stays as the
research playground and the parity reference.

One finding shapes the whole design: **085's result needs both its curated rank allocation and an ADMM
schedule strong enough for the ranks that allocation produces.** Same byte budget throughout:

| Run | Allocation | ADMM | Joint decoder KL | WikiText-2 | Six-task | MMLU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 080 | uniform: every unit at two full-rank stages | cubic 600 | 0.7473 | 130.57 | 55.55% | 32.08% |
| 080 | uniform | linear 800 | no change from cubic 600 (user-reported; no file in the repo) | | | |
| 085 original (`Results/085/cubic-600`) | curated KL options | cubic 600 | 0.7467 | 127.65 | 55.48% | 30.43% |
| 085 | curated KL options | **linear 800** | **0.7263** | 124.59 | 56.08% | 31.83% |

Cubic 600 fits uniform two-stage ranks well. The curated allocation moves capacity into some units: up
to three full stages, and as low as 0.25× uniform. At those ranks cubic 600 underfits by about 5%, and that
fitting loss hid the allocation's benefit. Linear 800 removes it; on 080 the same change did nothing.

So the allocation cannot be skipped, and the ADMM settings cannot be chosen apart from it. Three
consequences:

1. **The profiling pipeline is first-class in v1** (§8). For a new model it includes:
   - a 5.5 h noise sweep;
   - rank-response probes;
   - a pilot compression;
   - isolated-KL anchors.

   Measured anchors cannot be dropped: with fit-only anchors, 102 of 130 units change rank (§8.3).
2. **ADMM settings must be validated over the whole rank range the allocator may choose**, not at uniform
   ranks (§8.4).
3. **Rank-response curves must use the production ADMM settings.** The 085 curves came from 080's
   cubic-600 probes (`measurement-manifest.json`), so they include the same underfitting they then priced.

## 2. What Experiment 085 actually is

This is the method the rewrite must preserve. It combines the `experiment073.json` reference config with
the deltas in `experiments/085-*.py`. Every value below was read from those files, not from memory.

### 2.1 Pipeline

```
HF snapshot (gemma-3-1b-it @ dcc83ea8…)
   │
   ├─ calibration tokens: 256 × 2048, 0.5 UltraChat train_sft + 0.5 WikiText-2 raw train,
   │    gemma-chat-plus-raw-text-v1 formatting, selection seed 0
   ├─ input importance: Unsloth llama.cpp imatrix (.dat, legacy sums/counts), per input column
   ├─ output importance: online Fisher diagonal on the calibration tokens, shrinkage 0.6,
   │    ≤ 4096 tokens/layer (the Unsloth imatrix has no row entries, so this fallback is always taken —
   │    resident_quantization.py:7472)
   │
   ├─ allocation profile (per model, built once; see §8):
   │    rank-response curves: imatrix-weighted NMSE vs rank, per unit (130 units: qkv, o, gate, up, down × 26)
   │    sensitivity fits: KL = c · error^p per projection, from a noise sweep on curated 160×256 teacher windows
   │    anchors: measured isolated KL of each projection of a reference compressed model
   │    options: KL(rank) = anchor_kl · (err(rank)/err(anchor_rank))^p, ranks aligned to 32,
   │             floor 0.25× / ceiling 3.0× uniform, ≤ 3 full-rank stages, in sweep range only
   │    plan: integer knapsack, minimise Σ KL subject to Σ factor bytes ≤ 219,368,448
   │
   ├─ block loop (26 blocks), per block:
   │    FP teacher stream: target = FP block(FP prefix activations)
   │    student stream: input = compressed prefix activations
   │    for unit in [gate, up, down, qkv, o]:
   │        non-factorized tuning of the still-dense layers (epochs by position 8,4,3,6,2; AdamW 1e-4)
   │        factorize with additive residual stages (ADMM 800 outer, 2 inner, reg 0.03, linear penalty ramp)
   │            + scale fit (2 alternating passes, rollback on regression)
   │        factorized tuning (STE on latent signs; 8 epochs; lr binary 3e-5, scale 1e-5)
   │        freeze
   │    post-block refit of scales (2 epochs, lr 1e-5)
   │    propagate both streams
   │
   ├─ export: packed binary factors + BF16 scales; Q8_0 token embedding; GGUF for modified llama.cpp
   └─ evaluate: WikiText-2 PPL (64×128), six tasks (first 1000 rows), MMLU 5-shot full, joint decoder KL
```

### 2.2 Settings that must carry over

| Area | Setting | 085 value |
| --- | --- | --- |
| Model | load dtype / seq len | bfloat16 / 2048 |
| Units | shared input group | `attn_qkv` = q, k, v with v multiplier 2.0 |
| Units | residual stages | on; each stage rank ≤ min(out, in) aligned to 32 |
| Factor format | per stage | U ∈ {±1}^{out×r}, V ∈ {±1}^{in×r}, BF16 `scale_pre`(in), `scale_mid`(r), `scale_post`(out) |
| Factor cost | bits | r·(out+in) sign bits + 16·(out+in+r) scale bits per stage |
| ADMM | outer / inner / reg / schedule / transpose_wide | 800 / 2 / 0.03 / linear / false |
| ADMM | compute / solve dtype | bfloat16 / float32 |
| Scale fit | passes / chunk rows / epsilon | 2 / 512 / 1e-8, rollback on regression |
| Outliers, bias, patches, codebooks | — | all disabled |
| Tuning order | unit positions | gate, up, down, qkv, o |
| Non-factorized tuning | epochs by position / batch / optimizer | 8,4,3,6,2 / 4 / AdamW lr 1e-4, wd 0 |
| Factorized tuning | epochs / batch / lr | 8 / 1 / binary 3e-5, scale 1e-5, bias 1e-5 |
| Factorized tuning | optimizer | `fused_adamw_scales_fp32` for gate/down/o from block 0; `parity` for up and qkv (inherited from 073) |
| Post-block refit | epochs / batch / scale lr | 2 / 1 / 1e-5 |
| Batch schedule | — | enabled from block 0, maximum batch 8, memory safety 1.5 |
| Tuning misc | — | compiled forward, reset seed each stage, no best-state restore, legacy epoch-loss mode |
| Allocation | budget / alignment / bounds | 219,368,448 factor bytes / 32 / 0.25×–3.0× uniform, overcomplete cap 3.0× |
| Distillation / retries | — | disabled |
| Export | token embedding / output tensor | Q8_0 / Q8_0 (Gemma ties output to embeddings, so no separate output tensor) |
| Runtime guard | WDDM shared memory | fail above 0.75 GiB |

The mixed optimizer split for factorized tuning is an inherited accident of the 073 config. The rewrite
reproduces it first. It can be unified only after an A/B run shows no quality loss (§12, Q3).

## 3. Scope

### 3.1 In scope (v1)

- Gemma 3 text decoders (270M, 1B, 4B text tower), behind a small model-adapter interface.
- The 085 method exactly as in §2, including its curated-KL allocation.
- The profiling pipeline that produces an allocation profile for a new model (§8.3), and re-allocation
  from an existing profile for a new budget.
- A stored-plan loader, used for exact 085 reproduction and as the golden reference.
- GGUF export for the modified llama.cpp (`D:\dev\research\llama.cpp`), plus one NanoQuant checkpoint format.
- The 085 evaluation protocol: WikiText-2, six tasks, MMLU, joint KL.
- Crash-safe resume at block granularity.

### 3.2 Removed (no port, no flags)

Each item below is unused by 085 and is dropped entirely, not hidden behind a flag:

- **Format variants:** product codebooks, sign-word codebooks, relational sign codes, partial sign planes.
- **Search methods:** binary factor search (tabu, variable-depth, one-bit passes), functional binary population.
- **Sidecars:** outliers (all selectors), bias correction, low-rank patches, structured sidecars.
- **Global objectives:** global distillation and top-k KD, mass-floor correction, final-norm calibration,
  foldable MLP multipliers, temperature calibration.
- **Allocation alternatives:** covariance refinement and Hessian/D2 objectives, D2-KL workflows,
  self-measured D2, rank retries, rank trust, protected units, edge-block boosts, rank expansion, KL splice.
- **Legacy compatibility:** legacy parity paths, legacy checkpoints, config migration, schema versions.
- **Runtimes:** the Python packed CUDA runtime and torch runtime bundle. llama.cpp is the deployment
  runtime. The rewrite keeps only a dense PyTorch reference used for evaluation.
- **Storage machinery:** the content-addressed artifact store, descriptors, validation caches, artifact
  GC, run registry, hard-link publication to `Results/`, orphan-commit adoption, `RESIDENT_ALGORITHM_VERSION`
  store coordination.
- **Infrastructure:**
  - observability: events framework, macro/micro profiler framework, CUDA trace capture;
  - resource management: memory-policy profiles, resource planner, runtime fallbacks;
  - other: interactive compression, teacher-trace and llama.cpp teacher generation, mmproj export,
    Qwen3 thinking-mode evaluation, long-context evaluation, Hugging Face upload.
- **Tests and tools:** almost all 228 tools and 356 test files. Replacements are listed in §10.

Anything here can come back later, but only through the promotion rule in §11.4.

## 4. Principles

1. **No hashing in the pipeline.** Identity is a name, plus a pinned Hugging Face revision, plus an exact
   config match. A file is committed when its atomic rename succeeds. Safetensors and JSON parsers detect
   truncation. Nothing computes a SHA-256 of a tensor, a directory, or a config.
2. **Plain paths that a person can read.** Example: `runs/gemma-3-1b-it-2p5/blocks/block-007.safetensors`.
   Never `artifacts/75/sha256-754a…/`.
3. **Recompute rather than store.** Activations, teacher logits, and tokenized eval sets are either rebuilt
   or kept in a disposable `cache/`.
4. **One compressed format.** The per-block files a run commits are the shards of the final checkpoint.
   GGUF is derived from them by a direct tensor copy.
5. **Fail loudly on mismatch.** If a resume uses a different config, or a plan belongs to a different
   model or shape, the run stops with an error that names the difference. It never silently adopts or migrates.
6. **Only knobs with a measured use.** The config contains the fields that 085 sets, and nothing else.

## 5. Code structure

### 5.1 Repository layout

```
nanoquant/                         new repository (sibling of this one)
├─ pyproject.toml                  torch, transformers (pinned), safetensors, gguf, numpy, scipy, pydantic, pyyaml
├─ recipes/
│  └─ gemma-3-1b-it-2p5.yaml       the 085 recipe (§6)
├─ profiles/                       small, git-tracked allocation profiles and stored plans (§8)
│  ├─ gemma-3-1b-it/curated-085/   curves.json, fits.json, anchors.json, units.json
│  └─ gemma-3-1b-it/plan-085.json  the 130 stage ranks 085 ran with
├─ src/nanoquant/
│  ├─ cli.py                       one entry point: `nanoquant <command>`
│  ├─ config.py                    strict pydantic models for recipes (extra = forbid)
│  ├─ workspace.py                 path conventions (§7), GPU lock file, atomic writes
│  ├─ log.py                       run.log + metrics.jsonl + a `timed()` context manager
│  │
│  ├─ core/                        pure torch, no I/O; ported nearly verbatim (~2k lines)
│  │  ├─ factor.py                 BinaryFactor / ResidualFactor dataclasses, reconstruct, bit cost, residual split
│  │  ├─ admm.py                   from domain/factorization.py (schedules, power iteration, sign projection)
│  │  ├─ scale_fit.py              from domain/scale_fit.py
│  │  ├─ importance.py             from domain/calibration_math.py (Fisher accumulators, shrinkage, group importance)
│  │  ├─ packing.py                sign bits ↔ int32 words, lsb-first, positive bit 0
│  │  └─ seeds.py                  logical_seed derivation (kept exactly, for reproducibility)
│  │
│  ├─ models/
│  │  ├─ adapter.py                ModelAdapter protocol: blocks, projections, groups, block forward, gguf names
│  │  └─ gemma3.py
│  │
│  ├─ data/
│  │  ├─ corpus.py                 build calibration / teacher token sets → tokens.npy
│  │  ├─ imatrix.py                read llama.cpp imatrix (.dat legacy and .gguf)
│  │  └─ eval_sets.py              WikiText-2, six tasks, MMLU at pinned revisions
│  │
│  ├─ calibrate.py                 Fisher output importance + imatrix input importance → importance.safetensors
│  │
│  ├─ allocation/
│  │  ├─ plan.py                   Plan type; load/save plan.json; `fixed` loader
│  │  ├─ rank_response.py          ADMM probes per unit at a rank grid → curves (production ADMM settings)
│  │  ├─ sensitivity.py            noise sweep + power-law fit per projection
│  │  ├─ anchors.py                isolated-KL measurement of a compressed checkpoint, one projection at a time
│  │  ├─ options.py                curves × fits × anchors → KL(rank) table (rerank_staged_kl_budget.py logic)
│  │  └─ solve.py                  integer knapsack (HiGHS via scipy.optimize.milp), pinned solver version
│  │
│  ├─ compress/
│  │  ├─ streams.py                FP teacher and student activation streams; GPU/host placement; replay
│  │  ├─ tuning.py                 non-factorized, factorized (STE), refit loops; both optimizers; batch schedule
│  │  ├─ block.py                  one block: the §2.1 unit loop, freeze, refit
│  │  └─ pipeline.py               block loop, commit, resume
│  │
│  ├─ export/
│  │  ├─ checkpoint.py             write/read the NanoQuant checkpoint (§9.1)
│  │  ├─ gguf.py                   checkpoint → GGUF (§9.2)
│  │  └─ dense.py                  checkpoint → dense HF model, for evaluation and KL
│  │
│  └─ evaluate/
│     ├─ kl.py                     full-vocab KL vs FP teacher (joint, and isolated per projection)
│     ├─ perplexity.py
│     ├─ tasks.py                  piqa, arc_easy, arc_challenge, hellaswag, winogrande, boolq
│     ├─ mmlu.py
│     └─ report.py                 report.md + report.json
└─ tests/                          §10
```

### 5.2 Dependency rule

The packages form layers, and each layer may import only the layers below it:

1. `core`
2. `models`
3. `data`, `calibrate`
4. `allocation`, `compress`
5. `export`, `evaluate`
6. `cli`

`core` imports only torch. One import-graph test enforces this rule. It replaces the current
architecture contract suite.

### 5.3 Size budget

The estimates below use the closest existing code as a guide:

| Package | Lines (est.) | Notes |
| --- | ---: | --- |
| core | 2,000 | factorization 433, scale_fit 243, calibration_math 169, linear_math 234, planning subset ≈ 300, packing |
| models | 400 | Gemma 3 only |
| data + calibrate | 1,000 | corpus builder, imatrix reader, Fisher pass |
| allocation | 1,400 | the four tools behind the 085 profile, merged and de-duplicated, plus the plan loader |
| compress | 2,500 | the part of `resident_quantization.py` (9,907 lines) that 085 exercises |
| export | 800 | checkpoint + GGUF via llama.cpp's converter classes |
| evaluate | 1,200 | |
| cli, config, workspace, log | 800 | |
| **Total** | **≈ 10,000** | vs 80,587 + 67,450 tools today |

`compress/` is the risky part to port. Phase 3 of §11 walks `resident_quantization.py` against the
085 config. Any branch that 085 never takes is deleted, not translated.

## 6. Configuration

### 6.1 Recipe format

A recipe is one YAML file, validated by strict pydantic models:

- No delta-on-reference scheme.
- No schema version or migrations.
- No config hash.

A field exists only if a production recipe sets it. Recipes live in the repository and change together
with the code.

```yaml
# recipes/gemma-3-1b-it-2p5.yaml — Experiment 085
name: gemma-3-1b-it-2p5
seed: 0

model:
  repo: google/gemma-3-1b-it
  revision: dcc83ea841ab6100d6b47a070329e1ba4cf78752
  dtype: bfloat16
  sequence_length: 2048

calibration:
  tokens: gemma3-chat-wikitext-256x2048      # datasets/tokens/<name>/
  imatrix: gemma-3-1b-it/unsloth             # datasets/imatrix/<name>.dat
  fisher: {shrinkage: 0.6, max_tokens_per_layer: 4096}

allocation:
  profile: gemma-3-1b-it/curated-085         # profiles/<name>/: curves, fits, anchors
  factor_bytes: 219368448                    # or: target_bpw: 2.51511
  rank_multiple: 32
  rank_bounds: {floor: 0.25, ceiling: 3.0, max_full_stages: 3}
  # fixed: gemma-3-1b-it/plan-085.json       # exact-reproduction mode: skip the solve, load stored ranks

factorization:
  admm: {outer: 800, inner: 2, regularization: 0.03, schedule: linear}
  scale_fit: {passes: 2}
  groups:
    - {name: attn_qkv, members: [q_proj, k_proj, v_proj], weights: {v_proj: 2.0}}

tuning:
  order: [gate_proj, up_proj, down_proj, attn_qkv, o_proj]
  dense:      {epochs: [8, 4, 3, 6, 2], batch: 4, lr: 1.0e-4}
  factorized: {epochs: 8, batch: 1, lr_sign: 3.0e-5, lr_scale: 1.0e-5,
               fused_scales_fp32_for: [gate_proj, down_proj, o_proj]}
  refit:      {epochs: 2, batch: 1, lr_scale: 1.0e-5}
  max_batch: 8

export:
  embeddings: q8_0

evaluation:
  suite: full     # quick = WikiText-2 + joint KL; full adds six tasks (1000) + MMLU 5-shot
```

### 6.2 Things that are not in the recipe

Device, workspace root, cache placement, and diagnostics live on the command line or in environment
variables. A run directory then records:

- the fully resolved recipe;
- the git commit of the code;
- `ALGORITHM_VERSION`, a hand-maintained integer bumped when numerics change.

## 7. Workspace and stored data

### 7.1 Two roots

- **The repository:** code, recipes, profiles. Everything here is small and git-tracked.
- **The workspace:** everything bulky. Set by `NANOQUANT_HOME`, default `D:\nanoquant`.

```
$NANOQUANT_HOME/
├─ datasets/                         inputs we build or import; permanent; small
│  ├─ tokens/<name>/                 tokens.npy (int32 [N, L]) + meta.json (sources, revisions, formatting,
│  │                                 tokenizer repo@revision, seed, counts)
│  ├─ imatrix/<model>/<name>.dat     + <name>.json (origin URL, date, notes)
│  └─ eval/                          nothing by default — eval sets come from the HF datasets cache
├─ cache/                            disposable; `rm -r` at any time costs only time
│  ├─ teacher/<model>/<tokens>/      FP teacher log-probs (fp16 memmap) for KL scoring
│  └─ eval/<model>/                  tokenized eval sets
├─ runs/<run-name>/                  §7.3
├─ releases/<release-name>/          §7.4
└─ locks/cuda0.lock                  OS file lock + pid; one compressing process per GPU
```

### 7.2 Where each class of data lives

| Class | Examples | Location | Lifetime |
| --- | --- | --- | --- |
| Base models | Gemma snapshots | Hugging Face hub cache (`HF_HOME`), referenced by repo + revision | managed by HF; never copied |
| External datasets | UltraChat, WikiText, MMLU, task sets | HF datasets cache, pinned revisions | managed by HF |
| Built inputs | calibration tokens, curated teacher windows, imatrix files | `datasets/` | permanent; a token set is ~2 MB |
| Allocation profiles and stored plans | curves, fits, anchors; `plan-085.json` | repo `profiles/` | permanent; versioned with code (§8) |
| Caches | teacher logits (21 GB for 160×256 × 262k vocab), tokenized evals | `cache/` | disposable |
| Runs | importance, plan, committed blocks, logs | `runs/` | until released; then delete scratch |
| Compressed models | checkpoint + GGUF + report | `releases/` | permanent |

There is no `Results/` hard-link tree. A release directory is the published result. A run directory is
the working record.

### 7.3 Run directory

```
runs/gemma-3-1b-it-2p5/
├─ run.json                    resolved recipe, code commit, ALGORITHM_VERSION, start time, host, GPU
├─ calibration/
│  └─ importance.safetensors   per-projection input (imatrix) and output (Fisher) importance vectors
├─ plan.json                   rank per unit, stage ranks, bytes per unit, total bytes, predicted KL
├─ blocks/
│  ├─ block-000.safetensors    committed compressed block (checkpoint shard format, §9.1)
│  ├─ block-000.json           per-unit metrics: weighted/raw error, losses per epoch, seconds
│  └─ …
├─ checkpoint/                 finalize step: index + non-quantized tensors + tokenizer + config
├─ model.gguf                  export step
├─ eval/report.md, report.json
├─ run.log                     human-readable
└─ metrics.jsonl               one record per unit/block/stage with timings
```

No activations are persisted. The 085 run kept rolling activation generations as content-addressed
artifacts, and reloaded them after every block. That reload is part of the 1,043 s commit phase.

**Commit and resume rules:**

1. **Commit point.** A block is committed when `block-NNN.safetensors.tmp` is renamed to its final name.
   The metrics JSON is written afterwards and is informational only.
2. **Resume check.** Resume compares the stored resolved recipe and `ALGORITHM_VERSION` with the current
   ones. Any difference is an error that lists the differing keys.
3. **Replay.** Resume skips the committed blocks. It rebuilds both streams by replaying the run from the
   embeddings:
   - FP blocks for the teacher stream;
   - committed blocks for the student stream.

   In 085 the propagate and teacher-forward phases took 135 s in total, so a full replay costs a few minutes.
4. **Propagate from the committed file.** The student stream is always advanced by a block reconstructed
   from its committed file: packed signs, BF16 scales. It is never advanced by the in-memory tuned tensors.
   Then an uninterrupted run and a resumed run see the same activations, as the old code ensured by
   reloading from disk. This relies on deterministic forward kernels, which 085's reproducibility settings
   (`deterministic: true`) already require.

Stream memory is 256 × 2048 × hidden × 2 bytes per stream in BF16:

| Model | Per stream |
| --- | ---: |
| Gemma 3 1B | 1.2 GB |
| Gemma 3 4B | 2.7 GB |

Both streams are kept in GPU memory when they fit, otherwise in pinned host memory.

### 7.4 Release directory

```
releases/gemma-3-1b-it-nanoquant-2p5/
├─ gemma-3-1b-it-nanoquant.gguf
├─ checkpoint/                 the NanoQuant checkpoint (optional; needed to re-export or re-evaluate)
├─ recipe.yaml                 copy of the resolved recipe
├─ plan.json
└─ report.md / report.json     protocol, scores vs BF16, sizes, wall time, peak memory
```

`nanoquant release` copies the files; nothing is hard-linked. If a release is published to Hugging Face,
the upload tool computes whatever checksums the hub wants at that moment. Nothing upstream depends on them.

### 7.5 What happens to today's data

| Existing item | Size | Fate |
| --- | ---: | --- |
| `Datasets/gemma-3-1b-it-imatrix-unsloth/imatrix_unsloth.dat` | 1.4 MB | import → `datasets/imatrix/gemma-3-1b-it/unsloth.dat` |
| 080 run calibration tokens (`evidence/080/080-d2-uniform-control-gemma-3-1b-it`) | (inside 17 GB) | export the token tensor → `datasets/tokens/gemma3-chat-wikitext-256x2048/` |
| 085 stage ranks (read from the 085 GGUF tensor shapes) | < 10 KB | **v1 input** → `profiles/gemma-3-1b-it/plan-085.json` |
| Curated 160×256 teacher window tokens (inside the sensitivity dataset; exact file to locate in Phase 0) | small | import → `datasets/tokens/gemma3-curated-160x256/` (joint-KL gate) |
| `Datasets/gemma-3-1b-it-staged-rank-response-080-v1` curves (cubic-600 probes) | ≈ 10 MB | import → `profiles/gemma-3-1b-it/curated-085/curves.json`, labelled with its ADMM settings |
| `…fp16-sensitivity-curated-160x256/measurements/8c98…/kl-response-fits.json` | 240 KB | import → `fits.json` |
| `outputs/085/082-curated-isolated.json` anchors (+ 082 reference ranks) | 1.5 MB | import → `anchors.json`, without per-sequence scores |
| `…sensitivity-curated-160x256/artifacts/66` (teacher logits) | 20 GB | becomes cache; archive or delete after the gate passes |
| `Datasets/*-kl-48x512`, `*-gguf-local-quantize`, other sensitivity sweeps | ≈ 60 GB | not needed by production; archive or delete later |
| `evidence/`, `outputs/`, `Results/` | ≈ 357 GB | stay in this repository as research history; production reads none of it |

Nothing is deleted as part of this design. Deletion is a separate decision, made after the parity gate passes.

## 8. Allocation

### 8.1 Why allocation and ADMM are coupled

The allocation and the ADMM settings depend on each other:

- **Uniform ranks:** cubic-600 ADMM fits the uniform two-stage ranks well. Switching 080 to linear-800
  changed nothing. That result is from a run reported by the user, not a file in the repo.
- **Curated ranks:** the curated allocation produces ranks far from uniform. Some units get three full
  stages, others drop to a quarter of uniform. At those ranks, cubic 600 left about 5% fitting error.
- **The result:** with cubic 600 the curated plan tied uniform (joint KL 0.7467 vs 0.7473). With linear
  800 it reached 0.7263. The ADMM study at 085's ranks (`Results/085/085-admm-schedule-study.md`) shows
  the local effect: −4.5% objective error and −3.5% activation error.

So "the 085 method" is a *pair*: an allocator, plus a factorizer validated at the ranks that allocator
chooses. §8.4 makes that pairing a design rule.

### 8.2 Plans

`allocate` writes `plan.json`:

- unit id;
- total rank;
- stage ranks;
- bytes per unit;
- total bytes;
- predicted KL.

`compress` reads only that file. Two modes produce it:

| Mode | What it does | Use |
| --- | --- | --- |
| `profile` (default) | KL options from the profile (§8.3), then the knapsack under `factor_bytes` | production: any budget, any profiled model |
| `fixed` | loads a stored plan; checks shapes, alignment, and total bytes against `factor_bytes` | exact 085 reproduction; golden tests |

`plan-085.json` is read from the 085 GGUF's tensor shapes, so it records what 085 actually ran.

The `profile` mode can't reproduce 085's ranks exactly. A fresh HiGHS solve over the published 085
options (`integer_allocate`, the 085 budget) matches 124 of 130 ranks. The other 6 differ by ±32 ranks,
byte-neutral swaps within the solver's gap. So the parity gate uses `fixed`. The `profile` mode is tested
on predicted KL: within the solver gap of the stored plan's predicted KL. The solver version is pinned.

### 8.3 Allocation profiles

A profile is the per-model knowledge that the allocator needs. It is independent of the budget. It
consists of four plain JSON files:

| File | Content | Produced by |
| --- | --- | --- |
| `units.json` | unit list: id, members, shape, group weights | model adapter |
| `curves.json` | per unit and member: rank grid, imatrix-weighted NMSE per rank | `nanoquant profile curves` |
| `fits.json` | per projection: coefficient, exponent, measured error range, fit quality | `nanoquant profile sensitivity` |
| `anchors.json` | reference checkpoint name, its rank per unit, measured isolated KL and noise-equivalent error per projection | `nanoquant profile anchors` |

`nanoquant allocate` computes the KL(rank) options at run time with the `rerank_staged_kl_budget.py` formula:

1. Restrict each projection to its sweep range.
2. Solve the knapsack.
3. Write `plan.json`.

This replaces the published "KL options" dataset. That dataset was a derived product, and the
derivation takes seconds. A budget change therefore never needs a new profile, as long as the
requested ranks stay inside the measured curve range.

**Profiling a new model: the two-pass flow**

| Step | Command | 1B cost (from the 085 lineage) |
| --- | --- | --- |
| 1 | `nanoquant profile curves`: ADMM probes per unit on a rank grid, no tuning | probe ADMM only |
| 2 | `nanoquant profile sensitivity`: Gaussian noise at 6 levels × 3 seeds per projection, scored on curated windows | 3,276 trials × ~6 s ≈ 5.5 h |
| 3 | `nanoquant allocate --anchors fit`: first plan from fit-predicted anchors | seconds |
| 4 | `nanoquant compress` with that plan: the pilot | ≈ 3.8 h |
| 5 | `nanoquant profile anchors --checkpoint <pilot>`: isolated KL, one projection at a time | 182 × ~6 s ≈ 18 min |
| 6 | `nanoquant allocate` + `compress`: the final run | ≈ 3.8 h |

**Why the pilot exists.** This document re-solved the 085 options with each measured anchor replaced by
the sensitivity fit's prediction at the same error (`weighted_kl` in place of `measured_kl`):

- 102 of 130 units chose a different rank, by 219 ranks on average.
- Priced with the measured anchors, the fit-only plan costs 4.3% more predicted isolated KL: 2.098 vs 2.012.

That check is optimistic for fit-only. It used 082's real post-tuning projection errors. A model that has
never been compressed has only untuned ADMM probe errors, which are further from the deployed errors.

These are predicted isolated-KL sums, not measured benchmark results.

Steps 3–5 can be skipped when a release of the same model, under the same recipe family, already exists;
that release can serve as the anchor checkpoint. This is how 085 itself worked: its anchors came from 082.

### 8.4 Rule: ADMM settings belong to the rank range, not the model

The playground's result has to hold in production, including for new models and budgets. Three rules
make sure it does:

1. **Probes use production settings.** `profile curves` runs ADMM with the recipe's own settings,
   linear 800. Each curve records the settings it was measured with. `allocate` refuses a profile whose
   ADMM settings differ from the recipe's.
   - The imported 085 curves were measured with cubic 600, so they keep a known bias: they overstate
     error at high ranks, which is where cubic 600 underfit.
   - Phase 6 (§11) re-measures the 1B curves with linear 800.
   - The resulting plan may then differ from 085's. That is a playground question to settle before
     production uses it, and the gate in §11.2 still runs on `plan-085.json`.
   - The imported 085 curves load only in `fixed`-plan runs, or with an explicit `--allow-probe-mismatch`.
2. **Fit check over the chosen ranks.** After `allocate`, a fitting check runs on a few units at the
   extremes of the plan:
   - the highest rank relative to full;
   - the most stages;
   - the lowest rank.

   It compares the recipe's ADMM against a longer reference run, such as 2× iterations. If the gap is
   above a threshold (proposed 1%, versus the 5% that 085 hit), the run fails before compression and
   names the units. This is the CPU/GPU-minutes version of the 085 ADMM study.
3. **No rank-dependent ADMM schedule in v1.** One schedule for all units is what 085 validated.
   Scaling iterations with rank or stage count is a natural next experiment, and it belongs in the
   playground first (§11.4).

## 9. Compressed model format

### 9.1 NanoQuant checkpoint (canonical)

The checkpoint is a directory that mirrors a Hugging Face model layout:

```
checkpoint/
├─ config.json, tokenizer files    copied from the pinned snapshot
├─ nanoquant.json                  format version, bit order, positive bit, per-unit stage ranks, groups,
│                                  source repo@revision, recipe name
├─ model.safetensors.index.json
├─ block-000.safetensors … block-025.safetensors     (identical to runs/<name>/blocks/*)
└─ dense.safetensors               embeddings, norms, any non-factorized tensors, BF16 as in the source
```

Per-unit tensor names keep the current llama.cpp-checkpoint naming:

- `…<unit>.U_packed`, `…<unit>.V_packed`: int32 words, sign bits packed along rank, lsb-first,
  positive = bit 0;
- `…<unit>.scale_pre`, `scale_mid`, `scale_post`: BF16;
- `…<unit>.residual.<n>.*` for each later stage;
- the qkv group appears as one fused `attn_qkv` unit.

The `*_shape` and `residual_stage_count` scalar tensors move into `nanoquant.json`.

### 9.2 GGUF (deployable)

The tensor and metadata contract is fixed by the modified llama.cpp, and the 085 GGUF shows it.

Factorized tensors:

- **Names:** `blk.N.<attn_qkv|attn_output|ffn_gate|ffn_up|ffn_down>.nq_stage0_{u,v,scale_pre,scale_mid,scale_post}`,
  and `nq_residual_<k>_*` for later stages. A unit with a single stage drops the stage tag, for example
  `blk.23.attn_qkv.nq_u` at rank 896.
- **Types:** u and v are I32; scales are BF16.

Other tensors:

- `token_embd.weight` is Q8_0.
- Norms are F32.

Metadata:

- `quantization.nanoquant.version` = 3
- `stage_format` = `additive_binary_v1`
- `bit_order` = `lsb_first`
- `positive_bit` = `0`
- `tensor_count`

The current export runs llama.cpp's converter, then a `llama-quantize` subprocess for the embeddings. The
rewrite drops the subprocess. **It was checked that `gguf.quants.quantize(..., Q8_0)` on the BF16 source
embeddings is byte-identical to the 085 GGUF's `token_embd.weight` (320,864,256 bytes).**

The rewrite still imports the `Gemma3Model` class from the pinned llama.cpp `convert_hf_to_gguf.py` in-process,
for hyperparameters and tokenizer metadata, as `tools/llamacpp/convert_nanoquant_to_gguf.py` does today.
Re-implementing tokenizer metadata is not worth the risk.

Export test: for every tensor in the checkpoint, the matching GGUF tensor must be byte-equal. A full-model
gate also compares against the 085 GGUF tensor by tensor and field by field.

### 9.3 Dense reference

`export/dense.py` reconstructs each projection into an ordinary HF Gemma 3 model:

W = Σ_stages diag(post)·U·diag(mid)·Vᵀ·diag(pre)

The qkv group is split back into its members. This model is what evaluation and KL scoring run on.
It is a correctness reference only; deployment performance and memory are measured in llama.cpp.

## 10. Tests

The suite is small and every test guards a real failure mode:

| Tier | Tests | Runs |
| --- | --- | --- |
| Unit (CPU, seconds) | bit cost and residual-stage split vs hand-computed cases; pack/unpack round trip; ADMM reduces weighted error on a random matrix and is deterministic for a seed; scale fit never regresses; knapsack equals brute force on small instances; KL-option formula matches `rerank_staged_kl_budget.py` on a few saved units; imatrix reader on a tiny fixture; config rejects unknown fields; import-layering rule | every commit |
| Golden math (CPU) | three real 085 projections (one MLP, o, qkv): rewrite ADMM + scale fit vs saved old-code outputs, bit-exact or within a stated tolerance | every commit (fixtures ≈ a few MB) |
| End to end (CPU, < 2 min) | random-initialized 2-block Gemma 3 config → calibrate → allocate → compress → kill/resume at a block → export → GGUF read-back equals checkpoint → dense eval is finite | every commit |
| Allocation golden (CPU) | `plan-085.json` loads, totals 219,368,448 bytes, and matches the 085 GGUF stage shapes; the imported 085 profile at the 085 budget (with `--allow-probe-mismatch`) gives a plan within the byte budget, whose predicted KL is within the solver gap of 2.01198 (the in-range re-solve measured for this doc) | every commit |
| Parity gate (GPU, hours) | §11.2, the full 085 reproduction | manual, before each release |

Test infrastructure stays minimal: one `conftest.py`, the tiny-model factory, and the fixtures above.
There is no support package and no contract suite beyond the layering test.

## 11. Migration plan

### 11.1 Phases

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 0. Freeze references | From this repo, export: the 080 calibration tokens, the imatrix, the profile inputs (curves, fits, anchors; §7.5), `plan-085.json` (from the GGUF shapes), golden ADMM inputs/outputs for three projections (including one at three full stages, where cubic 600 underfit), block-0 and block-1 inputs plus per-unit losses, the curated teacher tokens, and the 085 GGUF tensor manifest. Answer Q4 (eval embeddings). | reference bundle written; this repo untouched otherwise |
| 1. Skeleton + core | new repo, config, workspace, core math port, unit + golden-math tests | golden ADMM matches |
| 2. Allocation | profile import, options, solver, `fixed` loader, the §8.4 fit check | allocation golden tests pass; the fit check flags cubic 600 at 085's ranks and passes linear 800 |
| 3. Block compressor | streams, tuning, block, pipeline, resume | blocks 0–1 per-unit weighted errors and post-tuning losses within ±1% of the values 085 recorded (`weight-errors.md`, run events); tiny e2e passes |
| 4. Export + eval | checkpoint, GGUF, dense, evaluators | GGUF tensors equal checkpoint; evaluating the old 085 checkpoint reproduces its report. This depends on Q4: use the same embedding tensors as the 085 dense evaluation. |
| 5. Full parity | full 085 recipe on GPU | §11.2 gate passes |
| 6. Profiling commands | `profile curves`, `sensitivity`, `anchors` on GPU | re-derive the 1B profile from scratch with linear-800 probes. The resulting plan compresses to within the §11.2 tolerance bands, though not necessarily with identical ranks. |
| 7. Retire | decide on old repo archival and data deletion | user decision |

Phases 1–5 reproduce 085 from its stored plan and imported profile. The 5.5 h sweep is not re-run.
Phase 6 must pass before a model that has never been profiled goes through production, so 270M and
4B depend on it.

### 11.2 Parity gate

End-to-end bitwise equality is not a realistic target: compiled forwards, cuBLAS kernel selection, and
atomics all differ. The gate is therefore:

- **Exact:**
  - same 130 ranks and stage splits as `plan-085.json`;
  - GGUF tensor shapes and types;
  - `token_embd` bytes;
  - GGUF size within the metadata-string difference of 547,406,560 bytes.
- **Tolerance (provisional):**
  - joint decoder KL 0.726 ± 1.5%, which is narrower than the 2.8% gap between 085 and 080, so the gate
    can tell them apart;
  - WikiText-2 PPL 124.59 ± 2%;
  - six-task mean 56.08% ± 1.5 pp;
  - MMLU 31.83% ± 1.0 pp, about 2.5σ of sampling error for 14,042 questions.
- **Not worse:**
  - compression wall time ≤ 13,799 s on the same GPU (the 085 profile, excluding export and evaluation);
  - peak GPU memory ≤ the 085 run's peak.

The tolerance bands are provisional. A seed-repeat of the rewrite (seed 1) sets the real run-to-run
band, and the gate is then rewritten with it. Keeping the `logical_seed` derivation lets per-layer
factorization stay comparable even when end-to-end runs drift.

### 11.3 GPU safety during migration

Only one CUDA compression may run per GPU. The rewrite takes an OS lock on `locks/cuda0.lock` (msvcrt on
Windows, fcntl elsewhere) that records its pid. It refuses to start if the old repo's device lease or
`nvidia-smi` shows another NanoQuant process. The WDDM shared-memory guard (0.75 GiB) is kept as a hard
failure, because Windows spills to shared memory silently.

### 11.4 Promotion rule (after v1)

The playground remains the place where methods compete. A method enters production only when:

1. it beats the current release on the parity metrics in a playground experiment; and
2. it is ported with its own golden test and passes the gate.

A method is never added as a flag "in case". When a newer method replaces one in production, the old
code is deleted.

## 12. Open questions

| # | Question | Recommendation |
| --- | --- | --- |
| Q1 | New repository, or rewrite in place? | New sibling repository. This one stays read-only as the parity reference and research playground. |
| Q2 | Accept the two-pass profiling cost for new models: the 5.5 h sweep plus a pilot compression? | Yes. Measured anchors move 102 of 130 ranks, and the allocation is half of 085's gain (§8.1). Re-allocating for a new budget on a profiled model stays a single pass. |
| Q9 | Should the 1B curves be re-measured with linear-800 probes before the first release? | Yes, in Phase 6. The imported cubic-600 curves overstate high-rank error. If the re-measured plan differs from 085's, compare the two in the playground before the release recipe switches. |
| Q3 | Unify the factorized optimizer (fused for gate/down/o, `parity` for up/qkv)? | Reproduce first; unify only after an A/B run inside the gate. |
| Q4 | Did 085's dense evaluation use BF16 or dequantized Q8_0 embeddings? The joint-KL report says its "full model KL" swaps in the Q8_0 embeddings; the dense eval path was not confirmed. | Check in Phase 0. The rewrite should then evaluate exactly the tensors it ships, and record which. |
| Q5 | Scope of v1 model support? | Gemma 3 text only. Add Llama and Qwen adapters when a playground result justifies them. |
| Q6 | Per-block loss snapshots (285 s) and entry-loss probes (73 s)? | Drop them as separate phases; record the losses the tuning loops already compute. Add an opt-in `--diagnostics` flag later if needed. |
| Q7 | Keep a cached teacher (21 GB) for the sensitivity sweep, or recompute? | Cache under `cache/` when disk allows; otherwise recompute the FP teacher per batch, at about 2× sweep time. |
| Q8 | Is code-commit mismatch allowed on resume? | Warn on a different git commit; refuse on a different `ALGORITHM_VERSION` or recipe. |
