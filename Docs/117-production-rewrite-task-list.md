# 117 — Production rewrite task list

Status: not started, 2026-11-19. This is the implementation checklist for the design in
[116-production-rewrite-design.md](116-production-rewrite-design.md). Section references (§) point there.

An item is checked only when its stated evidence exists. A phase is done when its exit criterion passes,
not when its items are written.

## Decisions before starting

- [ ] Q1: confirm a new sibling repository (recommended) rather than a rewrite in place.
- [ ] Q2: accept the two-pass profiling cost for new models (5.5 h sweep + pilot compression).
- [ ] Q5: confirm v1 supports Gemma 3 text decoders only.
- [ ] Q6: confirm per-block loss snapshots and entry-loss probes are dropped as separate phases.
- [ ] Q7: choose cached teacher logits (21 GB) or recompute-per-batch for the sensitivity sweep.
- [ ] Q8: confirm resume policy: warn on a different git commit; refuse on a different recipe or
      `ALGORITHM_VERSION`.
- [ ] Pick the workspace root (`NANOQUANT_HOME`, proposed `D:\nanoquant`) and check free disk space.
- [ ] Q3 and Q9 are deferred to their phases below; they are not blockers.

## Phase 0 — Freeze references (this repository, read-only apart from new exports)

Write everything to one reference bundle, e.g. `D:\nanoquant\reference\085\`, with a `README.md`
saying where each file came from.

- [ ] Export the 080 calibration token tensor (256 × 2048, int32) from
      `evidence/080/080-d2-uniform-control-gemma-3-1b-it` → `tokens.npy` + `meta.json` (sources, dataset
      revisions, formatting `gemma-chat-plus-raw-text-v1`, tokenizer repo@revision, seed 0).
- [ ] Copy the Unsloth imatrix `Datasets/gemma-3-1b-it-imatrix-unsloth/imatrix_unsloth.dat` + origin notes.
- [ ] Locate the curated 160 × 256 teacher-window tokens inside the sensitivity dataset and export them.
- [ ] Export the profile inputs (§7.5):
  - [ ] curves from `Datasets/gemma-3-1b-it-staged-rank-response-080-v1`, labelled cubic 600;
  - [ ] fits from `…fp16-sensitivity-curated-160x256/measurements/8c98…/kl-response-fits.json`;
  - [ ] anchors from `outputs/085/082-curated-isolated.json`, plus the 082 reference ranks,
        without per-sequence scores.
- [ ] Build `plan-085.json` from the 085 GGUF tensor shapes: 130 units, stage ranks, bytes per unit;
      check the total is 219,368,448 bytes.
- [ ] Write the 085 GGUF manifest: every tensor's name, shape, type, byte size, and every metadata
      key/value.
- [ ] Save golden ADMM + scale-fit inputs and outputs for three real projections (one MLP, one o, one qkv),
      including one unit at three full stages. Save both cubic-600 and linear-800 outputs for that unit.
- [ ] Save block-0 and block-1 inputs (teacher and student streams, or the seed and tokens to rebuild them)
      and the per-unit weighted errors and post-tuning losses 085 recorded.
- [ ] Record the 085 reference numbers the gate uses: wall time 13,799 s, peak GPU memory, and the four
      quality metrics with their protocols.
- [ ] Answer Q4: did the 085 dense evaluation use BF16 or dequantized Q8_0 embeddings? Write the answer
      into the bundle README.
- [ ] Confirm `git status` of this repository shows no source changes from Phase 0.

**Exit:** the reference bundle exists and its README lists every file with its source path.

## Phase 1 — Skeleton and core math

### Repository

- [ ] Create the repository with the §5.1 layout; `pyproject.toml` with pinned torch, transformers,
      safetensors, gguf, numpy, scipy, pydantic, pyyaml; Python ≥ 3.10.
- [ ] Set up ruff, mypy, and pytest; one `conftest.py`.
- [ ] Add the import-layering test (§5.2): `core` imports only torch; each layer imports only lower ones.

### Infrastructure modules

- [ ] `config.py`: strict pydantic recipe models (`extra = forbid`), with exactly the §6.1 fields.
- [ ] `recipes/gemma-3-1b-it-2p5.yaml`: the 085 recipe.
- [ ] `workspace.py`: `NANOQUANT_HOME` path conventions (§7.1), atomic write (`.tmp` + rename),
      GPU lock file with pid (msvcrt on Windows, fcntl elsewhere).
- [ ] `log.py`: `run.log`, `metrics.jsonl`, `timed()` context manager.
- [ ] `ALGORITHM_VERSION` constant, and a note in the contributor docs on when to bump it.

### Core (ported nearly verbatim from `src/nanoquant/domain`)

- [ ] `core/factor.py`: `BinaryFactor`, `ResidualFactor`, reconstruct, bit cost, residual-stage split.
- [ ] `core/admm.py`: schedules (linear, cubic), power iteration, sign projection; bf16 compute, fp32 solve.
- [ ] `core/scale_fit.py`: 2 alternating passes, 512-row chunks, epsilon 1e-8, rollback on regression.
- [ ] `core/importance.py`: Fisher accumulators, shrinkage, group importance (v multiplier 2.0).
- [ ] `core/packing.py`: sign bits ↔ int32 words, lsb-first, positive = bit 0.
- [ ] `core/seeds.py`: `logical_seed` derivation, unchanged.

### Tests

- [ ] Unit: bit cost and residual split vs hand-computed cases.
- [ ] Unit: pack/unpack round trip.
- [ ] Unit: ADMM reduces weighted error on a random matrix and is deterministic for a seed.
- [ ] Unit: scale fit never regresses.
- [ ] Unit: config rejects unknown fields.
- [ ] Golden math: rewrite ADMM + scale fit on the three Phase 0 projections match the saved outputs,
      bit-exact or within a stated tolerance.

**Exit:** golden ADMM matches; ruff, mypy, and pytest are clean.

## Phase 2 — Allocation

- [ ] `models/adapter.py` and `models/gemma3.py`: units, projections, groups, shapes, GGUF names
      (enough for `units.json`; block forward can wait for Phase 3).
- [ ] `allocation/plan.py`: `Plan` type; load/save `plan.json`; `fixed` mode checks shapes, 32-alignment,
      ≤ 3 full stages, and total bytes against `factor_bytes`.
- [ ] Profile import: write `profiles/gemma-3-1b-it/curated-085/{units,curves,fits,anchors}.json` and
      `profiles/gemma-3-1b-it/plan-085.json` from the Phase 0 bundle. Curves record their ADMM settings.
- [ ] `allocation/options.py`: KL(rank) = anchor_kl · (err(rank)/err(anchor_rank))^p, restricted to the sweep
      range, floor 0.25× / ceiling 3.0× uniform.
- [ ] `allocation/solve.py`: integer knapsack with HiGHS via `scipy.optimize.milp`; pin the scipy version.
- [ ] Probe-settings check: `allocate` refuses a profile whose ADMM settings differ from the recipe's,
      unless `--allow-probe-mismatch` is set or the plan is `fixed`.
- [ ] §8.4 fit check: refit the units at the plan's extremes (highest relative rank, most stages, lowest
      rank) with the recipe's ADMM and with a 2× reference; fail above the threshold (proposed 1%) and name
      the units.
- [ ] `nanoquant allocate` CLI command.

### Tests

- [ ] Unit: knapsack equals brute force on small instances.
- [ ] Unit: KL-option formula matches `tools/rerank_staged_kl_budget.py` on a few saved units.
- [ ] Allocation golden: `plan-085.json` loads, totals 219,368,448 bytes, matches the 085 GGUF stage shapes.
- [ ] Allocation golden: the imported profile at the 085 budget gives a plan within budget whose predicted
      KL is within the solver gap of 2.01198.
- [ ] Fit check: flags cubic 600 at 085's ranks and passes linear 800.

**Exit:** all allocation tests pass, including both fit-check outcomes.

## Phase 3 — Block compressor

### Port preparation

- [ ] Walk `src/nanoquant/resident_quantization.py` against the 085 config and list the branches 085
      takes. Record the list in the new repo; every other branch is not ported.

### Modules

- [ ] `models/gemma3.py`: block forward, projection access, qkv fusion and split.
- [ ] `data/imatrix.py`: legacy `.dat` and `.gguf` readers.
- [ ] `data/corpus.py`: build token sets → `datasets/tokens/<name>/`.
- [ ] `calibrate.py`: imatrix input importance + online Fisher output importance (shrinkage 0.6,
      ≤ 4096 tokens/layer) → `runs/<name>/calibration/importance.safetensors`.
- [ ] `compress/streams.py`: FP teacher and student streams; GPU placement when they fit, pinned host
      memory otherwise.
- [ ] `compress/tuning.py`:
  - [ ] non-factorized tuning: epochs by position 8,4,3,6,2; batch 4; AdamW lr 1e-4, wd 0;
  - [ ] factorized STE tuning: 8 epochs, batch 1, lr sign 3e-5, scale 1e-5, bias 1e-5;
  - [ ] both optimizers: `fused_adamw_scales_fp32` for gate/down/o, `parity` for up/qkv;
  - [ ] post-block refit: 2 epochs, batch 1, scale lr 1e-5;
  - [ ] batch schedule from block 0, max batch 8, memory safety 1.5;
  - [ ] compiled forward, seed reset per stage, no best-state restore, legacy epoch-loss mode.
- [ ] `compress/block.py`: the unit loop in order gate, up, down, qkv, o; freeze; refit.
- [ ] `compress/pipeline.py`:
  - [ ] block loop with `.tmp` → rename commit of `blocks/block-NNN.safetensors`, then `block-NNN.json`;
  - [ ] `run.json` with resolved recipe, git commit, `ALGORITHM_VERSION`, host, GPU;
  - [ ] resume check that lists differing recipe keys;
  - [ ] replay of both streams from the embeddings on resume;
  - [ ] student stream always advanced from the block rebuilt from its committed file.
- [ ] GPU lock, refusal when the old repo's device lease or `nvidia-smi` shows another NanoQuant process,
      and the WDDM shared-memory guard (fail above 0.75 GiB).
- [ ] `nanoquant calibrate` and `nanoquant compress` CLI commands.

### Tests

- [ ] End to end (CPU, < 2 min): random-initialized 2-block Gemma 3 → calibrate → allocate → compress →
      kill and resume at a block → identical blocks to an uninterrupted run.
- [ ] Resume refuses a changed recipe and a changed `ALGORITHM_VERSION`.
- [ ] GPU run of blocks 0–1 with `plan-085.json`: per-unit weighted errors and post-tuning losses within
      ±1% of the Phase 0 values.

**Exit:** blocks 0–1 within ±1% of 085; tiny end-to-end passes.

## Phase 4 — Export and evaluation

- [ ] `export/checkpoint.py`: §9.1 layout; `nanoquant.json` carries format version, bit order, positive bit,
      stage ranks, groups, source repo@revision, recipe name; the `*_shape` and stage-count scalars move there.
- [ ] `export/gguf.py`: §9.2 names and types; Q8_0 embeddings via `gguf.quants.quantize`; metadata via
      llama.cpp's `Gemma3Model` converter classes imported in-process; no `llama-quantize` subprocess.
- [ ] `export/dense.py`: reconstruct W = Σ diag(post)·U·diag(mid)·Vᵀ·diag(pre), split qkv, load into an HF
      Gemma 3 model, with the embedding choice from Q4.
- [ ] `data/eval_sets.py`: WikiText-2, six tasks, MMLU at pinned revisions; tokenized cache under `cache/eval/`.
- [ ] `evaluate/kl.py`: full-vocab joint KL and isolated per-projection KL vs the FP teacher; teacher
      log-probs cached under `cache/teacher/`.
- [ ] `evaluate/perplexity.py`: WikiText-2 64 × 128, batch 8.
- [ ] `evaluate/tasks.py`: piqa, arc_easy, arc_challenge, hellaswag, winogrande, boolq; first 1000 rows, batch 4.
- [ ] `evaluate/mmlu.py`: full 14,042-question test split, 5-shot same-subject dev, answer-letter logits.
- [ ] `evaluate/report.py`: `report.md` + `report.json` with protocol, scores vs BF16, sizes, wall time,
      peak memory, and which embedding tensors were evaluated.
- [ ] `nanoquant export`, `nanoquant evaluate`, `nanoquant release` (plain copy into `releases/`).

### Tests

- [ ] Every checkpoint tensor equals its GGUF tensor byte for byte (tiny e2e and real model).
- [ ] Q8_0 `token_embd.weight` equals the 085 GGUF's 320,864,256 bytes.
- [ ] Converting the old 085 checkpoint with the new exporter matches the Phase 0 GGUF manifest tensor by
      tensor and field by field.
- [ ] Evaluating the old 085 checkpoint with the new evaluators reproduces the 085 report.

**Exit:** GGUF equals checkpoint; the old 085 checkpoint's report reproduces.

## Phase 5 — Full parity

- [ ] Confirm the GPU is free: process list, old-repo device lease, `nvidia-smi`.
- [ ] Run the full 085 recipe with `fixed: gemma-3-1b-it/plan-085.json` (calibrate → compress → export →
      evaluate). Run it detached; poll rarely.
- [ ] Exact checks: 130 ranks and stage splits, GGUF shapes and types, `token_embd` bytes, GGUF size within
      the metadata-string difference of 547,406,560.
- [ ] Tolerance checks: joint decoder KL 0.726 ± 1.5%, PPL 124.59 ± 2%, six-task 56.08% ± 1.5 pp,
      MMLU 31.83% ± 1.0 pp.
- [ ] Not-worse checks: compression wall time ≤ 13,799 s, peak GPU memory ≤ 085's peak.
- [ ] Load the GGUF in the modified llama.cpp and run a short generation smoke test.
- [ ] Seed-repeat (seed 1); rewrite the §11.2 tolerance bands from the measured run-to-run spread.
- [ ] Optional (Q3): A/B of a unified factorized optimizer inside the gate; adopt only if it passes.
- [ ] Cut the first release under `releases/gemma-3-1b-it-nanoquant-2p5/`.

**Exit:** the §11.2 gate passes and the bands are set from the seed repeat.

## Phase 6 — Profiling commands

- [ ] `allocation/rank_response.py` and `nanoquant profile curves`: ADMM probes per unit on a rank grid
      with the recipe's ADMM settings, recorded in `curves.json`.
- [ ] `allocation/sensitivity.py` and `nanoquant profile sensitivity`: Gaussian noise at 6 levels × 3 seeds
      per projection on the curated windows; power-law fit per projection.
- [ ] `allocation/anchors.py` and `nanoquant profile anchors --checkpoint <run>`: isolated KL, one
      projection at a time.
- [ ] `nanoquant allocate --anchors fit` for the first pass of a new model.
- [ ] Re-derive the 1B profile from scratch with linear-800 probes (two-pass flow, §8.3).
- [ ] Q9: compare the re-derived plan with `plan-085.json`; if it differs, run the comparison in the
      playground before the release recipe switches.
- [ ] Compress with the re-derived plan; result within the §11.2 bands (ranks may differ).

**Exit:** the from-scratch 1B profile passes the bands. Only then do 270M and 4B go through production.

## Phase 7 — Retire

- [ ] User decision: archive this repository as read-only playground, or keep it active.
- [ ] User decision: archive or delete bulky data (§7.5), only after the gate passes. Use
      `tools/cleanup_artifacts.py` dry runs first; delete nothing while a run is active.
