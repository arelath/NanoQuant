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
- [ ] Operations decisions, needed before track O2/O3 (not before Phase 1):
  - [ ] Q12: job image in a registry (recommended: GHCR, pulled by digest) or stock image + git clone;
  - [ ] Hugging Face namespace for releases, and for the private inputs dataset repo (Q13);
  - [ ] default privacy for uploads (recommended: private until Q14 is answered);
  - [ ] default RunPod spend cap per job (`max_hours`, `max_cost_usd`) and preferred data center.

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

## Operations track (§13)

These run alongside the numbered phases. Each one lists the phase it needs first. They add no numerical
code, so they never block the parity gate.

### O1 — Menu-driven CLI (after Phase 4; Plans items after Phase 2)

- [ ] `menu.py` with injected input/output functions; plain stdlib, numbered choices.
- [ ] Every item calls an existing command function and prints the equivalent `nanoquant …` command first.
- [ ] Runs menu: list with status (blocks committed / 26, phase, heartbeat, local or job); log tail and
      metrics summary; new run; resume; export; evaluate; release.
- [ ] Plans menu:
  - [ ] allocate at a new byte budget or target BPW;
  - [ ] show a plan by projection and block (ranks, stages, bytes, predicted KL);
  - [ ] diff two plans (units changed, rank deltas, byte and predicted-KL change), backed by a new
        `nanoquant plan diff` command;
  - [ ] run the §8.4 fit check;
  - [ ] set a run's `fixed` plan before it starts.
- [ ] Profiles menu: list profiles with their probe ADMM settings; start a profiling step (needs Phase 6).
- [ ] Releases menu: list with headline scores; compare two reports; upload (needs O2).
- [ ] Workspace menu: disk use per root; clear `cache/`; delete a released run's scratch, with confirmation.
- [ ] `nanoquant run` convenience command: calibrate → allocate → compress → export → evaluate → release,
      each step resuming.
- [ ] Tests: scripted input reaches every item; each prints its command; destructive items need confirmation.

**Exit:** every menu item has a scripted test, and each printed command reproduces the item's effect.

### O2 — Hugging Face upload (after Phase 4; first real upload after Phase 5)

- [ ] `ops/hub.py` and `nanoquant upload <release> --repo … [--public] [--with-checkpoint]`.
- [ ] Refuse unless `report.json` shows completed evaluation with finite metrics on the release's own files.
- [ ] Create repos private by default; `--public` or a menu confirmation is the only way to publish.
- [ ] Generate `README.md` from `report.json`: `base_model`, `base_model_relation: quantized`,
      `library_name: gguf`, license, tags, scores vs BF16, sizes, BPW, plan summary, code commit, the
      llama.cpp revision needed to run it.
- [ ] Q14: check the Gemma terms for published derivatives and put what they require into the card template.
- [ ] Upload GGUF, card, `recipe.yaml`, `plan.json` with `huggingface_hub`; no local checksums.
- [ ] Write `releases/<name>/upload.json`.
- [ ] Token only from `HF_TOKEN` or the hub login cache; a test asserts it reaches no file or log.
- [ ] Tests against a fake `HfApi`: refusal on incomplete report, private default, card content,
      `upload.json`.
- [ ] Upload the Phase 5 release to a private repo; download it and load it in the modified llama.cpp.

**Exit:** the Phase 5 release is on Hugging Face (private) and loads after download.

### O3 — RunPod jobs (after O2)

Verify first: these are not yet confirmed in the v2 spec.

- [ ] Read the full v2 OpenAPI spec; confirm the create-pod path, single-pod GET path, logs path, GPU
      catalog/price path, and API-key scopes. Save the excerpt the client uses to `tests/fixtures/`.
- [ ] Q10: confirm the pod id is available inside a pod and that a pod can terminate itself with a key
      delivered as a RunPod secret; decide on a scoped or dedicated low-privilege key.
- [ ] Q11: confirm whether a pod whose main process exited keeps billing.

Build:

- [ ] `docker/Dockerfile`: code at a tagged commit, pinned deps, pinned llama.cpp converter; build, push,
      record the digest.
- [ ] Private inputs dataset repo on Hugging Face with the calibration tokens and imatrix (Q13); pod-side pull.
- [ ] `ops/runpod.py`: thin v2 client (catalog, volumes, pods, actions, logs) with 429 backoff, cursor
      pagination, and `ERROR` handling.
- [ ] `ops/jobs.py`: job spec JSON under `jobs/`; no secrets in it.
- [ ] `nanoquant job submit`: check GPU availability in the volume's data center, show price, estimate,
      cap, and upload target, then require confirmation or `--yes`.
- [ ] `nanoquant job-run` (pod entrypoint):
  - [ ] `NANOQUANT_HOME` on the network volume; inputs pulled on first use;
  - [ ] `runs/<name>/owner.json` lease with pod id and minute heartbeat; refuse while another owner's
        heartbeat is under 10 minutes old;
  - [ ] run, release, upload; write `job-status.json`;
  - [ ] self-terminate on success or failure; watchdog terminates at `max_hours` / `max_cost_usd`.
- [ ] `nanoquant job watch | logs | resume | fetch | cancel`, and the Jobs menu items.
- [ ] Reports record GPU type, driver, CUDA, torch, and local vs job.
- [ ] Tests (CPU, no network): request bodies against the saved spec excerpt; 429 and pagination;
      lease refusal; no token in job spec, `run.json`, or logs.
- [ ] Paid smoke test: tiny 2-block job on the cheapest GPU → private test repo → pod terminates itself;
      check the pod shows `TERMINATED` and billing stopped.
- [ ] Interruption test: cancel a real job mid-run, `job resume` on a new pod, committed blocks reused.
- [ ] Full 085 recipe as a job; quality within the §11.2 bands (wall time and memory not compared).
- [ ] Replace `RUNPOD.md` in the new repo with a short guide for the job commands.

**Exit:** the 085 recipe runs end to end as a RunPod job, uploads its release, and terminates its pod
without the local machine involved after submit.

## Phase 7 — Retire

- [ ] User decision: archive this repository as read-only playground, or keep it active.
- [ ] User decision: archive or delete bulky data (§7.5), only after the gate passes. Use
      `tools/cleanup_artifacts.py` dry runs first; delete nothing while a run is active.
