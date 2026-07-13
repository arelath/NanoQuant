# Behavior-Preserving Optimization Catalog

This document catalogs concrete optimization candidates found by static inspection of the current pipeline,
restricted to changes that do **not** change behavior: identical committed artifacts, identical numerics,
identical decisions. Each finding names its location, explains why it is safe, and estimates the saving.
It complements [Performance profiling and micro-profiling](15-performance-profiling.md): every estimate
below is a hypothesis to be confirmed by that framework's tier-0/1 measurements before and after the change,
and every landed change must pass the fixture-replay identity test.

## 1. Method and anchors

Findings come from reading the hot paths (`resident_quantization.py`, `domain/factorization.py`,
`application/tuning.py`, `application/parity_adamw.py`, `application/quantization_stages.py`,
`application/distillation.py`, the tensor/activation stores) against the pinned parity workload.

Measured anchors (from `evidence/m4` run reports and `evidence/m0` legacy config):

- Gemma 3 1B: 26 blocks × 7 layers = **182 layers**; the legacy rank-utility log shows **238 factorization
  attempts** (retries included). Calibration set: 256 samples × 2048 tokens; block activations are
  256×2048×1152 bf16 ≈ **1.2 GB per stream**.
- `gemma-3-1b-it-parity-factor-scale` (ADMM 800 outer iterations, scale fit on, tuning off):
  **1439 s total, 1144 s (80%) factorization**, ~295 s everything else.
- `admm1` control runs: 87–89 s total — an upper bound on per-run fixed overhead (load, prefix capture,
  calibration, forwards, commits) of roughly 60–90 s.
- The full Experiment-019 protocol additionally enables per-layer factorized tuning (8 epochs, batch 1 →
  2048 optimizer steps per layer), non-factorized tuning (schedule 8,4,3,2,2,2,2), post-block refit
  (2 epochs), and global KD (8 epochs). None of that is exercised by the 1439 s anchor, so tuning-side
  findings are expressed as a share of their phase, not of 1439 s.

Estimates are stated as *phase share × expected reduction*, with confidence Low/Medium/High. They are
deliberately ranges; the profiler decides, these findings tell it where to look.

## 2. Safety classes

- **S0 — bitwise-identical.** Same values, same artifacts, same events. Removing redundant work, hoisting
  loop-invariant transfers, eliminating double copies. Landable with fixture replay alone.
- **S1 — schedule-only.** Same values and artifacts; only execution overlap, allocator state, or the
  durability window changes (e.g. async writes completed before the journal append, pinned-memory
  transfers, deferred synchronization). Landable with fixture replay plus an interruption-matrix rerun.
- **S2 — numerics-identical, observable surface differs.** Same numbers, but the artifact set, a report
  byte-count field, or log cadence changes (e.g. no longer persisting rejected attempts). Requires an
  explicit sign-off that the changed surface is not load-bearing.

Anything that would change floating-point results — reordered reductions, fused/batched GEMMs with
different tiling, `torch.compile`, TF32 policy changes, foreach RNG batching — is out of scope and listed
in section 6 so nobody drifts into it by accident.

## 3. Findings, ranked by expected impact

| Done | # | Finding | Class | Phase affected | Est. saving (phase) | Est. saving (run) | Confidence | Effort |
|------|---|---------|-------|----------------|--------------------|-------------------|------------|--------|
| [ ] | 1 | Pin + double-buffer tuning/forward datasets | S1 | tuning, block forwards | 25–50% of transfer-bound steps | 10–25% of full protocol | Medium | M |
| [x] | 2 | Foreach ParityAdamW step | S0 | tuning | 50–80% of optimizer host time | 2–6% of full protocol | High | M |
| [x] | 3 | Gate `torch.cuda.empty_cache()` on pressure | S1 | tuning, per-block | ~390 calls avoided | measured below | High | S |
| [ ] | 4 | Overlap block-activation persistence with compute | S1 | block commit | most of ~3–5 s/block | 4–7% of 1439 s anchor | Medium | M |
| [ ] | 5 | Halve/defer ADMM cholesky `info` syncs | S0* | factorization | 1–3% of ADMM | ~1–2% of anchor | Medium | S |
| [x] | 6 | Skip the always-zero self-reference MSE | S0 | per-block bookkeeping | ~0.3–1 s/block | 0.5–1.5% of anchor | High | S |
| [x] | 7 | Hoist per-microbatch `importance.to(device)` | S0 | tuning, block loss | ~0.1–0.2 s/layer | 0.5–2% of tuning phase | High | S |
| [x] | 8 | Fix `_run_block_batched` double copy | S0 | block forwards | one 1.2 GB copy per pass | 0.5–1.5% of anchor | High | S |
| [ ] | 9 | Stop persisting rejected-attempt tensors (or make puts async) | S2/S1 | factorization | store I/O ≈ halved | 1–3% of anchor | Medium | M |
| [ ] | 10 | Hash during write instead of write-then-reread (mmap store) | S0 | activation commits | one full re-read per generation | 1–3% of anchor | High | S |
| [ ] | 11 | Device-side KD loss accumulation + pinned teacher cache | S0/S1 | global KD | 1–3% of KD phase | ≤1% of full protocol | Medium | S |
| [x] | 12 | Keep the JSONL event file handle open | S0 | events | ~3k opens/run | ~0.1% | High | S |
| [x] | 13 | Device-side calibration threshold accumulation | S0 | calibration | ~20k small syncs | ~0.1–0.2% | High | S |

Combined outlook, avoiding double counting: roughly **8–15% on the factor+scale anchor** (items 4–6, 8–10)
and **15–35% on the full protocol** once tuning and KD phases exist to optimize (items 1–3, 7, 11 dominate).
Against the ~30% gap to legacy recorded in the agent guide, this catalog plausibly covers a large fraction —
but only the Docs/15 baseline can apportion it.

### [ ] 3.1 Pin and double-buffer the tuning/forward datasets (S1)

**Where.** `_run_block_batched` stores teacher/compressed activations into plain pageable CPU tensors
([resident_quantization.py:1079](../src/nanoquant/resident_quantization.py) `storage_device="cpu"`, same at
propagation, line 1398). Every consumer then streams them back per batch with
`.to(device, non_blocking=True)` — [tuning.py:52–53, 112–113](../src/nanoquant/application/tuning.py),
[resident_quantization.py:265–267, 292](../src/nanoquant/resident_quantization.py). `non_blocking=True` on
pageable memory is silently synchronous: each copy stages through a driver bounce buffer and blocks the
host.

**Why it matters.** With the legacy protocol's `fact_batch_size=1`, one factorized-tuning call is 2048
steps, each moving one 2048-token sample (11–28 MB input, depending on layer width, plus target) H2D, plus a
full-dataset `_evaluate_loss` pass per epoch. Per tune call that is ~18 dataset passes ≈ 40+ GB of pageable
traffic; there are ~390 tune calls in the full protocol (182 factorized + 182 non-factorized + 26 refits).
Compute per step is small (a single-layer or single-block forward/backward), so these phases are plausibly
transfer-stalled a large fraction of the time. The legacy implementation pinned CPU activations
(`pin_cpu_activations=True`); the rewrite has pinned support in `MemoryActivationStore`
([activation_store.py:37](../src/nanoquant/infrastructure/activation_store.py)) but the resident path does
not use it.

**Change.** Allocate the block-loop activation tensors (`teacher_inputs`, `teacher_outputs`,
`compressed_inputs/outputs`) pinned; in `tune`/`_evaluate_loss`/`_block_loss`/`_run_block_batched`, prefetch
batch *k+1* on a copy stream while batch *k* computes. Values are identical; only overlap changes.

**Estimate.** If transfers are 30–60% of step wall in these phases (to be confirmed by the Docs/15 micro
tier), overlap plus pinned bandwidth (~2–3× pageable) recovers most of it: **25–50% of tuning-phase wall;
10–25% of a full-protocol run**. Also applies to `_block_loss` passes and block forwards at batch 4
(smaller share). Confidence: Medium (transfer-boundedness inferred, not yet measured). Effort: M. Pinned
memory pressure must respect the existing resource-planning rules (1.2 GB × up to 4 live streams).

**Pinned half done (2026-07-12).** CUDA-produced CPU activation streams are now allocated directly in
pinned host memory, and activation streams loaded at a CUDA resume boundary are pinned once. The tuning
loop already requests nonblocking H2D copies, so every subsequent factorized, non-factorized, refit, and
loss pass reuses those pinned tensors without an extra copy. On the parity GPU, an exact 37.75 MiB BF16
batch transferred in a median 3.045 ms pinned versus 3.997 ms pageable (**1.31x**); blocking D2H improved
only 1.05x. A CUDA regression test confirms pinned block outputs are bitwise equal. This does not check the
item off: double buffering is deferred until copy-stream lifetimes and the four-stream pinned-memory budget
are represented by the resource plan.

### [x] 3.2 Foreach ParityAdamW (S0)

**Where.** [parity_adamw.py:66–93](../src/nanoquant/application/parity_adamw.py) — a Python loop over
parameters issuing ~8–10 elementwise kernels each (`mul_`, `lerp_`, `addcmul_`, fresh `sqrt().add_()`
allocation at line 85, Kahan sequence at 87–91).

**Why it matters.** 2048 steps per tune call × ~4–7 selected tensors × ~10 launches ≈ 10⁵ launches plus
Python dispatch per layer, repeated for 390 tune calls. Optimizer host time likely rivals the actual
backward for the small per-layer parameter sets.

**Change.** Rewrite the update with `torch._foreach_*` ops (including the Kahan branch:
`_foreach_copy_`, `_foreach_addcdiv_`, `_foreach_add_`, `_foreach_sub_`). Foreach ops apply the same
elementwise arithmetic per tensor with no cross-tensor reduction, so results are bitwise-identical; the
existing `denominator` allocation can also be reused via preallocated state buffers. A one-time unit test
asserting bitwise equality against the current loop implementation (both dtypes, with/without Kahan, several
steps) locks this in.

**Estimate.** 50–80% of optimizer host time; **2–6% of a full-protocol run** (scales with tuning share).
Confidence: Medium-High. Effort: M.

**Done (2026-07-12).** `ParityAdamW` groups parameters by device, dtype, and Kahan mode, applies the legacy
recurrence with `torch._foreach_*`, and reuses a state-owned denominator buffer. Tuning also resolves the
model device once per call instead of once per batch. The scalar-loop oracle is bitwise-equal for FP32 and
BF16/Kahan, with and without weight decay, on CPU and CUDA across multiple tensors and steps. On a
representative five-tensor BF16/Kahan CUDA workload, 128 optimizer steps improved from a median 0.0735 s
to 0.0322 s (**2.28x**).

### [x] 3.3 Gate `torch.cuda.empty_cache()` on memory pressure (S1)

**Where.** [tuning.py:147](../src/nanoquant/application/tuning.py) — every `tune()` call ends with
`empty_cache()`; also per block at
[resident_quantization.py:1115](../src/nanoquant/resident_quantization.py), post-prefix at line 862, resume
boundary at 1052, calibration at [calibration.py:311](../src/nanoquant/application/calibration.py), and
twice in `global_distillation.py`.

**Why it matters.** `empty_cache()` returns cached blocks to the driver; the next phase re-pays `cudaMalloc`
for gigabytes of workspace. At ~390 tune calls plus 26 block-loop calls in the full protocol, each costing
tens to low hundreds of milliseconds (release + re-warm), this is minutes of pure allocator churn. The
call sites exist to protect against fragmentation-induced OOM at phase boundaries (the comments say so), not
for correctness.

**Change.** Replace unconditional calls with a pressure-gated helper: only flush when
`torch.cuda.memory_reserved() - memory_allocated()` exceeds a threshold fraction of device memory, or keep
the boundary flushes that the resume/OOM-fallback design actually depends on (resume boundary, prefix
teardown) and drop the per-tune one. Numerics are untouched; only allocator state differs. The
OOM-fallback path (`on_cuda_oom`) remains as the safety net, so a wrong threshold degrades to today's
behavior, not to failure.

**Estimate.** **1–4% of a full-protocol run**; near zero on the factor+scale anchor. Confidence: Medium.
Effort: S. Verify with the interruption/OOM-injection tests, not just replay.

**Done (2026-07-12).** Per-tune cleanup now retains the allocator cache unless reserved memory reaches
80% of device capacity. Prefix, resume, calibration, global-distillation, and per-block boundary flushes
remain unchanged, so the existing coarse fragmentation/OOM safeguards still run. On the parity GPU, 30
representative 256 MiB allocation cycles took a median 0.829 ms when reusing cached storage and 8.159 ms
after a flush, a **9.84x re-warm penalty** (excluding the flush call itself). At 390 tune calls this avoids
at least about 2.9 seconds of re-warm work; the original 1–4% run estimate was too optimistic. Pressure
boundary tests and the tuning/optimizer suite pass.

### [ ] 3.4 Overlap block-activation persistence with compute (S1)

**Where.** Block commits persist both activation streams for resume (`commit_block` /
`load_block_activations`, [commits.py:118,170](../src/nanoquant/infrastructure/commits.py)); rolling
retention then retires the previous block's copies. Write traffic ≈ 2 × 1.2 GB per block ≈ **62 GB per
run**, each byte also SHA-256 hashed (~1.5–2 GB/s single-threaded).

**Why it matters.** ~2.4 GB write + hash per block ≈ 2–5 s × 26 blocks ≈ 60–120 s, serialized between
propagation and the next block's teacher forward.

**Change.** Start the activation serialization+hash on a worker thread as soon as `compressed_outputs`
exists; block on completion immediately before the corresponding `journal.append("block", ...)`. Artifact
bytes, hashes, and the durable-before-journal ordering are unchanged; the write simply overlaps the next
block's GPU work. (The same applies to the per-layer `commit_layer` factor writes, which are ~20 MB each —
included in the estimate's low end.)

**Estimate.** Hides most of the 60–120 s: **4–7% of the 1439 s anchor**; smaller share of longer runs.
Confidence: Medium (depends on how much compute is available to overlap). Effort: M — needs a small
single-worker writer with strict completion-before-journal semantics, exercised by the existing
interruption matrix.

### [ ] 3.5 Reduce ADMM cholesky `info` synchronizations (S0, one caveat)

**Where.** [factorization.py:113](../src/nanoquant/domain/factorization.py):
`int(info.max())` after every `cholesky_ex` — a host-device sync **twice per outer iteration**, i.e. 1600
per attempt × 238 attempts ≈ 380k syncs per run.

**Why it matters.** Each sync drains the launch pipeline. For the wide MLP layers the iteration is
GEMM-bound and the stall is partially hidden; for the 1152×1152 attention layers the iteration is
launch-bound and stalls bite. The convergence check already syncs deliberately once per
`convergence_check_interval` (100) — these two are extra.

**Change (two steps).**
1. *Halve:* run both solves' `cholesky_ex` first, check both infos with one combined `.item()` per
   iteration. Strictly identical behavior; 800 syncs instead of 1600.
2. *Eliminate:* compute `cholesky_solve` speculatively, and only consult `info` to decide whether to
   *discard* that result and recompute with `torch.linalg.solve`. When `info == 0` (the always-observed
   case — the system is regularized SPD by construction, line 107–108), results are bit-identical and the
   check can be batched or made lazy. The caveat: if a cholesky ever fails, the speculative variant does
   wasted work but still returns exactly what today's code returns — behavior is preserved in both branches;
   only the failure path's cost changes.

**Estimate.** 1–3% of the ADMM phase ≈ **10–35 s of the anchor**; more on launch-bound small layers
(the Docs/15 wall-vs-CUDA divergence flag will show exactly how much). Confidence: Medium. Effort: S.

**Inspected, not accepted (2026-07-12).** The proposed one-sync variant is not executable without changing
the recurrence: the right-hand solve consumes `left` from the left-hand solve, so both Cholesky
factorizations cannot be launched before checking the first result. Deferring either `info` check also
changes the documented fallback behavior because a failed factorization would feed invalid values into
later operations before `torch.linalg.solve` could replace it. Always trusting regularization to make the
systems SPD would remove the syncs, but changes failure behavior and is therefore outside S0. Keep this
item unimplemented unless profiling justifies an explicitly behavior-changing policy decision.

### [x] 3.6 Skip the always-zero self-reference MSE (S0)

**Where.** [resident_quantization.py:1098](../src/nanoquant/resident_quantization.py):
`record_source_reference(_weighted_mse(teacher_outputs, teacher_outputs, ...))` — computes the weighted MSE
of a tensor against itself. `_weighted_mse` (lines 238–246) runs a 256-iteration Python loop of
subtract/square/mul/sum over 1.2 GB, **on CPU** (both tensors live there), per block.

**Why safe.** For finite inputs, IEEE-754 guarantees `x − x = 0`, so the result is exactly `0.0`. The only
input that changes this is a non-finite activation (then the metric is NaN). Replace the computation with a
single `torch.isfinite(teacher_outputs).all()` scan: finite → record `0.0` (bit-identical), non-finite →
fall back to the full computation. One pass instead of four-plus, and the common case is a cheap reduction.

**Estimate.** ~0.3–1 s per block × 26 ≈ **10–25 s, 0.5–1.5% of the anchor**. Confidence: High. Effort: S.

**Done (2026-07-12).** Added `_self_reference_weighted_mse` in `resident_quantization.py`, called at the
former call site instead of `_weighted_mse(teacher_outputs, teacher_outputs, ...)`. All 162 tests, ruff, and
`mypy --strict` pass.

### [x] 3.7 Hoist loop-invariant `importance.to(device)` (S0)

**Where.** [tuning.py:40](../src/nanoquant/application/tuning.py) — `_loss_sum` re-materializes
`importance.to(device, dtype)` on every microbatch (train and eval), i.e. ~4000+ times per tune call when
the importance tensor lives off-device; [resident_quantization.py:270](../src/nanoquant/resident_quantization.py)
does the same inside `_block_loss`'s batch loop.

**Change.** Resolve the device/dtype copy once per `tune()` / `_block_loss()` call and reuse it. `.to()`
returns a fresh copy of the same values each time — hoisting is bit-identical.

**Estimate.** A small pageable H2D plus sync per step; ~0.1–0.2 s per tune call ≈ **30–80 s across a full
protocol** (and it removes per-step sync points that item 1's overlap would otherwise trip on).
Confidence: High. Effort: S.

**Done (2026-07-12).** Added `_resolve_output_importance` helper in `tuning.py`; hoisted the resolved
importance out of `_evaluate_loss`'s microbatch loop and `tune()`'s per-step microbatch loop, and out of
`_block_loss`'s batch loop in `resident_quantization.py`. All 162 tests, ruff, and `mypy --strict` pass.

### [x] 3.8 Remove the `_run_block_batched` double copy (S0)

**Where.** [resident_quantization.py:300](../src/nanoquant/resident_quantization.py):
`result[start:end].copy_(output.to(destination))` — `output.to(destination)` allocates a full intermediate
on the destination, then copies it into the slice. Same pattern in `_run_prefix_batched` (line 327).

**Change.** `result[start:end].copy_(output)` performs the cross-device copy directly into the
preallocated slice — one copy instead of two, identical bytes. (Combined with item 1, the destination
becomes pinned and the copy becomes async.)

**Estimate.** Saves one full activation-sized copy per pass; with ~3–4 full passes per block, **~10–25 s
per anchor run (0.5–1.5%)**. Confidence: High. Effort: S.

**Done (2026-07-12).** `_run_block_batched` and `_run_prefix_batched` in `resident_quantization.py` now do
`result[start:end].copy_(output)` instead of `copy_(output.to(destination))`. All 162 tests, ruff, and
`mypy --strict` pass.

### [ ] 3.9 Stop persisting rejected-attempt tensors, or persist asynchronously (S2 / S1)

**Where.** Every attempt writes its outputs through `LocalTensorStore.put`
([tensor_store.py:33–38](../src/nanoquant/infrastructure/tensor_store.py)): outlier selection persists the
full residual weight per attempt ([quantization_stages.py:120–130](../src/nanoquant/application/quantization_stages.py)),
factorization persists 7 factor tensors (lines 214–225), scale fit 3 more; the attempt path then reads many
of them straight back ([resident_quantization.py:503–514](../src/nanoquant/resident_quantization.py)). Each
put is a synchronous D2H `.cpu().clone()` plus SHA-256 plus safetensors write. Per attempt ≈ 35–50 MB
written+hashed; ×238 attempts ≈ 8–12 GB, consistent with the 12.4 GB
`artifact_bytes_before_report` in the parity report.

**Change, two flavors.**
- *S1 (fully identical):* keep writing everything, but hand tensors to downstream stages in-memory
  (device-resident) and push the disk persistence to a background writer that completes before
  `commit_layer`. Artifact set and hashes unchanged.
- *S2 (smaller disk footprint):* persist only the accepted attempt's tensors. Numerics identical, but
  rejected-attempt artifacts no longer exist on disk and `artifact_bytes_before_report` shrinks — an
  observable-surface change that needs sign-off (it also reduces GC pressure,
  cf. [14-artifact-retention-and-disk-usage.md](14-artifact-retention-and-disk-usage.md)).

**Estimate.** Store traffic ≈ halved (retries + rejected attempts) and round-trips off the critical path:
**1–3% of the anchor**, plus reduced allocator/sync churn inside `execute_attempt`. Confidence: Medium.
Effort: M.

### [ ] 3.10 Hash during write instead of write-then-re-read (S0)

**Where.** [activation_store.py:67–72](../src/nanoquant/infrastructure/activation_store.py):
`MmapGenerationWriter.commit` writes the full mapping, then `_hash_file` re-reads the entire file from disk
to compute the digest — an extra full read of every committed activation generation (2.4 GB per block on
the parity workload when the mmap tier is active; the same pattern is the fallback for any store that
hashes files post hoc).

**Change.** Feed the same bytes to the hasher as they are written (or hash the mapped buffer before
`close`). The digest is over identical bytes, so artifact identity is unchanged.

**Estimate.** Removes a full-file read per generation: **15–40 s per anchor-scale run where the mmap tier
is used (1–3%)**; also cuts page-cache pressure alongside item 4. Confidence: High. Effort: S.

**Attempted, not accepted (2026-07-12).** An incremental SHA-256 implementation preserved identities and
the out-of-order-write fallback in unit-sized tests, but a 128 MiB Windows mmap benchmark failed to finish
within a 10-minute cap. Because it did not produce a trustworthy before/after measurement and could add
critical-path page reads to every `write`, it was reverted. Revisit only with chunk-level profiling and a
bounded benchmark before changing this checkbox.

### [ ] 3.11 KD step-loop hygiene (S0/S1)

**Where.** [distillation.py:387–405](../src/nanoquant/application/distillation.py): per step,
`cpu_tokens.index_select(...).to(device)` and three teacher-target `.to(device)` transfers from pageable
cache, then `total_loss += float(loss.detach())` — a hard sync every step (2048 steps × 8 epochs).

**Change.** Pin the cached epoch tensors once (S1, same values), prefetch the next step's batch on a copy
stream, and accumulate `loss.detach()` into a float64 device scalar, converting once per epoch. Python's
`total_loss` is already a float64 sequential sum; a float64 device accumulator adds the same values in the
same order, so the recorded `epoch_losses` are bit-identical.

**Estimate.** 1–3% of the KD phase (≤1% of a full-protocol run). Confidence: Medium. Effort: S.

**Partially tested, not accepted (2026-07-12).** Replacing 2,048 per-step Python-float additions with
sequential float64 device additions preserved the final double bit-for-bit, but improved the isolated
accumulation loop only from 21.878 ms to 16.859 ms (**1.30x, just 5 ms absolute**) while adding one CUDA
kernel per step. That does not justify the extra device work or support the estimated run-level saving, so
device-side accumulation was not implemented. Pinned teacher-cache transfer and prefetch remain deferred
to the shared pinned/double-buffer design in item 1; implementing them independently would duplicate the
same memory-budget and copy-stream machinery.

### [x] 3.12 Keep the JSONL event file handle open (S0)

**Where.** `JsonlEventSink.emit` ([events.py](../src/nanoquant/infrastructure/events.py)) opens, appends,
flushes, and closes the file for every event; a parity run emits a few thousand events (per-attempt stage
events dominate). Windows file opens are ~0.1–0.5 ms.

**Change.** Open once, `flush()` per emit as today (no fsync semantics change), close on finalize.
**Estimate.** ~0.5–1.5 s per run (~0.1%). Confidence: High. Effort: S. Do it opportunistically alongside
Docs/15 P0, which touches the same file.

**Done (2026-07-12).** `JsonlEventSink` now opens the handle once in `__init__` and reuses it in `emit()`;
added `close()` plus context-manager support, called at the natural finalize point in
`_run_resident_quantization`, `_run_resident_factorization_slice` (in its `finally`), and
`run_tiny_pipeline`. Verified empirically on Windows that a persistent append-mode handle does not block a
concurrent reader or a second append-mode handle to the same path (load-bearing for
`test_events_are_monotonic_across_reopen_and_spans`, which opens a second sink on the same path). All 162
tests, ruff, and `mypy --strict` pass.

### [x] 3.13 Device-side calibration threshold accumulation (S0)

**Where.** [calibration_math.py:48](../src/nanoquant/domain/calibration_math.py) and the profiling hooks in
[calibration.py:225–235](../src/nanoquant/application/calibration.py): `robust_tau(...).cpu()` and
`torch.maximum(cpu, new.cpu())` run per layer per batch — ~20k small D2H syncs per calibration pass.

**Change.** Keep the running maxima on device, move to CPU once at finalize. Elementwise `maximum` is exact
and order-insensitive here, so results are bit-identical.

**Estimate.** 1–3 s of the calibration phase (~0.1–0.2% of a run). Confidence: Medium. Effort: S.

**Done (2026-07-12).** Both causal-model and block two-phase Fisher hooks now initialize their scalar
threshold maxima on each layer's device and update them without `.cpu()`. CUDA validation showed the
device maximum, the former host maximum, and the resulting clipped importance arithmetic are bitwise
equal; a CUDA two-phase calibration completed with finite input and output importance.

## 4. Smaller observations (bundle opportunistically)

- [x] `next(iter(model.parameters()), None)` previously executed per batch step in `tune`; the foreach
  optimizer change now resolves the device once per tuning call.
- `FactorizationAttemptStage` computes `latent_prediction=left_latent @ right_latent` per attempt for
  metrics ([quantization_stages.py:212](../src/nanoquant/application/quantization_stages.py)) — necessary,
  but a candidate to fold into the attempt-level metrics pass if the profiler shows it (~0.05% each).
- `JsonlEventSink._read_last_sequence` parses the whole event log at construction — only matters for
  resumed runs with large logs; fine today, worth a tail-scan if event volume grows.
- `_artifact_bytes` walks the whole artifact tree once at report time — keep an eye on it as artifact
  counts grow (currently once per run).

## 5. What was checked and found already efficient

For future readers: these were inspected and are *not* wasteful under the parity contract.

- The per-sample Python loops in `_mse`/`_weighted_mse` and per-sample `total +=` accumulation look
  batchable, but batching changes reduction order and therefore bits — they encode legacy accumulation
  semantics. Leave them (see item 3.6 for the one call where the *result* is knowable without computing).
- The outlier residual probe runs a real 80-iteration ADMM per attempt (~10% of factorization). Its seeds
  differ per attempt by design (`logical_seed(..., attempt)`), so probe results are not cacheable across
  attempts without changing selections.
- Scale-fit's "before" metrics are computed in float32 while the factorizer's internal reconstruction is
  bf16 — they are different numbers by design, not a redundant recomputation.
- `_factor_slice_source_inventory` already caches source/inventory across slice invocations.
- The distillation teacher cache already precomputes per-epoch top-k targets instead of re-running the
  teacher (the expensive part is done once).

## 6. Explicitly out of scope (behavior-changing)

Do not reach for these while parity is the gate; each changes floating-point results or decisions:

- fusing/batching GEMMs (e.g. concatenating `system` and `rhs` solves) — different cuBLAS tiling, different
  bits;
- `torch.compile`, `cudnn.benchmark`, TF32 policy changes (TF32 is already pinned by
  `_legacy_cuda_numerics` for parity);
- fused/`foreach` *RNG* (batching the per-iteration `randn` draws in `_power_iteration`): philox offset
  alignment suggests a batched draw *may* be bit-identical for the shapes involved (all divisible by 4),
  but this depends on generator internals — speculative; only with a dedicated bitwise verification
  harness;
- replacing SHA-256 with a faster hash (artifact identity is schema surface);
- reducing probe iterations, ADMM iterations, epochs, or early-stop thresholds (quality/behavior);
- batching exact causal-evaluation windows: measured batch 8 at 1.092 s versus serial at 4.857 s
  (**4.45x**), but total NLL changed from 49,559.943115 to 49,554.102051 and peak GPU memory rose from
  2.30 GB to 4.63 GB. It is useful only as an explicitly approximate evaluation mode, not for the pinned
  parity protocol;
- skipping `verify_hashes` on loads (it is a documented integrity behavior; make it a config choice, not an
  optimization).

## 7. Verification protocol

Every item, regardless of class, lands with:

1. the fixture-replay identity test (committed artifact hashes equal to an unoptimized control) —
   for S0/S1 this must be bitwise;
2. a before/after `profile.json` on the parity workload attributing the claimed phase saving
   (Docs/15 tooling; until P0 lands, a targeted `perf_counter` harness is acceptable);
3. for S1 items: one pass of the interruption/resume matrix (the durability windows they touch are exactly
   what that matrix exercises);
4. for S2 items: an explicit note in the PR of the observable surface that changed, and agreement that no
   consumer depends on it.

## 8. Suggested sequencing

1. **Now (S0, small):** items 6, 7, 8, 12 — trivial diffs, measurable individually, no design risk.
   **Done (2026-07-12).**
2. **With Docs/15 P0 baselines in hand:** items 5 and 2 (verify the sync/launch-bound hypothesis first),
   then 1 (largest but needs the transfer-boundedness numbers to size the buffering).
3. **Design-reviewed (S1 durability):** items 4, 9, 10 as one "store path" workstream — shared writer
   infrastructure, shared interruption-matrix validation.
4. **When tuning/KD phases are enabled in anger:** items 3 and 11, measured on the full protocol.

Items 1–4 of this sequence are independent of parity sign-off (all preserve behavior by construction);
only their *measurement* waits for the Docs/15 P1 baseline if we want clean before/after evidence.
