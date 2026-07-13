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

`Done` means implemented and retained. `Rejected` means a measured candidate was reverted or deliberately not
implemented. `Deferred` means the candidate needs a contract decision or stronger profile evidence. The detailed
disposition under each unchecked item is authoritative; unchecked items are not silently pending implementation.

| Disposition | # | Finding | Class | Phase affected | Est. saving (phase) | Est. saving (run) | Confidence | Effort |
|-------------|---|---------|-------|----------------|--------------------|-------------------|------------|--------|
| Done | 1 | Pin + double-buffer tuning/forward datasets | S1 | tuning, block forwards | measured below | full-run rerun pending | High | M |
| Done | 2 | Foreach ParityAdamW step | S0 | tuning | 50–80% of optimizer host time | 2–6% of full protocol | High | M |
| Done | 3 | Gate `torch.cuda.empty_cache()` on pressure | S1 | tuning, per-block | ~390 calls avoided | measured below | High | S |
| Rejected | 4 | Overlap block-activation persistence with compute | S1 | block commit | measured 1.16x over the safe overlap window | ~17 s over 26 blocks | High | M |
| Rejected | 5 | Halve/defer ADMM cholesky `info` syncs | S0* | factorization | proposed launch order is not recurrence-preserving | none accepted | High | S |
| Done | 6 | Skip the always-zero self-reference MSE | S0 | per-block bookkeeping | ~0.3–1 s/block | 0.5–1.5% of anchor | High | S |
| Done | 7 | Hoist per-microbatch `importance.to(device)` | S0 | tuning, block loss | ~0.1–0.2 s/layer | 0.5–2% of tuning phase | High | S |
| Done | 8 | Fix `_run_block_batched` double copy | S0 | block forwards | one 1.2 GB copy per pass | 0.5–1.5% of anchor | High | S |
| Deferred | 9 | Stop persisting rejected-attempt tensors (or make puts async) | S2/S1 | factorization | requires artifact-contract approval or stronger I/O attribution | none accepted | Medium | M |
| Rejected | 10 | Hash during write instead of write-then-reread (mmap store) | S0 | activation commits | bounded Windows benchmark did not finish | none accepted | High | S |
| Rejected | 11 | KD loss accumulation + teacher-cache prefetch | S0/S1 | global KD | measured ≤0.02% | negligible | High | S |
| Done | 12 | Keep the JSONL event file handle open | S0 | events | ~3k opens/run | ~0.1% | High | S |
| Done | 13 | Device-side calibration threshold accumulation | S0 | calibration | ~20k small syncs | ~0.1–0.2% | High | S |
| Done | 14 | Cache verified tensor hashes by immutable file signature | S0 | tensor loads | repeated memory hashes removed | ~10–20 s | High | S |
| Done | 15 | Bypass STE signs for immutable binary KD factors | S0 | global KD | measured 24% per layer-step | full-run rerun pending | High | S |
| Done | 16 | Fuse ADMM factor promotion into FP32 additions | S0 | factorization | measured 2.1% per solve | ~5–6 s of anchor | High | XS |
| Done | 17 | Reuse ADMM symmetrization storage | S0 | factorization | measured 3.6% per solve | ~9–10 s of anchor | High | XS |
| Done | 18 | Lower-allocation binary sign extraction | S0 | factorization, factorized tuning | measured 1.21–1.92x per sign | ~3–12 s factorization, tuning rerun pending | High | XS |

The retained factor+scale changes currently support an estimated **roughly 3–7% anchor improvement** before
interaction effects; the earlier 8–15% outlook incorrectly counted rejected/deferred store and synchronization
items. The full-protocol tuning changes still have larger measured phase-local gains, but their end-to-end share
remains pending a clean, protocol-matched rerun. Only the Docs/15 baseline can apportion the remaining gap.

### [x] 3.1 Pin and double-buffer the tuning/forward datasets (S1)

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
only 1.05x. Shuffled advanced indexing was found to discard pinning, so training now gathers into two
reusable pinned buffers and waits only for each buffer's prior H2D event before reuse. Across 32 exact
two-tensor shuffled transfers this reduced 0.4051 s to 0.2688 s (**1.51x**) with bitwise-equal device
values and tuning parameters. Evaluation loss passes now prefetch one pinned input/target pair on a copy
stream while the prior pair computes, using two reusable readiness events. A representative BF16
1152×1152 forward/loss pass improved from 0.0721 s to 0.0545 s (**1.32x**) with an identical accumulated
loss. The iterator is now shared by block-loss snapshots and block forwards. On a representative
seven-layer BF16 block, block loss improved from 0.1169 s to 0.0605 s (**1.93x**) and host-output block
forward from 0.1136 s to 0.0613 s (**1.85x**), with bitwise-equal loss and outputs. CUDA regression tests
cover every landed path. Shuffled training now schedules microbatch k+1 on the same bounded copy stream
before microbatch k's forward/backward, without moving optimizer or scheduler steps. Across 32
representative three-layer BF16 forward/backward steps this reduced 0.6408 s to 0.2826 s (**2.27x**) with
identical loss; the full tuning oracle also produces bitwise-identical parameters and metrics. The two-slot
design adds only one future input/target pair, about 75 MiB at parity batch 8, and completes this item. A
full Gemma rerun is still required to replace the original run-level estimate with an end-to-end number.

**Device-backpressure correction validated (2026-07-13).** Live sampling of the retained-Fisher
long-tuning run exposed a flaw in that two-slot claim. Over 58 seconds, driver-visible GPU use climbed from
3,029 to 11,673 MiB while Windows private commit climbed from 17.74 to 25.87 GiB, then both fell to 5,840 MiB
and 20.23 GiB; the 7.89 GiB working set and 26.31 GiB lifetime private high-water stayed flat. This is a
bounded allocation sawtooth rather than a cross-layer leak, but the pinned stager was waiting only for a
slot's H2D-ready event. It could therefore allocate later device batches after the copy completed while the
compute stream still consumed earlier batches. The correction allocates exactly two fixed device input/target
pairs, records a compute-completion event after each backward, and makes the copy stream wait on that event
before overwriting a device slot. The host waits only for the prior H2D-ready event before refilling the matching
pinned host buffer, retaining one-batch copy/compute overlap without creating disposable device tensors.

On a 64-sample, 2,048-token, width-1,152 BF16 staging canary with eight parity-sized batches, the old path
reserved 360 MiB for 144 MiB of live pairs; fixed slots reserved and allocated exactly 144 MiB (**60% less
reservation**). Across five runs with deliberately delayed compute, median wall was 0.08381 s old versus
0.08496 s fixed-slot (**1.4% difference**, within the short-canary spread), whereas the initial host-synchronized
backpressure prototype took 0.10667 s and was rejected. All 17 focused CUDA tests pass, including bitwise
pinned-versus-pageable tuning and the block batching oracles; CPU/static and full-suite checks also pass. A
resumed real run remains the final driver-visible peak confirmation.
This does not change `RESIDENT_ALGORITHM_VERSION`: it changes only when a reusable H2D staging slot may allocate
again; compute-stream operation order, batches, RNG, arithmetic, optimizer steps, and committed semantics are
unchanged. The CUDA bitwise check passed that compatibility gate.

**Generic evaluation/forward prefetch correction validated (2026-07-13).** The fixed shuffled-training stager
removed about 5.5 GiB of Windows/CUDA commit from the live run, but driver-visible use still touched 11.63 GiB.
Per-four-batch allocator tracing isolated the remaining growth to `iter_device_batches`: training stayed at
2.59 GiB allocated and 8.14 GiB reserved, then each epoch evaluation added about 0.9 GiB of reservation
(8.14 → 9.05 → 9.95 GiB) without increasing live allocation. The generic iterator had the same defect: its
two readiness events did not represent two reusable device slots, so every evaluation allocated a new pair.

The iterator now owns two fixed device-buffer tuples. A consumed event gates copy-stream overwrite of each slot,
while the existing ready event gates compute; this preserves one-batch overlap and works for evaluation loss,
block-loss snapshots, and resident block forwards. On a two-tensor 64×1,024×512 BF16 CUDA canary, incremental
reservation fell from 162 to 42 MiB (**74% less**) with median wall 0.06236 s versus 0.06011 s old across five
runs. The direct iterator and all downstream CUDA bitwise tests pass. More importantly, a full 32-epoch real
Gemma attention-layer diagnostic held reservation at 7,346–7,348 MiB with 2,176 MiB maximum allocation, zero
allocator retries, and zero OOMs; preceding non-factorized schedules shared the same 7,338 MiB plateau. The
driver-visible sample was about 8.0 GiB, leaving roughly 4 GiB of physical headroom instead of paging against
the 12 GiB limit.

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

**Measured, not accepted (2026-07-12).** Persisting one real 2.416 GB block activation generation took
2.167 s to write and 1.800 s to hash (3.968 s total). The final weighted metric scan available for safe
same-block overlap took only 0.760 s; running both concurrently reduced their combined 4.727 s to 4.082 s
(**1.16x**), about 17 seconds over 26 blocks. Cross-block overlap could hide more but would reorder durable
block and next-block layer commits, complicate resume discovery, and require thread-safe store composition.
That risk is not justified by the measured bound, so no asynchronous persistence code was added.

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

**Deferred (2026-07-12).** The S2 variant intentionally changes the retained evidence surface and report
byte count, so it will not be implemented without explicit approval of that contract change. The S1
variant needs the same thread-safe artifact-writer and tensor-lifetime machinery as item 4; the real-block
measurement there found too little hideable store time to justify that infrastructure yet. Revisit if
micro-profiling attributes a material share of `factorize-attempt` wall time to `LocalTensorStore.put`.

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

**Tested, not accepted (2026-07-12).** Replacing 2,048 per-step Python-float additions with
sequential float64 device additions preserved the final double bit-for-bit, but improved the isolated
accumulation loop only from 21.878 ms to 16.859 ms (**1.30x, just 5 ms absolute**) while adding one CUDA
kernel per step. That does not justify the extra device work or support the estimated run-level saving, so
device-side accumulation was not implemented.

A bounded two-slot CUDA prototype staged the token IDs, selected-token indices, teacher values, and teacher
vocabulary indices into pinned buffers and prefetched step *k+1* while step *k* computed. It preserved the
epoch loss and updated parameters bit-for-bit. Across 128 parity-shaped transfers (one 2048-token sample,
512 selected tokens, and top-64 bf16/int32 targets), it improved 50.168 ms to 29.799 ms (**1.68x**)—only
**0.159 ms per step**. A deliberately compute-light four-layer, width-1152 surrogate improved from a mean
0.6546 s to 0.6225 s across 128 steps (**1.05x**), but it spent only about 5 ms per step. The real Gemma KD
checkpoints show roughly 0.8 s per step, so even the transfer-only saving is an upper bound of about
**0.02% of the KD phase** (roughly 0.3 s over all 2,048 steps). The prototype was reverted: its extra
copy-stream/event machinery and pinned buffers are not justified by that run-level bound. Both parts of
this item are therefore rejected unless a future profile shows a materially different transfer share.

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

### [x] 3.14 Cache tensor verification by immutable file signature (S0)

**Where.** `LocalTensorStore.read` validates the artifact file and then recomputes the referenced tensor's
SHA-256 on every read. Factor outputs are commonly reread by scale fitting, trainable rehydration, and
freezing within the same process.

**Done (2026-07-12).** `LocalTensorStore` now remembers a successful tensor verification by artifact ID,
tensor key, declared content hash, file size, and nanosecond mtime. A store that just wrote the immutable
artifact seeds this entry from the exact tensor hashes computed for its returned refs; a reopened store
still hashes once. Every read continues to call artifact validation, and a changed file signature forces
full validation and tensor rehashing. Hashing all seven tensors in a real 51.1 MB factor artifact took a
median 20.2 ms; eliminating two to three repeats across 321 attempts bounds the saving at roughly
10–20 seconds. Tests prove reopened-store behavior and that cached tensor verification does not bypass
artifact corruption detection.

### [x] 3.15 Bypass STE signs for immutable binary KD factors (S0)

**Where.** Global distillation thaws each committed `left_binary`/`right_binary` tensor into a
`TrainableFactorizedLinear`, but selects only scales, outlier values, biases, and norm parameters for
optimization. Even after `distill_topk` disables gradients on both factors, every forward calls `_SignSTE`
and materializes two full-size `torch.where` results. The inputs are already exactly ±1 and cannot change
during this phase, so those values are redundant.

**Done (2026-07-12).** The global-KD thaw path now marks its factors as immutable binary values. Forward
uses those parameters directly only while they do not require gradients; ordinary factorized tuning still
uses `_SignSTE`, including when a marked module is later made trainable. This adds no cached tensor and no
VRAM. On a parity-shaped 1152×448×1152 bf16 layer with 2,048 tokens, 16 forward/backward scale-training
steps averaged 14.345 ms with redundant STE materialization versus 10.941 ms on the direct-binary path
(**1.31x**, or **24% less layer-step wall**). Outputs, loss, and all scale gradients were bitwise equal.
Unit coverage verifies both the frozen-factor fast path and the trainable-factor fallback; the global
distillation integration path also passes. A full Gemma rerun is still required for the run-level saving.

### [x] 3.16 Fuse ADMM factor promotion into FP32 additions (S0)

**Where.** `_solve` converted both `projected` and `dual` from bf16 to float32 into standalone temporary
tensors immediately before adding them to the already-float32 right-hand side. `Tensor.add_` promotes each
bf16 input to its float32 destination dtype exactly, so the conversion kernels and allocations were
redundant.

**Done (2026-07-12).** Both additions now consume the bf16 tensors directly. A parity-shaped
1152×448 design / 448×1152 right-hand-side CUDA solve produced bitwise-identical bf16 output. Across six
alternating 256-solve samples, median wall time improved from 183.867 ms to 180.057 ms (**1.021x**), with
matching CUDA-event time. At 800 iterations, two solves per iteration, and 238 attempts, the measured
per-solve delta projects to roughly **5–6 seconds on the anchor**. The existing exact bf16 recurrence
oracle protects the operation-order result.

### [x] 3.17 Reuse ADMM symmetrization storage (S0)

**Where.** `_solve` formed `0.5 * (system + system.mT)` as two out-of-place elementwise operations after
the Gram matrix, allocating a second rank² result solely for multiplication by 0.5. The addition result is
fresh and has no other consumer, so the multiplication can update it in place with identical arithmetic.

**Done (2026-07-12).** Symmetrization now adds out of place, then calls `mul_(0.5)` on that result. A
parity-shaped 1152×448 design / 448×1152 right-hand-side solve remained bitwise identical. In a longer
alternating benchmark, median time for 512 solves improved from 381.324 ms to 368.214 ms (**1.036x**), a
0.0256 ms saving per solve. Across 800 iterations × two solves × 238 attempts, that projects to roughly
**9–10 seconds on the anchor**, with no extra memory or fallback change.

### [x] 3.18 Lower-allocation binary sign extraction (S0)

**Where.** ADMM SVID projection and the factorized-tuning STE formed signs with
`torch.where(value >= 0, ones_like(value), -ones_like(value))`. That creates two full-size branch tensors
in addition to the comparison and result. The comparison already contains the complete binary decision.

**Done (2026-07-13).** Both hot paths now convert the comparison to the source dtype and transform its
0/1 result to -1/+1 in place. Tests cover NaN, infinities, signed zero, both FP32 and BF16 factorization,
and the STE identity gradient. On an otherwise idle RTX 4000 Ada, 512 attention-shaped 1152×448 BF16
signs improved from a median **25.645 ms to 21.199 ms (1.21x)**; 64 largest-MLP 6912×1056 signs improved
from **11.908 ms to 6.214 ms (1.92x)**. ADMM calls this twice before its loop, twice in each of 800
iterations, and twice during export. Applying the measured deltas across the anchor's 238 mixed-shape
attempts projects to roughly **3–12 seconds** saved in factorization. Factorized tuning invokes the same
STE twice per forward, so it should benefit as well; its run-level effect still needs the next full run.

### Parity corrections discovered by performance profiling (not S0)

The global-KD profile exposed two rewrite/legacy differences, so these are correctness fixes rather than
behavior-preserving optimizations. Legacy `NanoQuantLinear` fixes factor, scale, and salient training
parameters to BF16, and its Optimi `AdamW` call leaves weight decay at the zero default. The rewrite had
thawed 1,989,504 selected quantized-layer values as FP32 and explicitly used 0.01 weight decay. It now
thaws those values as BF16, obtains Optimi-compatible Kahan state for them, and uses zero weight decay.

On one pinned Gemma KD batch, with identical cached targets and gradient checkpointing, restoring BF16
reduced selected-state peak CUDA allocation from **4,075,673,088 to 2,696,238,080 bytes (33.8%)**. Five
clean post-warmup steps improved median forward-plus-backward-plus-optimizer wall from **0.705 s to
0.552 s (1.28x)**. This deliberately changes the rewrite's KD recurrence to match legacy, so exact quality
must be remeasured rather than inferred from the speedup.

## 4. Smaller observations (bundle opportunistically)

- [x] `next(iter(model.parameters()), None)` previously executed per batch step in `tune`; the foreach
  optimizer change now resolves the device once per tuning call.
- `FactorizationAttemptStage` computes `latent_prediction=left_latent @ right_latent` per attempt for
  metrics ([quantization_stages.py:212](../src/nanoquant/application/quantization_stages.py)) — necessary,
  but a candidate to fold into the attempt-level metrics pass if the profiler shows it (~0.05% each).
- **Measured, not implemented (2026-07-12):** `reconstruction_metrics` rebuilds the same input/output
  importance grid for each weighted error and recomputes the export weighted error when no separate
  unwhitened prediction is supplied. Reusing one grid and the identical export scalar preserved every
  reported value exactly and improved 16 largest-shape (6912×1152) calls from a median 88.736 ms to
  67.172 ms (**1.32x**), but that is only **1.35 ms per attempt**, or roughly 0.3 s across 238 attempts.
  The extra private metric path and validation-order risk are not justified by that run-level bound.
- **Measured, not implemented (2026-07-12):** `ScaleFitStage` and `fit_scales` both materialize the
  original reconstruction, while `fit_scales` also rescans the selected best error that it already holds.
  At the largest 6912×1056×1152 Gemma MLP shape, the duplicate reconstruction measured 1.491 ms and the
  duplicate weighted-error pass 2.158 ms. Even charging both to all 238 attempts gives an upper bound of
  only **0.87 s per anchor**, and real attention-layer shapes are smaller. Adding an original-prediction
  field to the internal result contract is not justified by that bound.
- **Measured, rejected (2026-07-12):** SVID sign extraction and power iteration only read their shared
  input, but overlapping the sign kernel on a reusable CUDA stream caused resource contention. For a
  1152×448 attention factor, 128-call median time regressed from 108.936 ms to 116.592 ms (**0.93x**);
  for the largest 6912×1056 MLP factor, 32-call median time regressed from 27.724 ms to 31.760 ms
  (**0.87x**). Both produced bitwise-identical projections, but the side-stream/event design was not
  implemented because it degraded both relevant shapes.
- **Legacy right-factor stride rejected for S0 (2026-07-13):** a transposed-stride `right_latent` canary
  reproduced the full-batch control's final reconstruction and frozen tensor identities, including the
  exact **0.3734416366** best/final tuning loss. Its 1335.03 s layer interval is not usable because it
  overlapped KD and crossed the 03:58:38 `nvlddmkm` reset. A subsequent exclusive BF16 microbenchmark on
  the actual 8×2048×1152→1056 shape found no repeatable speedup across three reversed grouped rounds; the
  first stable forward group regressed from **0.605 ms to 0.974 ms**. More importantly, forward values and
  input gradients were exact but trainable-weight gradients differed in **695,904 elements** (maximum
  absolute delta 4). Changing the layout is therefore outside this behavior-preserving pass and is not
  implemented; any future parity-motivated layout change needs quality validation as a numerical change.
- **Resolved after a clean GPU measurement (2026-07-13):** the lower-allocation `_sign` candidate is now
  implemented as item 3.18. Its uncontaminated speedups were 1.21x on the representative attention shape
  and 1.92x on the largest MLP shape, with exact edge semantics and recurrence coverage.
- **Replacement-KD slowdown was environmental, not reproduced (2026-07-13):** the completed replacement
  run's seven epoch intervals were **608, 602, 605, 589, 589, 627, and 624 seconds**, versus a 174-second
  median after the first checkpoint in the prior run. It finished all 2,048 steps in **4,923.63 seconds**
  with 4,079,121,920 peak allocated CUDA bytes. On the now-idle GPU, however, the exact same model path and
  first cached batch measured **0.70–0.85 seconds per complete step**, matching the prior checkpoint rate;
  its parity optimizer occupied only about 5 ms. The earlier 29–34% GPU and 21–28% memory-controller
  snapshots therefore reflect external/system contention, not an optimizer regression. No speculative
  code change was made for the non-reproducible slowdown.
- **Replacement-KD quality rejected (2026-07-13):** despite lowering cached-target training loss from
  2.15325 to 2.13831, the exact serial WikiText-2 result regressed from **444.7151 to 459.7149 perplexity
  (3.37%)**, and is **6.19% worse** than the 432.9306 pre-KD result. The artifact remains as evidence but
  is not accepted as a parity improvement. Profiling then found the BF16 and zero-weight-decay mismatches
  described above; the corrected run below shows those mismatches did not explain the quality gap.
- **Legacy-matched BF16/zero-decay KD also rejected (2026-07-13):** artifact
  `sha256-078aeb721c8257347297eb9d5d477da8899f5530addad4dcb7e9d7479b32774a` completed all 2,048
  steps with final cached-target loss **2.14116740**, 2,702,332,928 peak allocated CUDA bytes, and
  4,114,694,144 peak host bytes. The exact serial 64×128 WikiText-2 evaluation produced **461.5446
  perplexity**: 6.61% worse than the 432.9306 immutable pre-KD result, 0.40% worse than the 459.7149
  FP32/weight-decay replacement, and 19.90% worse than the retained 384.954 legacy result. The lower-memory,
  faster recurrence is retained because it matches legacy behavior, but KD is rejected as a quality
  improvement; the remaining parity error is upstream or elsewhere in the distillation protocol.
- **Thermally throttled block timing rejected (2026-07-13):** a full-protocol, batch-8 block-0 canary
  reproduced every retained layer loss and the final 1.37940836 loss exactly, but took **654.99 s** versus
  **546.75 s** for the matching prior artifact (19.8% slower). During the run NVIDIA reported software
  thermal slowdown at 86 C, with the SM clock falling as low as 765 MHz. Peak allocated CUDA memory was
  6,194,949,632 versus 6,120,173,056 bytes. The exact numerical replay accepts the code path, but the wall
  result is excluded from performance comparisons until it is repeated without thermal throttling.
- **Corrected-KD timing remains invalid after driver resets (2026-07-13):** the legacy-matched BF16,
  zero-weight-decay recurrence durably reached epoch 4 with losses **2.39330782, 2.25053802, 2.20468449,
  and 2.18120250**. Epoch 4 overlapped another process's CPU/model preparation and took approximately
  15 minutes, so it is not a performance sample. Windows then recorded `nvlddmkm` event 153 resets at
  **02:24:03** (during the prior continuous attempt) and **03:13:05** (39 seconds into the next resume),
  while the checkpoint pointer correctly remained at the last durable epoch. The one-epoch checkpoint
  boundary is validated, but neither interrupted interval is evidence for or against a code optimization.
  Quality is now complete and rejected above; end-to-end timing remains unavailable because the run spans
  thermally contaminated, driver-reset, and deliberately cooled processes. The final epoch-8 process took
  165.90 seconds including reload and finalization, which is not a full-run sample. Opt-in initial and between-epoch
  cooldowns now retain the CUDA lease: the initial delay happens before model loading, and later delays
  happen only after activating each non-final checkpoint. They keep the recurrence and protocol identity
  unchanged, avoid repeated model reload/allocation, and prevent another worker from filling the intended
  thermal-rest windows; their sleep time is excluded from performance comparisons.
- **Cross-environment CUDA lease splits fixed (2026-07-13):** two workers first used `%TEMP%` roots `Temp`
  and `Temp\\1`, created independent `cuda:0` leases, and together drove WDDM usage to 11–12 GiB. Moving the
  lease under `%LOCALAPPDATA%` closed that split, but a later diagnostic deliberately redirected
  `%LOCALAPPDATA%` to the repository and again admitted a second CUDA owner. Windows CUDA leases now use a
  session-wide named kernel mutex, independent of `TEMP`, `TMP`, `TMPDIR`, `LOCALAPPDATA`, and filesystem
  ACLs; process death releases it automatically. POSIX retains a stable per-UID `/tmp` root. A true Windows
  CUDA subprocess test proves a second owner is rejected and that termination releases the mutex.
  Non-CUDA fixture tests may explicitly set an absolute `NANOQUANT_DEVICE_LEASE_ROOT`, but CUDA devices
  ignore it. A bounded KD stop also showed the checkpoint exception releasing its lease while Python still
  held roughly 3.5 GiB of CUDA state; exceptional exits now offload the student, synchronize, and empty the
  cache before the lease scope unwinds.
- **Corrected KD exposed a cache-sampling parity bug (2026-07-13):** the completed zero-decay/Kahan run
  reduced cached-target loss from **2.39330782** to **2.14116740**, but exact retained WikiText-2
  evaluation regressed from pre-KD PPL **432.078117** to **462.207656** (legacy checkpoint
  **383.938808**). The rewrite cache began epoch 1 with samples `[172, 55, 225, ...]`; the seeded legacy
  loop begins `[1, 88, 132, ...]`. Legacy carries Python `random.shuffle` state for sample order and the
  training-device Torch RNG for token subsampling across all epochs, whereas the rewrite used two CPU
  Torch generators. Cache planning now replays the legacy Python/device RNG streams, records a sampling
  version in the protocol identity, and allows an explicit replacement run to atomically supersede a
  mismatched cache journal without deleting its immutable artifacts. The corrected run completed all
  2,048 steps with final cached-target loss **2.14039628**, 2.70 GB peak allocated CUDA memory, and exact
  serial PPL **454.431449**. This improves on the incompatible-cache serial PPL **461.544627**, but remains
  worse than immutable pre-KD **432.930572** and the legacy checkpoint through the rewrite backend
  **383.938808**. KD is therefore not the clean end-to-end performance baseline; upstream frozen-state
  parity remains the quality gate. The retained Experiment 018 log strengthens that attribution: legacy
  selected the same 885 parameters and reduced top-k loss **2.3058 -> 2.0443** (delta 0.2615), while the
  exact-sampler rewrite reduced **2.400274 -> 2.140396** (delta 0.259878). The nearly equal optimization
  gain with a persistent ~0.10 starting/ending offset is consistent with KD receiving a worse frozen
  student rather than executing a materially different update rule.
- **Dense reference weights cached (2026-07-13):** `FrozenReferenceLinear` previously reconstructed its
  immutable dense weight from both binary factors on every forward. A single-threaded 256x64x256 CPU
  fixture with an 8x256 input measured **0.1197 ms** median uncached versus **0.0233 ms** cached
  (**5.14x**) with exactly equal output. The cache costs one dense-weight allocation (256 KiB in the
  fixture, proportional to layer weight size), so it is limited to the explicitly dense reference
  backend. `FactorizedReferenceLinear` opts out and continues executing mutable/trainable factors
  directly, including global KD.
- **Legacy hook-chunk replay measured and rejected as a parity improvement (2026-07-13):** replaying
  Experiment 018's ordered 512-token hook reductions made all 182 input-importance vectors bitwise
  equal to the retained statistics, but output error moved from **0.490226% to 0.543882%** layer-mean.
  Independent one-sample repeats had exact inputs and **0.271838%** output disagreement because the
  pinned CCE Triton kernels use lock-protected multi-CTA accumulation. The regenerated allocation was
  unchanged at 38 legacy rank mismatches. Its seven-layer untuned block-0 canary matched only one
  outlier set, retained the `o_proj` rank mismatch, and averaged approximately 50% factor-sign agreement.
  The code-faithful numerical path remains implemented and versioned, but the expensive tuned/full run
  is not being performed because the measured structural gate did not improve.
- **Exact retained-Fisher replay also rejected as a full-run candidate (2026-07-13):** preprocessing can
  now validate and replay all 364 retained Experiment 018 importance vectors exactly, isolating every
  downstream stage from CCE accumulation variance. On block 0, untuned normalized reconstruction was
  equal or better than the historical log for five of seven layers; the two regressions were only 0.07
  and 0.23 percentage points. The historical-batch eight-epoch tune and two-epoch refit nevertheless
  finished at **1.3784899712**, effectively identical to v19 and still 18.59% above the retained legacy
  **1.1624**. It took 476.06 seconds inside the block and peaked at 6,194,081,280 allocated CUDA bytes.
  Current official rank allocation and the rewrite plan match on all 182 layers when given these exact
  vectors (rank sum **105,376**); Experiment 018's logged initial allocation differs in 32 layers (rank
  sum **105,216**), consistent with historical numerical-environment drift across discrete allocation
  thresholds rather than a rewrite planner error. The retained-Fisher block is evidence, not a candidate
  for extension through the remaining 25 blocks.
- **ADMM multi-start rejected (2026-07-13):** an exact-retained-objective block-0 `gate_proj` sweep over
  legacy-reset seeds 0 through 7 produced weighted normalized errors from **0.1971011 to 0.1972472**.
  The best seed improved seed 0 by only 0.0000508 absolute (**0.026% relative**) while each additional
  start costs another complete 800-iteration factorization. Promoting that best reconstruction seed (6)
  through the exact eight-epoch gate tune made the result worse: **0.849965 -> 0.377420**, versus seed 0's
  **0.370981** final and historical **0.31145**. The variation cannot explain the historical gate gap, so
  no multi-start selection logic or further per-seed tuning runs are being added. The parity CLI retains
  an explicit seed option for bounded diagnostics and exact replay; its default remains legacy seed 0.
- **Contemporary legacy gate isolates the historical initialization basin (2026-07-13):** the pinned
  contemporary Experiment 018 launcher recomputed Fisher state from the exact rewrite calibration tensor
  and reached gate `Fact-tune summary` loss **0.37733**, close to the rewrite's exact-retained-Fisher
  **0.37098** and not the historical log's **0.31145**. Its pre-tune quantized block loss was **0.86194**
  versus historical **0.74527**. This is not an online-versus-evaluation reporting mistake: legacy's epoch
  value is the accumulated online statistic, but `Fact-tune summary` calls the hardened full-calibration
  evaluator (contemporary epoch 8 **0.37748**, final **0.37733**; historical **0.31162**, final
  **0.31145**). The gate residual selection is already exact to the retained historical state
  (`[367, 768]`, with bitwise-equal BF16 salient values), and current legacy/rewrite implement the same
  residual-score formula. Contemporary legacy's LS scale fit improved its own weighted objective
  **51.161 -> 50.999 (0.32%)**, comparable to the historical relative improvement
  **39.557 -> 39.464 (0.24%)**; the absolute objectives use differently scaled realized Fisher vectors
  and are not directly comparable, while normalized post-fit errors are **0.1972** and **0.1984**.
  Across the complete contemporary compression phase's 195 fits (including retries), the mean relative
  LS improvement is **0.2155%** and the aggregate-objective improvement is **0.2212%**. The complete
  historical log's 197 fits report **0.2079%** and **0.2126%**, respectively. This broader
  distribution agrees just as closely as the first-layer comparison; increasing LS passes remains an
  explicit block-loss sweep, not a proposed correction to the two-pass implementation.
  Therefore neither residual selection, LS-fit equations, nor loss reporting explains the old advantage.
  The retained historical gate initialization differs from the rewrite by only 0.06--0.16% of factor
  signs and roughly 0.14--0.20% relative L2 in fitted scales. The shared-evaluator replay and extended
  training run below test those behavior-changing parity hypotheses rather than folding them into this
  behavior-preserving perf pass.
  The same contemporary run finished the complete first block at **1.3728** after joint refit, versus the
  rewrite's **1.37848997** with the same pinned tokens/downstream profile and independently realized
  current-versus-retained Fisher state (**0.41% higher**), and historical
  Experiment 018's **1.1624**. This confirms that most of the apparent 18.59% rewrite quality gap was a
  historical numerical-realization gap. The seven contemporary-legacy versus rewrite post-layer losses
  differ by only **-0.83% to +1.70%** and straddle zero rather than showing a systematic evaluator bias.
  All 26 complete post-refit block boundaries provide the stronger accumulating-state check: rewrite
  versus contemporary legacy differs by only **-2.20% to +2.01%** at every boundary and ends at
  **1623.59** versus **1619.00**. Both can differ dramatically from the historical trajectory at the
  same point (for example, block 8 is
  **250.52** rewrite and **256.14** contemporary legacy, versus **115.45** historical), so the historical
  activation-loss path is not evidence of a rewrite-specific regression. The contemporary checkpoint
  contains all 182 layers at exactly the rewrite ranks (rank sum **105,856** for both), so binary BPW and
  count-based outlier cost match exactly. Residual-selected indices differ in 123 layers and same-rank
  signs agree at the expected independent-basin rate of roughly 50%, consistent with the measured CCE
  nondeterminism rather than a structural allocation mismatch. Its eight model-KD losses
  `[2.3977, 2.2443, 2.2136, 2.1837, 2.1664, 2.1524, 2.1462, 2.1430]` nearly overlay the rewrite's
  `[2.40027, 2.24464, 2.20950, 2.18233, 2.16111, 2.14793, 2.14395, 2.14040]`.
  Exact serial WikiText evaluation reports **444.3328** PPL for contemporary legacy versus
  **454.4314** for the rewrite's legacy-sampled tuned artifact (**2.27%** higher) and **432.9306** for
  immutable rewrite pre-KD (**2.57%** lower). The rank/BPW, block trajectory, optimizer trajectory,
  resume/replay, and end-quality comparisons therefore close M4 parity with the numerical-realization
  spread explicitly approved; the historical 383.94 PPL checkpoint remains a retained historical
  quality reference, not a reproducible current-run oracle.
  Contemporary legacy took **424.87 s** for block 0 versus the
  rewrite's **476.06 s**, leaving a measured **12.0%** rewrite wall-time gap on this block; this timing is
  actionable performance evidence, unlike the historical quality delta. The complete contemporary
  compression phase finished in **3:01:25 (10,885 s)**, versus the retained rewrite report's
  **15,284 s** total, making the observed full-run gap roughly **40%**. Exact phase boundaries and the
  newly landed hot-path changes must be remeasured in a fresh rewrite run before assigning that entire
  difference to current code, but the result raises rather than closes the post-parity performance gate.
- **Extra LS passes rejected; longer tuning promoted (2026-07-13):** the exact block-0 gate replay swept
  0/1/2/4/8 alternating LS passes. Full block losses were respectively **0.976456**, **0.818065**,
  **0.818384**, **0.818384**, and **0.818384**. Four and eight passes are a numerical plateau, while the
  one-pass result is only 0.039% below the legacy-compatible two-pass result and does not justify changing
  the parity default. A source audit also found that legacy performs its final rollback comparison after
  casting the fitted reconstruction to the export dtype. The rewrite now does the same: its reported
  before/after objective is measured on the actual exported tensors, and a rare BF16 rounding regression
  rolls back the fitted scales instead of failing stage validation. The resident algorithm and scale-fit
  stage versions were advanced so older commits cannot be adopted under the corrected boundary. This does
  not alter successful fits in the active isolated run, whose persisted stage metrics already validate
  every accepted export against its pre-fit reconstruction. In contrast, a 32-epoch tuning horizon changes
  the cosine schedule materially: gate
  loss reaches **0.303956** by epoch 8 and **0.221278** by epoch 32, versus **0.371515** for the separate
  eight-epoch schedule and historical **0.31145**. Promoting 32 epochs through the complete first block
  lowers post-refit loss from **1.378489** to **0.826978** (40.0%) and beats historical **1.1624** by 28.9%.
  Block 1 likewise improves from the retained full-run boundary **3.583139** to **2.658979** (25.8%).
  Block-0 wall time rises from **476.06 s** to **1201.49 s** (2.52x), but peak allocation is effectively
  unchanged (**6,195,498,496** versus **6,194,081,280** bytes). The 26-block promotion is therefore a
  quality candidate, not a performance optimization, and remains an active detached run.
- **Batched inference graph retention fixed (2026-07-13):** the first gate replay accumulated every
  source-block autograd graph through `copy_` into its host result, reaching 10.85 GiB allocated before an
  eager-attention allocation failed. `_run_block_batched` is now intrinsically no-grad and has a regression
  test; the completed replay peaks at 4.05 GiB during tuning. This was a diagnostic-path lifetime bug, not
  evidence that the 1B model itself requires the full 12 GiB card.
- `JsonlEventSink._read_last_sequence` parses the whole event log at construction — only matters for
  resumed runs with large logs; fine today, worth a tail-scan if event volume grows.
- **Measured, not implemented (2026-07-13):** a fresh process inventories the pinned Gemma snapshot in a
  median **1.759 s** with source-shard verification versus **0.038 s** without it. A cross-process
  signature cache would save about 1.72 s per bounded resume (roughly 14 s across eight epoch processes),
  but less than 0.1% of an uninterrupted full run and at the cost of broadening the snapshot-integrity
  trust boundary. The existing verification remains enabled.
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
- removing the rewrite's full-dataset tuning evaluations or best-epoch restore. For `E` epochs the rewrite
  intentionally performs `E` training passes plus an initial evaluation, `E` epoch evaluations, and a final
  restored-state evaluation (`2E + 2` dataset passes; 18 at the parity setting of 8 epochs). Contemporary
  legacy non-factorized and factorized tuning use the accumulated pre-update training loss and do not restore
  the best epoch, so they normally execute only `E` dataset passes. That structural difference is a credible
  part of the measured wall-time gap, but deleting the extra passes would change the selected weights and
  tuning metrics. The micro profile must measure these phases separately; any semantic alignment belongs in
  an explicitly behavior-changing parity decision, not this optimization list;
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
4. **When tuning/KD phases are enabled in anger:** item 3 is complete; item 11 was measured and rejected
   because its run-level upper bound is negligible.

Items 1–4 of this sequence are independent of parity sign-off (all preserve behavior by construction);
only their *measurement* waits for the Docs/15 P1 baseline if we want clean before/after evidence.
