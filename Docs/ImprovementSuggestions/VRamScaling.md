# Scaling Compression Past 12 GiB: 7B Now, 21B Next

**Status:** Proposed

**Audience:** Maintainers and algorithm researchers planning the next model-size step

**Related:** [ADR-0004 block streaming](../adr/0004-block-streaming-for-large-models.md), [04-execution-and-scaling.md](../04-execution-and-scaling.md), [17-vram-diagnostics.md](../17-vram-diagnostics.md), [18-legacy-quantization-performance-and-vram-investigation.md](../18-legacy-quantization-performance-and-vram-investigation.md), [ExcessiveVRamIssue.md](../ExcessiveVRamIssue.md), [13-implementation-task-list.md](../13-implementation-task-list.md) (Milestone 5)

## 1. Summary

The binding constraint on the 12 GiB card is not the algorithm — it is the **resident executor's policy of keeping the whole BF16 model shell on CUDA**. Gemma 3 4B (~8 GiB of BF16 weights) fits with a few GiB to spare; any 7B-class model is ~13–15 GiB of weights before a single workspace tensor is allocated, so it can never fit under this policy no matter how carefully transients are trimmed.

The good news is that the block-sequential algorithm only ever *needs* the active block, and the architecture was explicitly designed for this ([ADR-0004](../adr/0004-block-streaming-for-large-models.md)). The recommendation, in one sentence each:

- **7B on 12 GiB:** implement the `cpu_offload` executor from the [Docs/04 executor table](../04-execution-and-scaling.md) — model weights stay in host RAM, one source block is materialized on CUDA per block step. The extra transfer is ~20–40 ms per block against minutes of per-block compute, i.e. well under 0.1% wall clock. Peak VRAM drops to roughly active blocks + workspace ≈ 4–6 GiB — *lower* than today's 4B runs.
- **Speed:** the freed VRAM funds the opt-in **activation GPU cache** already recommended in [Docs/18 P1](../18-legacy-quantization-performance-and-vram-investigation.md). Source weights are read once per block; activation streams are re-read on every tuning pass (~14 dataset passes per block). Trading shell residency for activation residency should make 7B runs *faster* per block than today's 4B policy, not slower.
- **21B on 12 GiB:** finish wiring the **streaming executor**. Nearly all of its components (block-aligned safetensors source reads, mmap activation store, tier auto-selection, double-buffered propagation, incremental packed shard writes) are implemented and integration-tested under Milestone 5; what is missing is production workflow wiring, prefetch (M5.9), and the equivalence/canary gates (M10.11, M10.12).

## 2. Where the 12 GiB goes today

Evidence from the measured runs ([Docs/18](../18-legacy-quantization-performance-and-vram-investigation.md), [ExcessiveVRamIssue.md](../ExcessiveVRamIssue.md)):

| Component (resident executor) | Scales with | Gemma 3 4B observed |
|---|---|---|
| Full BF16 model shell on CUDA ([resident_quantization.py:1353](../../src/nanoquant/resident_quantization.py:1353)) | total parameters | ~8 GiB |
| Working block + factor workspace + ADMM transients | largest block | ~1–2 GiB |
| Hessian/objective storage | layer width² (dense) or width (diagonal) | ≤ ~0.2 GiB |
| Tuning state (params, best-state, optimizer) | active block | < 1 GiB |
| Two device batch slots for activation staging (v28 policy) | batch × seq × hidden | ~0.1–0.3 GiB |
| Allocator pool retention / fragmentation | history-dependent | ~5.46 GiB reserved-but-unallocated gap observed on 1B |

Residency bugs that used to inflate this (inline quality logits, WDDM pinned-cache growth, completed-block accumulation, factor retention during dense replay) are all fixed as of resident algorithm v30 and the compact dense replay loader. What remains is the by-design shell residency — the one term that scales with *total* model size instead of block size.

The activation streams themselves are already model-size-independent on the device: complete teacher/compressed streams are pageable host memory, with two pinned host slots and two device slots for overlap. That policy is correct and should not change as a default.

## 3. Recommendation for 7B (fits 12 GiB, negligible speed cost)

### R1. [x] Turn on `expandable_segments` now (zero code)

The measured allocator behavior shows up to ~5.5 GiB reserved-but-unallocated after transient peaks. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lets the allocator return that slack and largely removes fragmentation-driven OOM at near-zero speed cost. The manifest already records this variable, so runs stay comparable. This does not make a 7B shell fit — it buys headroom and makes every estimate below more predictable. Verify with the reserved-vs-allocated gap from the [Docs/17](../17-vram-diagnostics.md) sampler.

Implemented as the package-process default while preserving an explicit operator `expandable_segments` choice and
coexisting allocator options. The 4B sampler verification remains open in the rollout checklist below.

### R2. [x] Host-resident source blocks: implement the `cpu_offload` executor (the 7B enabler)

Today `cpu_offload` exists only as a schema enum ([schema.py:27](../../src/nanoquant/config/schema.py:27)), a name the resource planner accepts ([resource_planning.py:93](../../src/nanoquant/infrastructure/resource_planning.py:93)), and a calibration OOM fallback action. Implement it as the natural extension of the v30 partial-residency mechanics that already exist:

- v30 already honors `restore_completed_blocks=False`, leaves the shell slot empty, and releases each completed block after its durable commit. Extend the same policy to **pending** source blocks: keep the shell's parameters on host (pageable RAM), and materialize block `N`'s source weights on CUDA only when block `N` becomes active.
- The per-block loop then holds on device: source block (teacher forward → next teacher activations, already stored host-side), working block, workspace. Release both at the block commit, exactly where v30 releases the completed block today.
- Token embeddings and the LM head never need device residency during block compression: prefix capture already materializes embedding outputs once, and the head is only exercised by quality/distillation paths that are already streamed/batched.

**Speed analysis (why this is the "no heavy impact" option):** a 7B block is ~0.4 GiB of BF16; one H2D copy at ~20 GB/s effective PCIe 4.0 bandwidth is ~20–25 ms, twice per block if the working copy is cloned from the transferred source (it is — same tensors). Blocks take minutes to hours under the parity tuning schedule. The added cost is < 0.1% of wall clock, and it is one transfer per block — unlike activations, which move every pass and are already overlapped.

**Expected peak for a 7B run:** 2 active block copies (~0.9 GiB) + factor workspace and transients (~2 GiB, scaled from 4B measurements) + Hessians (dense 4096² = 64 MiB; a 14336-wide `down_proj` input is ~0.8 GiB fp32, or use `low_rank_diagonal`) + tuning state (< 1 GiB) + staging slots ≈ **4–6 GiB**, leaving several GiB of genuine headroom on the 12 GiB card.

**Host RAM cost:** the shell moves to RAM (~14 GiB pageable) plus two activation streams (scaling by hidden width from the measured 1B streams, roughly ~4 GiB each for a 4096-hidden 7B). With the ~11–12 GiB working sets already measured, plan for a **48 GiB minimum, 64 GiB comfortable** host. The planner's existing host-limit margins and `RES001` refusal already express this.

**Planner change:** teach `auto` the three-step ladder `resident → cpu_offload → streaming` (today it jumps straight from resident to streaming), with `peak_gpu` for `cpu_offload` computed as the streaming GPU formula plus the second block copy. The estimate-versus-measured `budget_utilization` loop from [Docs/17 §5.2](../17-vram-diagnostics.md) is the tool to calibrate these estimates on the first real 7B run.

Implemented in the production resident composition: `cpu_offload` keeps the Hugging Face shell and calibration on
pageable CPU, materializes each active source/working block on the compute device from safetensors, moves captured
block metadata with it, and requires completed-block restoration and inline quality to remain disabled. Model-level
KD is rejected until its teacher forward is streamed. The resource planner now uses the requested three-step ladder
and accounts for the host shell plus a second active-block GPU copy. CPU tiny-model resident/offload execution is
artifact- and metric-equivalent; the requested 4B CUDA memory/equivalence canary remains open below.

### R3. [x] Spend the freed VRAM on the activation GPU cache (net speed win)

[Docs/18 P1](../18-legacy-quantization-performance-and-vram-investigation.md) already recommends an opt-in `off/inputs/both/auto` activation-cache policy (legacy behavioral reference exists). Under R2 the shell no longer occupies the device, so for 7B both full streams (~8 GiB… too big) or at least the *inputs* stream (~4 GiB) can live on CUDA behind a declared reserve, eliminating the per-pass H2D retransfer that Docs/18 identifies as the dominant self-inflicted tuning overhead. `mem_get_info` gating and the resource plan's activation-tier machinery already support the decision. This is the piece that turns R2 from "no slower" into "probably faster per block than the current 4B configuration".

Implemented as `runtime.activations.gpu_cache = off|inputs|both|auto` with a validated
`runtime.activations.gpu_reserve_gib`. The production block loop caches the compressed inputs first and teacher targets
second, logs every admission or rejection, requires explicit policies to fit, lets `auto` fall back to the existing
pageable streaming path, and releases final-boundary aliases before evaluation/assembly. The policy is deliberately
excluded from semantic commit identity. A CUDA wall-clock comparison remains open in step 5 below because experiment
006 currently owns the device.

### R4. [ ] Quality evaluation must not reload a dense 7B model

The separate quality stage currently loads the full reconstructed model through the compact dense replay module — measured 7.54 GiB peak for 4B, so ~13+ GiB for 7B: it will OOM even after R2. Two options, in order of preference:

1. [x] **Evaluate through the packed runtime.** A 1–2 bpw packed 7B is ~1.5–3 GiB and the packed backend already matches the logical backend exactly; the 1B packed model peaked at ~0.8 GiB during benchmarking. This also evaluates the artifact you actually ship. Complete compression workflows now pass their newly exported packed artifact to the quality evaluator by default.
2. [ ] **Block-streamed dense replay:** install reconstructed layers one block at a time, run the quality forward block-sequentially over an activation stream, release. Reuses the same streamed-forward machinery as compression.

### R5. [ ] Keep the v30 guards as defaults for ≥7B recipes

`restore_completed_blocks=False`, inline quality disabled, streamed block loss, compact replay, pinned-cache release per block — all already exist; make the base recipe for larger models pin them explicitly so a future recipe cannot silently regress residency.

## 4. Recommendation for 21B: finish the streaming executor

A 21B-class BF16 checkpoint is ~40 GiB. `cpu_offload` alone would demand ~40 GiB of pageable shell plus ~6 GiB per activation stream in host RAM — feasible only on a 96 GiB host, and pointlessly fragile. This is exactly the regime [ADR-0004](../adr/0004-block-streaming-for-large-models.md) targets: **source stays on disk as sharded safetensors, one block is materialized directly from the shards, activations tier to RAM or mmap.**

Most of the machinery already exists and is tested (Milestone 5): block-aligned source reads without a full `state_dict` (M5.3), the mmap activation store with atomic generation commits (M5.4), automatic tier selection (M5.5), double-buffered propagation (M5.6), one-block teacher generation with release (M5.7), incremental packed shard writes (M5.8), forward-only streamed calibration (M5.10), block-diagonal and low-rank+diagonal objectives (M5.12–13), and `StreamingBlockExecutor` with integration coverage (`test_streaming_blocks_calibration.py`, `test_streaming_activations.py`, `test_mmap_activation_store.py`).

The actual remaining work:

1. [ ] **Wire `StreamingBlockExecutor` into the production compression workflow** so the full block loop (calibrate → factorize → tune → freeze → commit) runs against a `ModelSource` instead of the resident shell. This is the bulk of the effort; the algorithm stages themselves are placement-independent by design, so it is orchestration work, not math work.
2. [ ] **M5.9 source-block prefetch:** overlap the ~45 ms per-block shard read (0.85 GiB at NVMe speeds) with the previous block's compute. Cheap once leases exist; measure before keeping, per the task's own wording.
3. [ ] **Activation tier at 21B:** two streams at ~6 GiB each (hidden ~6144) still fit pageable RAM on a 64 GiB host; the mmap tier is the pressure valve, and the planner's `auto` logic already picks it. Budget NVMe accordingly (source ~40 GiB + activations ~12 GiB + packed output).
4. [ ] **Hessian policy:** dense fp32 at a ~16–20k intermediate width is 1.0–1.6 GiB — allowed under the existing `HES001` workspace reservation one layer at a time, or drop to `low_rank_diagonal` for the widest layers. Both paths exist; the recipe just has to choose per layer width.
5. [ ] **Global distillation:** the top-k teacher cache is already disk-backed, but *producing* a teacher epoch means full-model teacher forwards. Either run the teacher pass block-streamed through the same executor (writing top-k targets to the existing cache), or disable global KD for the first 21B recipe — the [Docs/04 70B plan example](../04-execution-and-scaling.md) already anticipates "Global KD: disabled by recipe".
6. [ ] **Evaluation:** dense replay is out of the question at 21B; the packed runtime (~2.6–5 GiB at 1–2 bpw) is the only realistic quality/benchmark path. Make R4 option 1 the default before starting 21B work.
7. [ ] **Close the gates:** M10.11 resident-versus-streaming equivalence report on a small model (this is also the cheapest way to trust streaming for 7B/21B), then the M10.12 large-model canary with bounded memory, interruption, and resume.

M5.11 (streamed forward/backward calibration for Fisher statistics) can stay open: forward-only calibration is implemented and is what current recipes use.

## 5. What not to do

- **`device_map=auto` offload:** already rejected in ADR-0004 — it moves placement but keeps the two-model architecture, gives no stage-level memory planning, no bounded-memory guarantee, and no atomic progress story.
- **Quantizing or truncating the teacher/source to fit** (fp8 shell, fewer calibration samples, shorter sequences): changes semantic compression identity and invalidates parity/quality evidence. Memory policy must stay a non-semantic execution option, as the config architecture already requires.
- **Shrinking microbatches as the primary lever:** batch sizes only bound transients; the shell is the binding term. Squeezing batches to fit a 7B shell would trade real speed for a constraint R2 removes outright.
- **Hot-loop `empty_cache()`:** M4.3 explicitly avoids allocator clearing in hot loops; R1 addresses pool retention at the allocator-policy level instead.

## 6. Expected memory envelopes

Rough planning numbers (BF16 source, current parity-style recipes; verify with the Docs/17 `budget_utilization` loop on first runs):

| Model | Resident (today) | + R1/R5 | cpu_offload (R2–R4) | Streaming |
|---|---:|---:|---:|---:|
| 4B | ~8–10 GiB peak — fits | ~8 GiB | ~4–5 GiB | ~4–5 GiB |
| 7B | ~15–17 GiB — **OOM** | still OOM | **~4–6 GiB — fits** | ~4–6 GiB |
| 21B | ~42+ GiB — OOM | OOM | ~6–8 GiB GPU, but ~55+ GiB host RAM | **~6–8 GiB GPU, ~20–25 GiB host — fits** |

Wall-clock impact: R2 adds one ~0.4–0.9 GiB H2D per block (< 0.1%); R3 should more than repay it by removing per-pass activation retransfer; streaming adds per-block shard reads that M5.9 prefetch hides under compute.

## 7. Suggested order of work

1. [ ] R1 allocator setting + Docs/17 sampler verification on the existing 4B workload (hours).
2. [ ] R4 packed-runtime quality evaluation path (needed by everything ≥7B, useful for 4B today).
3. [ ] R2 `cpu_offload` executor + planner ladder; validate on Gemma 3 4B first — peak should drop to ~4–5 GiB with unchanged block losses (same math, different placement), giving a cheap equivalence check before any 7B run.
4. [ ] First 7B compression run; calibrate planner estimates against measured window peaks.
5. [ ] R3 activation GPU cache behind `mem_get_info` reserve; measure per-block wall clock against step 4.
6. [ ] Streaming workflow wiring + M10.11 equivalence on 1B/4B, then M5.9 prefetch.
7. [ ] 21B canary (M10.12): bounded memory, interruption, resume, packed export.

Every step lands independently, and steps 1–5 never touch the algorithm — they are placement and policy, which is exactly the boundary the architecture was drawn to protect.
