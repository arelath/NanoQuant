# Legacy quantization performance and VRAM investigation

Date: 2026-07-14  
Rewrite revision inspected: `9af731f66b4a0b460c28f8a32b66254a75aa73cc`  
Legacy comparison: `D:\dev\research\NanoQuant-OfficalCode`, including the retained contemporary Experiment 018 run

## Executive summary

The rewrite has had a real performance regression, but the current v28 parity configuration is not uniformly
slower than contemporary legacy. The most recent clean, protocol-matched block-0 result is 380.05 seconds for the
rewrite versus 424.87 seconds for legacy, so the current rewrite was 10.5% faster on that boundary. Older rewrite
runs were substantially slower: the retained v20 block-0 result was 578.73 seconds, or 36.2% slower than legacy.
The apparent contradiction is explained mainly by execution configuration and by optimizations that landed between
v20 and v28.

The easiest way to recreate the severe slowdown today is to use the parity launcher's default tuning microbatch of
1 while comparing it with Experiment 018's batch/microbatch of 8. That makes every logical batch execute as eight
small forward/backward launches. The clean v28 run explicitly overrides the microbatch to 8. Cooldown sleeps and a
5,081.84-second machine-suspension gap also contaminate several retained wall-clock profiles and must not be counted
as quantization work.

There are nevertheless important remaining sources of rewrite overhead:

1. The rewrite performs one extra full-dataset evaluation at the start of every non-factorized and factorized
   tuning call. Under the Gemma schedule this is 14 extra dataset passes per block, or 364 extra passes over 26
   blocks, compared with legacy tuning.
2. The rewrite synchronizes the full CUDA stream before every optimizer step. This is approximately 67,000 explicit
   stream synchronizations over the 26-block parity schedule. It was introduced after a real numerical handoff
   failure, so it cannot simply be removed without a correctness experiment.
3. Every factorized epoch makes a synchronous resumability snapshot of parameters, best-state tensors, and optimizer
   state and writes a durable checkpoint. Legacy has no equivalent per-epoch D2H/disk path.
4. Rewrite activations live in pageable CPU memory and are staged to CUDA on every pass. Legacy Experiment 018 keeps
   block activations on CUDA. The rewrite policy fixes Windows shared-memory pressure and bounds memory, but it is a
   deliberate transfer-for-memory tradeoff.
5. Legacy's optimizer can select a fused Triton kernel. The rewrite intentionally uses an auditable foreach
   implementation, which launches several elementwise kernels and keeps an additional denominator tensor.
6. The rewrite durably serializes and reloads each block's two approximately 1.12 GiB activation generations.
   This costs roughly 6--10 seconds per block in the v28 profile; it matters, but it is not the dominant gap.

The VRAM regression is real. Contemporary legacy recorded 2.057 GB peak CUDA allocation and 2.103 GB peak CUDA
reservation. The v28 block-0 window recorded 4.751 GB peak allocated and 6.237 GB peak reserved: 2.31x and 2.97x the
legacy measurements, respectively. The peak is reached during post-block refit, not ADMM. After the transient dies,
PyTorch retains most of the 6.237 GB allocator pool: block 1 starts with only 0.781 GB allocated while 6.237 GB is
still reserved. This is primarily a high-water/allocator-pool problem rather than 6.2 GB of live tensors.

The older WDDM shared-memory blowout is a different issue and is fixed in v28. Shared GPU memory now peaks at
0.580 GiB and returns to 80 MiB at each block boundary. Dedicated CUDA reservation remains high, so Task Manager or
`nvidia-smi` can still show materially more GPU use than legacy even though the former 10 GiB shared-memory problem
is gone.

## Scope and method

This was a read-only source/evidence investigation except for this report. No CUDA job was launched. A v28 resident
run was already active under device lease PID `110824`, at block 12 when inspected, so new GPU benchmarks would have
been unsafe and non-comparable.

The comparison covered:

- legacy block orchestration in `src/nanoquant/core/compress_model.py`;
- legacy tuning/factorization in `src/nanoquant/core/compress_block.py` and `core/admm_nq.py`;
- legacy `NanoQuantLinear` and Optimi AdamW implementation;
- rewrite resident orchestration in [resident_quantization.py](../src/nanoquant/resident_quantization.py);
- rewrite tuning, optimizer, factorization, batching, persistence, and frozen-layer execution;
- the contemporary legacy result under
  `evidence/m4/gemma-legacy018-contemporary-pinned-v3`;
- v20/v21/v26/v27/v28 real-model evidence and the v28 macro profile.

The comparison uses contemporary Experiment 018 rather than the historical checkpoint as the performance anchor,
because it was rerun on the pinned model, pinned calibration tensor, current CUDA environment, and current legacy
source. The complete legacy run includes model-level KD, while the v28 four-block canary does not, so full-run wall
times are not compared directly.

## Measured evidence

### Wall time

| Evidence | Work measured | Wall time | Interpretation |
|---|---|---:|---|
| Contemporary legacy Experiment 018 | compression plus 8-epoch model KD | 14,267.90 s | Complete reference; not directly comparable to the unfinished rewrite run |
| Contemporary legacy block 0 | full block schedule | 424.87 s | Best retained apples-to-apples block anchor |
| Rewrite v20 block 0 | full block schedule | 578.73 s | 36.2% slower than legacy |
| Rewrite v26 block 0 | full block schedule | 399.22 s | 6.0% faster than legacy, but later trajectory failed |
| Rewrite v28 block 0 | full block schedule, microbatch 8 | 380.05 s | 10.5% faster than legacy |
| Rewrite v28 blocks 0--3 | active block work | 1,548.79 s | 387.20 s/block average; excludes suspension |
| Rewrite v28 block 2 recorded interval | block work plus host suspension | 5,477.21 s | Not a valid performance sample |

The v28 profile records 6,664.14 seconds total, but 5,081.84 seconds are one no-event gap during a factorized-tuning
span. Subtracting that gap yields the retained 1,548.79-second active block total. Treating the raw profile total as
compute would overstate rewrite cost by more than 4x.

For the active v28 work, the phase distribution is approximately:

| Phase | Active seconds | Share of active four-block time |
|---|---:|---:|
| Factorized tuning, suspension removed | 765.75 | 49.4% |
| Non-factorized tuning | 409.21 | 26.4% |
| ADMM execution | 181.02 | 11.7% |
| Post-block refit | 66.40 | 4.3% |
| Per-layer loss snapshots | 44.25 | 2.9% |
| Block commits | 27.15 | 1.8% |

Tuning is therefore the optimization target. ADMM is no longer the dominant cost on the tuned workload, and
artifact commit overhead cannot explain a large multiple-of-legacy slowdown by itself.

### Device and host memory

All byte ratios below compare PyTorch CUDA counters with the same semantics. WDDM values are shown separately.

| Meter | Contemporary legacy | Rewrite v28 | Rewrite / legacy |
|---|---:|---:|---:|
| Peak CUDA allocated | 2,057,433,600 B (1.916 GiB) | 4,751,435,264 B (4.425 GiB) | 2.31x |
| Peak CUDA reserved | 2,103,443,456 B (1.959 GiB) | 6,236,930,048 B (5.809 GiB) | 2.97x |
| Peak WDDM dedicated | not retained | 6,520,320,000 B (6.072 GiB) | n/a |
| Peak WDDM shared | not retained | 622,854,144 B (0.580 GiB) | n/a |
| Peak host working set | not retained | 11,439,714,304 B (10.654 GiB) | n/a |

At the start of v28 block 1, only 780,508,672 bytes were allocated but all 6,236,930,048 bytes remained reserved.
The approximately 5.46 GB gap is cached allocator capacity/fragmentation, not live model state. The process-level
driver footprint was correspondingly around 7.6 GiB because it also includes CUDA context and non-allocator use.

The active resumed run shows the same pattern at block 12: about 1.51 GB live allocation and about 6.96 GB reserved
during ordinary factorized tuning, with the board reporting about 8.35 GB used. It had already accumulated more than
16 million allocator allocations/frees, so churn and high-water retention are material.

## Performance findings

### P0: the parity launcher has a non-parity microbatch default

Legacy Experiment 018 uses `fact_batch_size=8`, `nonfact_batch_size=8`, and block-forward batch 8. The current
launcher defaults logical tuning batch size to 8 but independently defaults `--tuning-microbatch-size` to 1 in
[run_gemma_parity.py](../tools/run_gemma_parity.py). With the default, every logical optimizer batch is split into
eight forward/backward microbatches. The same number of tokens is processed, but GEMMs are smaller and Python,
kernel-launch, autograd, and staging overhead are paid eight times.

The clean v26/v27/v28 manifests all explicitly use microbatch 8. Any timing from an invocation that omitted this
override is not protocol-matched to contemporary legacy. This is the first configuration to check when a current
run appears several times slower.

Recommended action: make `None`/inherit-the-logical-batch the parity default, and require microbatch 1 to be an
explicit memory-fallback choice. Record logical batch and microbatch next to every performance comparison.

### P0: rewrite tuning still executes extra full-dataset passes

Legacy `tune_nonfact` and `tune_fact` execute one training pass per epoch and use the training loss already computed
for backward. The rewrite's `legacy_training` mode removed per-epoch and final reevaluations, but `tune()` still calls
`_evaluate_loss()` once before every tuning phase to populate `TuningMetrics.before`.

For one block:

- non-factorized legacy passes: `8+4+3+2+2+2+2 = 23`;
- factorized legacy passes: `7 layers * 8 = 56`;
- rewrite-only initial evaluations: `7 + 7 = 14`.

That is 93 rewrite passes versus 79 legacy passes before post-block refit, 17.7% more full-dataset traversal. Across
26 blocks it is 364 extra block evaluations. The relative penalty is especially large for the four layers with
only two non-factorized epochs: an extra initial pass adds 50% to their dataset-pass count.

Recommended action: accept a caller-supplied pre-tuning loss or make `before` optional in legacy mode. The resident
caller already computes nearby block-loss snapshots, so this can likely avoid work without changing the training
recurrence.

### P0: explicit gradient-handoff synchronization serializes every optimizer step

[tuning.py](../src/nanoquant/application/tuning.py) calls
`torch.cuda.current_stream(device).synchronize()` immediately before every optimizer step. Legacy calls
`backward()`, `optimizer.step()`, and `zero_grad()` without a full host synchronization.

At 256 samples and logical batch 8 there are 32 optimizer steps per epoch. The complete schedule produces roughly:

- 46,592 factorized-tuning synchronizations (`182 * 8 * 32`);
- 19,136 non-factorized synchronizations (`26 * 23 * 32`);
- 1,664 refit synchronizations (`26 * 2 * 32`);
- about 67,392 total.

This stalls the host, prevents useful queueing/prefetch across the boundary, and makes small-batch launch overhead
more visible. However, the barrier fixed a real run where asynchronously handed-off gradients did not follow the
legacy objective trajectory. It is a correctness boundary, not dead code.

Completed (2026-07-19): a bounded PyTorch CUDA trace on the pinned block-0 gate showed that every compute,
backward, loss, and foreach-optimizer kernel used the same CUDA stream; only the already-event-fenced pinned H2D
copies used the copy stream. The explicit host barrier therefore added no device dependency. Removing it preserved
all 16 epoch losses and final tensor-derived metrics exactly in the eight-epoch legacy/rewrite replay while reducing
mean tuning wall time from 19.328 seconds to 18.074 seconds (6.5%). Resident algorithm v38 records the execution-path
change. The full resident parity run remains the model-level quality/performance gate.

### P1: per-epoch durable tuning checkpoints have no legacy equivalent

Every factorized epoch calls `capture_optimizer_state()`, copies current parameters, best parameters, exponential
averages, squared averages, and Kahan compensation to CPU, then writes a new durable checkpoint generation. At the
parity schedule this is 1,456 epoch checkpoints (`182 * 8`). The D2H copies synchronize materialization, and the
filesystem work is inside the tuning phase. Legacy only retains the live optimizer state.

This is valuable resumability, but it changes the cost model. It should be reported separately from numerical
tuning time.

Recommended action: micro-profile `checkpoint_snapshot` on one clean block; evaluate a configurable checkpoint
interval or an asynchronous snapshot/write pipeline whose completion is fenced before the next durable boundary.
Any relaxation must retain the currently promised resume granularity or document the change.

### P1: best-state tensors are maintained even when best-state restoration is disabled

The v28 parity manifest sets `restore_best_tuning_state=false`, matching legacy non-factorized/factorized tuning.
Nevertheless, `tune()` clones every selected parameter into `best_state` before training and replaces that clone on
each improving epoch. Factorized checkpoint snapshots also copy this otherwise-unused best state to CPU.

Legacy non-factorized/factorized tuning has no best-state clone. Legacy post-block refit does keep one, so the issue
is specific to modes that do not restore the best state.

Recommended action: do not allocate or update `best_state` when restoration is false. Adjust the resume schema so
legacy-training checkpoints do not duplicate a state that will never be restored, while retaining exact resume of
the current parameter/optimizer state.

### P1: optimizer fusion and state footprint differ

Legacy Optimi AdamW auto-selects Triton when supported and otherwise foreach. Its Triton path fuses the elementwise
Adam/Kahan recurrence into a small number of kernels. The rewrite deliberately avoids the optional kernel and uses
multiple `torch._foreach_*` operations. This was a major improvement over the rewrite's old scalar loop (the retained
microbenchmark is 2.28x faster), but it is not the same launch structure as legacy Triton.

The rewrite also stores a full-size `denominator` tensor for each selected parameter. Legacy foreach reuses the
gradient buffer for the denominator, and legacy Triton computes the denominator in registers. Together with the
unused best-state clone, the rewrite's non-factorized/factorized selected-parameter footprint is approximately:

- legacy: parameter + gradient + three Adam/Kahan states = about `5P`;
- rewrite: the same + denominator + best-state clone = about `7P`.

This is a 40% increase in the selected-parameter portion of memory, although activations dominate the overall peak.

Recommended action: first eliminate the unused best state. Then benchmark a bitwise-compatible fused optimizer or
safe gradient-buffer reuse. Keep the current foreach implementation as the portable reference path.

### P1: CPU activation streaming is a deliberate speed/memory tradeoff

Legacy Experiment 018 sets `block_activation_device="cuda"`, so its tuning inputs and teacher targets can be
indexed on device across epochs. Rewrite v28 keeps the complete approximately 1.12 GiB streams pageable on CPU and
uses two pinned host slots plus two fixed device slots to overlap each H2D transfer with compute.

The rewrite design prevents the WDDM shared-memory blowout and makes activation storage independent of total VRAM,
but it retransfers both datasets on every training/evaluation pass. This is likely most visible on attention layers
where the factorized compute is small relative to moving a 2,048-token activation batch.

Recommended action: retain bounded pageable storage as the default, but consider an opt-in activation GPU cache
when `mem_get_info` proves that one or both full streams fit behind a declared reserve. The legacy code already has
an `off/inputs/both/auto` policy that can serve as the behavioral reference.

### P2: durability and observability add bounded overhead

The rewrite commits each layer, commits two block-boundary activation streams, hashes descriptors/members, appends
the journal, retires the predecessor generation, and reloads the just-committed boundary so continued and resumed
runs consume identical bytes. Legacy mutates an in-memory model and copies only the next boundary.

In the v28 four-block profile, block commits take 27.15 seconds total, about 1.8% of active time. Unattributed
pipeline time includes the subsequent activation reload. This is measurable but too small to be the main slowdown.

Recommended action: retain the durability semantics. Profile serialize/hash/write/read separately before attempting
overlap; previous overlap experiments improved only the safe window and were rejected because of durability/lifetime
complexity.

### P2: eager loss algebra may allocate more than the scripted legacy helper

Legacy wraps its weighted MSE expression in `@torch.jit.script`; the rewrite executes the equivalent FP32 casts,
subtraction, square, importance multiply, and reduction eagerly. Depending on the active PyTorch fuser, legacy may
use fewer launches and temporary FP32 activation tensors. This is a medium-confidence static finding, not yet
isolated by retained CUDA traces.

Recommended action: use the micro profiler/allocator history on `_loss_sum`, and only introduce a scripted or fused
path if exact output and gradient parity are proven on Gemma-sized BF16 tensors.

## VRAM findings

### The current peak is post-block refit

In v28 block 0, peak allocated memory remains 3.287 GB through the final factorized layer, then rises to 4.751 GB
immediately after `post_block_refit.started`. Live allocation observed after the step is only about 0.864 GB. This
identifies transient refit forward/backward state as the allocated high-water.

Refit trains scales/outliers/bias across all seven factorized layers simultaneously, so autograd must retain a full
block's factorized intermediates. The rewrite also rehydrates every frozen layer as a `TrainableFactorizedLinear`.
Those layers do not mark their binary factors immutable, even though refit never selects the factor parameters; the
forward still applies the sign STE. That creates avoidable sign operations and temporaries, although the exact share
of the 4.751 GB peak needs allocator tracing.

Recommended action: set the immutable-factor fast path for post-block refit and prove bitwise forward/gradient
parity. Capture a refit-only memory profile before and after. If peak remains excessive, tune refit with an explicit
microbatch smaller than its logical optimizer batch; this changes execution shape and must be separately qualified.

### Reserved memory, not live tensors, explains most steady visible VRAM

After refit, the caching allocator retains its largest blocks. The next block begins with 0.781 GB allocated and
6.237 GB reserved. `empty_cache()` is currently pressure-gated at 80% of device capacity; 6.237 GB on a 12 GB card
does not cross the gate, so the pool remains visible to the driver and unavailable to other CUDA processes.

This is not a leak in the strict sense: allocation falls after each phase. It is still operationally expensive
because it leaves less headroom and makes desktop/device monitors report high usage.

Recommended action: add a block-boundary policy that can release the CUDA cache after the refit high-water, based on
`reserved - allocated` as well as `reserved / total`. This will reduce steady visible VRAM, not the true peak, and
must be benchmarked against allocator re-warm cost.

### Completed frozen blocks remain on CUDA

Legacy explicitly executes `q_blocks[i] = q_block.cpu()` at the end of every block. The rewrite installs
`working_block` back into the model's decoder container without moving it to CPU. On resume it also restores every
completed frozen block to the request device. This makes live allocation grow with the number of completed blocks:
the v28 same-process block-start samples rise from approximately 0.647 GB at block 0 to 0.887 GB at block 3, and the
active resumed block-12 process sits near 1.51 GB during ordinary tuning.

The growth is bounded for Gemma 1B but conflicts with the desired active-block/workspace memory model and will be
more important on larger models. Frozen factors are stored as BF16 reference tensors, not packed one-bit runtime
weights, so their CUDA footprint is much larger than logical BPW suggests.

Recommended action: offload a completed frozen block to CPU after its activation boundary is committed. Restore or
stream completed blocks only for final inline quality/assembly. Alternatively, keep a packed resident form once the
runtime format exists. The block loop itself needs only the committed boundary and current working block.

### Full-model load is a transient peak source, but no longer the main one

The rewrite initially loads the complete model on CUDA, captures prefix/quality state, and then replaces all
uncompleted decoder blocks with `Identity`. This reduced live model-shell allocation from about 2.145 GB to
0.625 GB in the retained diagnostic. Because run-level peak counters reset before model load, reports that are not
using block windows may still include the full-load transient. The v28 block peak is higher and comes from refit, so
further model-load work will not solve the current block-window maximum.

### WDDM shared-memory pressure is fixed in v28

Version 27 pinned complete activation streams. Windows charged PyTorch's retained pinned-host blocks as shared GPU
memory, reaching roughly 10 GiB shared and a 15.153 GiB host working set. Version 28 keeps full streams pageable,
pins only two batch slots, and releases the pinned-host cache after every block.

The real v28 canary peaks at 0.580 GiB WDDM shared, returns to 80 MiB at each block boundary, and reduces peak host
working set to 10.654 GiB. This issue should not be conflated with the still-high 5.809 GiB CUDA reservation.

## What is not causing the main gap

- **Profiling/logging:** the v28 macro recorder reports 0.037 seconds of recorder overhead over 6,664 seconds. The
  JSONL file handle is persistent. Default five-second memory sampling is negligible.
- **ADMM alone:** ADMM is 11.7% of active v28 time. Exact component parity work has already aligned the source math;
  broad tuning changes have much larger leverage on the full schedule.
- **Layer/block artifact commits alone:** block commit is about 1.8% of active time. It is worth optimizing only
  after tuning and memory high-water issues.
- **A monotonic CUDA live-tensor leak:** current allocated bytes fall sharply after refit/tuning. The dominant
  steady gap is reserved allocator capacity, plus a smaller deliberate completed-block residency slope.
- **The old 10 GiB shared-memory bug:** v28 real-model evidence proves that it is corrected.

## Recommended experiment order

Do not benchmark while the current v28 device lease is active. After it finishes:

1. Run a clean block-0 legacy/rewrite comparison after an idle/thermal stabilization interval. Pin batch 8,
   microbatch 8, no cooldown, identical ranks/outliers/ADMM/tuning/refit, and no model KD.
2. Enable rewrite micro profiling, CUDA timing, and memory counters for one block. Capture allocator history only for
   one selected non-factorized layer, one factorized layer, and refit to keep trace size bounded.
3. Add a measurement-only breakdown for `initial_evaluation`, training forward/backward, gradient handoff,
   optimizer, checkpoint snapshot/write, and H2D staging. Account for at least 90% of each tuning call.
4. Test eliminating unused best-state maintenance and supplying the initial metric from the caller. These are the
   lowest-risk high-value rewrite-only costs.
5. Compare foreach against the exact legacy Triton/foreach optimizer on real selected parameter shapes and BF16
   Kahan state. Require bitwise parameter/state equality, not just close loss.
6. Test CPU offload of completed frozen blocks and a post-refit block-boundary CUDA cache release. Report peak
   allocated, peak/current reserved, WDDM dedicated/shared, and wall time separately.
7. Only then investigate narrower synchronization or activation GPU caching, because both have correctness or
   memory-policy tradeoffs.

## Bottom line

The report supports both parts of the original concern, with an important qualification:

- The rewrite **was** substantially slower in retained v20 evidence, and the source still contains several
  rewrite-only tuning costs. But the clean v28 microbatch-8 block result is currently faster than contemporary legacy;
  severe present-day slowdowns should first be checked for microbatch 1, cooldown, suspension, or other protocol
  mismatch.
- The rewrite **does use more CUDA memory** on the comparable real workload. Its block peak allocation is about
  2.31x legacy and reservation about 2.97x. Post-block refit creates the true peak; allocator high-water retention
  explains most steady visible VRAM; completed frozen blocks add a smaller linear residency cost. The previous WDDM
  shared-memory blowout has already been fixed.

The next optimization pass should focus on tuning pass count, checkpoint/best-state work, optimizer/synchronization
boundaries, and refit memory. ADMM and artifact commit work are secondary on the current tuned workload.
