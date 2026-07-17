# Execution, Checkpoints, Resume, and Scaling

## 1. One pipeline, multiple executors

The algorithm pipeline is independent of placement. Executors implement how tensors and modules reach compute:

| Executor | Intended use | Weight residency | Activation residency |
| --- | --- | --- | --- |
| `resident` | 1B and other models fitting comfortably on one GPU | Whole working model on GPU | GPU by default |
| `cpu_offload` | Medium models with sufficient RAM | Model in RAM, active block on GPU | RAM or GPU |
| `streaming` | 70B-class models or constrained hosts | Source on disk, active block/layer materialized | RAM or memory-mapped disk |
| `distributed` | Multi-GPU acceleration or global training | FSDP/tensor/pipeline sharded | distributed |

The production resident composition supports both `resident` and `cpu_offload`. In `cpu_offload`, the initial model
shell and calibration stay on pageable CPU, while each active block is loaded directly from safetensors onto the
compute device. Inline quality, completed-block restoration, and model-level KD are rejected for this placement until
their full-model forwards have streamed implementations. `auto` resource planning selects
`resident → cpu_offload → streaming` according to GPU and host limits.

The resident and `cpu_offload` compositions also expose an opt-in activation GPU cache through
`runtime.activations.gpu_cache`. `inputs` retains the compressed block inputs that are reread by every tuning and loss
pass; `both` also retains the teacher target outputs; `auto` tries those tiers in that order and falls back to pageable
host tensors when `torch.cuda.mem_get_info` cannot preserve the greater of `gpu_reserve_gib` and one activation
stream of free device memory. The stream-sized lower bound prevents two large caches from consuming the tuning and
factorization workspace. Explicit `inputs`
and `both` requests fail instead of silently changing placement. Cache placement is an execution policy and therefore
does not change resident semantic commit identity.

Block diagnostics report both objective-weighted MSE and a scale-independent normalized activation error. The
normalizer is the teacher activation's weighted mean square for that block, so
`normalized_error = weighted_mse / target_weighted_mean_square`. Tuning epoch events carry `normalized_loss`, and
`block.completed` plus the live `weight-errors.md` report carry entry/final normalized errors. This makes trends
comparable when hidden-state magnitude and calibration importance change across depth.

Quality workflows paired with `cpu_offload` or `streaming` use block-streamed BF16 baseline evaluation. The source
Transformers shell remains in pageable host RAM, exact prefix metadata is captured for each evaluation batch, one
decoder block visits the compute device at a time, and only the final norm/head are loaded for the suffix. The
NanoQuant side evaluates the exported packed artifact. Consequently neither side reloads a complete dense model onto
CUDA for large-model quality comparison.

An `auto` planner may choose an executor, but the resolved plan records the choice and the user can require a specific one. An executor cannot change mathematical settings to make a run fit without creating a visible plan revision.

## 2. Resource plan

Before execution, the planner estimates:

- source checkpoint bytes and sharding;
- packed output bytes;
- active block and largest matrix sizes;
- factorizer peak temporary tensors by dtype;
- dense or approximate Hessian storage;
- calibration activation streams;
- optimizer and gradient state;
- evaluator memory;
- temporary and committed disk;
- expected transfer volume;
- estimated stage durations from historical profiles when available.

Example:

```text
Model:                     70B, BF16 source
Source checkpoint:         140.2 GiB
Selected executor:         streaming
Largest source block:        1.9 GiB
Factorization workspace:     9.8 GiB
Activation streams:          8.0 GiB
Activation backend:          mmap (NVMe)
Estimated packed output:     14.7 GiB
Temporary disk required:    173.5 GiB
GPU limit:                   44.0 GiB
CPU limit:                   64.0 GiB
Global KD:                 disabled by recipe
```

The planner includes safety margins and fails before expensive work if minimum disk or memory is unavailable.

## 3. Resident execution

For small models, avoiding transfers is the fastest path:

- source/working model and calibration activations remain on GPU;
- calibration statistics remain on GPU when practical;
- block targets and compressed inputs reuse preallocated buffers;
- packing can occur asynchronously after a block is finalized;
- no generic cleanup function empties allocator caches inside hot loops;
- resource scopes release known tensors deterministically.

Resident execution uses the same stage and checkpoint contracts as streaming. It is an optimized placement, not a separate implementation.

## 4. Streaming 70B execution

The current two-model mental model must be replaced with source streaming. A 70B BF16 model is roughly 140 GiB; two copies are already roughly 280 GiB before activation, factorization, and tuning state.

The streaming block loop is:

1. Read original inputs for block `N` from the teacher activation store.
2. Materialize original block `N` directly from sharded safetensors onto the compute device.
3. Run batched teacher outputs and write them to the target-output store.
4. Create the working block from the same source tensors or preserve it if memory permits.
5. Tune non-factorized parameters according to the plan.
6. For each target layer, load only required objective statistics, factorize, evaluate retry policy, tune, freeze, and checkpoint the accepted result.
7. Run the completed quantized block over the compressed-input store.
8. Atomically commit the packed/frozen block and both next-block activation streams.
9. Release the source block, working block, targets, Hessians, and workspaces.
10. Advance to block `N + 1`.

Peak model weight memory is bounded by the active block plus temporary factors. The full source remains memory-mapped or read from sharded safetensors; no full `state_dict` is constructed.

### Source reader

The `ModelSource` port supports:

- reading config/tokenizer metadata;
- listing tensor keys, shapes, dtypes, shards, and hashes without materializing tensors;
- reading a tensor or contiguous tensor group directly to a destination device;
- safe memory mapping;
- optional one-block prefetch;
- revision and integrity verification.

The model adapter maps logical block/layer identities to source tensor keys. Tied embeddings are represented once and referenced explicitly.

### Activation stores

For 128 samples, sequence length 2048, hidden size 8192, and BF16, one dense activation stream is approximately 4 GiB. Teacher and compressed streams together are approximately 8 GiB, before targets and scratch buffers.

The `ActivationStore` interface supports:

```python
class ActivationStore(Protocol):
    spec: ActivationSpec

    def read(self, selection: BatchSelection, device: DeviceLike) -> TensorLease: ...
    def write(self, selection: BatchSelection, values: torch.Tensor) -> None: ...
    def commit_generation(self) -> ArtifactRef: ...
```

Backends include GPU tensors, pinned RAM, pageable RAM, and preallocated memory-mapped files. Stores use double buffering and batch-sized staging areas. They do not repeatedly allocate and pin complete multi-gigabyte tensors.

## 5. Scalable calibration

Calibration has distinct execution strategies with the same output schema.

### Forward-only streaming

This is the least expensive large-model mode. Blocks stream forward while input activation statistics are accumulated. It does not require a complete model or backward graph.

### Streamed forward/backward

When output-gradient or true-Fisher statistics are required:

1. stream the forward pass and commit block-boundary activations;
2. compute the loss at the final head;
3. walk blocks backward;
4. reload each block, recompute its forward values, propagate the boundary gradient, and accumulate compact statistics;
5. release the block and prior boundary data.

This trades I/O and recomputation for bounded memory. The manifest must distinguish these statistics from forward-only approximations.

### Distributed calibration

FSDP or pipeline-sharded calibration is an optional executor for multi-GPU environments. It still emits the canonical `CalibrationStats` artifact. Distributed behavior must not leak into factorizer interfaces.

### Hessian representations

Dense input covariance becomes costly at large dimensions. For FP32:

- `8192 × 8192` is about 256 MiB;
- `28672 × 28672` is over 3 GiB.

Supported objectives therefore include:

- diagonal;
- block-diagonal;
- low-rank plus diagonal;
- dense, computed one layer at a time under a workspace reservation.

The objective artifact records its approximation, sample count, token selection, regularization, condition diagnostics, and memory cost.

## 6. Commit hierarchy

Commit granularity balances lost work and overhead:

```text
run
  stage
    block
      layer
        attempt
```

- attempts are logged but only accepted attempts become committed layer results;
- a completed layer commit contains factors, metrics, seed, and plan identity;
- a completed block commit additionally contains next-block activations and frozen block state;
- a stage commit references all required child commits and passes stage validation;
- the run manifest points only to committed stage roots.

Default recovery loses at most the active layer. Large activation generations may commit at block granularity because partial next-block streams are not independently useful.

## 7. Atomic write protocol

For a local artifact:

1. allocate a temporary path under the target filesystem;
2. write data and metadata;
3. flush files and close handles;
4. calculate and verify content hashes;
5. write the commit descriptor last;
6. atomically rename the descriptor or directory into the committed namespace;
7. append a committed event and update the run reference.

Temporary paths are safe to remove after lease expiry. Readers ignore any artifact without a valid commit descriptor.

## 8. Resume algorithm

On resume:

1. acquire the run lease;
2. validate the manifest and requested recipe;
3. scan committed stage and loop-unit references, not arbitrary files;
4. verify hashes, schema versions, and semantic cache keys;
5. reconstruct executor state from the latest valid commit;
6. restore deterministic random streams from logical seed derivation;
7. resume the first incomplete unit;
8. emit a `run_resumed` event describing reused and discarded work.

Seeds are derived from stable identifiers:

```text
seed = derive(run_seed, stage_name, block_id, layer_id, attempt_number)
```

They do not depend on how many prior calls happened before a crash.

## 9. Fork versus resume

- **Resume** means identical semantic inputs and continuation of the same attempt.
- **Fork** means a new run with a parent and one or more changed semantic inputs.

If a user changes rank policy after block 20, the system must not call that a resume. It creates a fork, determines the earliest invalidated boundary, and documents any reused upstream artifacts.

## 10. OOM and resource failures

Resource fallbacks are predeclared policies. Example:

```yaml
runtime:
  on_cuda_oom:
    - reduce_stage_batch_size
    - move_activation_store_to_pageable_ram
    - move_activation_store_to_mmap
    - fail
```

Every fallback:

- emits an event with the requested and available memory;
- records whether numerical behavior may change;
- updates the execution plan revision;
- retries from a valid boundary;
- has a finite attempt limit.

Falling back from dense Hessian to diagonal changes the algorithm and therefore requires an explicit recipe policy and a visible derived run identity. It is not treated like changing a transfer batch size.

## 11. Concurrency and prefetch

Sequential error propagation limits block-level parallelism in the default algorithm. Safe overlap opportunities include:

- prefetching source tensors for the next layer/block;
- asynchronous D2H/H2D copies through dedicated streams;
- packing and checksumming a frozen prior block while the next block computes;
- parallel independent evaluation tasks;
- parallel factorization experiments on captured fixtures;
- sharded matrix kernels under a distributed factorizer.

The executor must prove buffer ownership with leases/events before overlapping work. Hidden asynchronous mutation is not an optimization strategy.

## 12. Scaling tests

The scaling suite includes:

- resident and streaming equivalence on the same tiny model;
- streaming with an artificially tiny RAM budget to force memory-mapped activations;
- source checkpoint sharding across many files;
- interruption after each layer and block commit;
- disk-full behavior before and during a commit;
- corrupted source and cached tensor detection;
- dense-Hessian rejection and approximate-objective fallback planning;
- a 70B metadata-only/dry-run plan in normal CI;
- periodic real large-model canary runs on designated hardware.
