# ADR-0004: Support Large Models Through Block Streaming

Status: proposed  
Date: 2026-07-11

## Context

A 70B BF16 checkpoint is roughly 140 GiB. Keeping both a working model and full-precision teacher can require roughly 280 GiB before activations and workspaces. Device-map offload moves placement but does not remove the two-model architecture or guarantee bounded memory.

NanoQuant's sequential block reconstruction naturally needs only the current source block, working block, teacher/compressed activation streams, and factorization workspace.

## Decision

Implement resident, offload, streaming, and distributed executors behind the same pipeline contracts. The streaming executor:

- reads block-aligned tensors directly from safe sharded checkpoints;
- materializes one active block/layer;
- stores activation streams in GPU, RAM, or memory-mapped disk through one interface;
- commits finalized layers/blocks incrementally;
- applies explicit workspace/resource limits;
- supports forward-only or streamed forward/backward calibration;
- writes packed output shards incrementally.

Algorithm components do not branch on the executor.

## Consequences

70B execution can run with bounded RAM/VRAM given adequate disk and time. The same streaming executor also enables low-memory development tests.

Costs include substantial source/activation I/O, adapter tensor mapping, prefetch/buffer ownership complexity, and the need for approximate Hessian forms at very large dimensions.

## Alternatives considered

### Rely only on `device_map=auto`

Rejected because it does not supply stage-level memory planning, activation tiers, atomic progress, or a one-block source/teacher model.

### Require a multi-GPU host large enough for every model

Rejected because it conflicts with the stated 1B-to-70B portability goal. Distributed execution remains an acceleration option.

### Create a separate 70B algorithm implementation

Rejected because it would diverge from small-model behavior and double correctness work.

## Validation

- resident and streaming executors produce equivalent results on the same fixture;
- peak weight memory stays within active block plus declared workspace;
- forced mmap activation tests pass under constrained RAM;
- a large-model canary resumes after interruption and yields valid packed blocks;
- planner estimates are compared with actual peak memory, I/O, and disk.

