# ADR-0003: Separate Trainable and Packed Runtime Representations

Status: proposed  
Date: 2026-07-11

## Context

The current custom linear module transitions among latent trainable factors, hardened binary signs, parameters, buffers, int8 outlier master state, packed state, and custom-kernel execution. This creates complex mutation and serialization behavior and forces deployment code to understand research state.

## Decision

Define three explicit representations:

1. `TrainableNanoQuantState` for latent factors and tuning parameters;
2. `FrozenNanoQuantState` for immutable backend-independent logical values;
3. `PackedNanoQuantState` and `PackedNanoQuantLinear` for a specific versioned runtime layout.

Conversion is explicit, validated, and one-way in normal execution. The deployment runtime accepts only packed artifacts and has no dependency on training/factorization code.

## Consequences

- state transitions and artifact schemas become clear;
- runtime installation and model loading become smaller and safer;
- multiple backend layouts can derive from one frozen logical state;
- reference parity can isolate logical, packing, and kernel errors.

Costs include conversion code, temporary retention of logical state when repacking is desired, and explicit migrations for layout changes.

## Alternatives considered

### Keep one mode-switching `nn.Module`

Rejected because its convenience during initial research does not justify deployment coupling and mutation complexity.

### Store only packed backend state

Rejected as the sole research artifact because changing backends would require re-quantization and make logical validation difficult. Release artifacts may omit logical state when distribution size requires it, but the producing run retains lineage.

## Validation

- each conversion has round-trip/logical parity tests where applicable;
- runtime clean-install imports no research packages;
- packed state is immutable and validated before inference;
- old/new runtime outputs agree on frozen reference fixtures.

