# Probe Reconstruction Cache

## Motivation

The corrected-codebook splice probe previously repeated every 800-iteration
factorization when only the KL evaluation window changed. The nine-block
screen and confirmation each refactorized 27 identical matrices, making
fresh confirmation unnecessarily expensive.

The probe now supports:

```text
--reconstruction-cache <directory>
```

It stores only the two fitted dense matrices needed by later analysis:

- rank-970 free-word reconstruction;
- equal-bit corrected-codebook reconstruction.

Operator and downstream scale refits remain evaluation-specific and are
recomputed from the requested fit tokens.

## Identity

Every cache entry has a canonical SHA-256 key covering:

- cache algorithm version;
- model source, revision, and full safetensors hash;
- calibration manifest and safetensors hashes;
- block, projection path, and factorization orientation;
- factorization shape;
- baseline and candidate ranks;
- code width, corrections, free rows, and assignment candidates;
- every ADMM, codebook-update, scale-fit, shrinkage, and seed setting.

Changing any numerical input produces a different key. Paths themselves are
not treated as content identity.

## Durable format

Each key is an atomically published directory containing:

- `reconstructions.safetensors`;
- `manifest.json`.

The manifest records the full identity, tensor-file hash, rank, source
matrix shape, reconstruction metrics, and candidate index metrics.

Reads strictly validate:

- schema and key;
- exact identity equality;
- SHA-256 of the tensor file;
- tensor inventory, shape, and finiteness;
- rank and metadata structure;
- compatibility with the current source tensor.

Missing entries are normal misses. Incomplete, corrupted, or incompatible
entries fail loudly rather than being silently adopted or overwritten.

## Provenance

Every probe output now records:

- whether caching was enabled;
- cache root;
- hit and miss counts;
- key for every block/projection unit;
- model and calibration hashes.

This makes a cached result distinguishable from a newly factorized one
without changing its numerical result contract.

## Real-model smoke test

A pinned Gemma block-2 down-projection invocation was run twice with a
reduced one-iteration diagnostic:

| Invocation | Hits | Misses | Free KL | Mixed KL |
| --- | ---: | ---: | ---: | ---: |
| First | 0 | 1 | 0.4743928602 | 1.7444182365 |
| Second | 1 | 0 | 0.4743928602 | 1.7444182365 |

Both KL results replay exactly. This reduced diagnostic validates the cache
boundary only; it is not compression-quality evidence.

## Decision

Use one shared reconstruction cache for the remaining depth sweep and
composition confirmations. Full 800-iteration misses remain authoritative,
while subsequent evaluation-window changes reuse their validated dense
reconstructions.

Increment `RECONSTRUCTION_CACHE_ALGORITHM_VERSION` whenever the numerical
factorization, scale-fit, reconstruction orientation, or cached metadata
semantics change.
