# ADR-0002: Execute Work as Typed Resumable Stages

Status: proposed  
Date: 2026-07-11

## Context

Quantization runs take hours or days. Current orchestration is primarily a nested function call, with final checkpoint saving and ad hoc incremental metric files. A crash or OOM can lose substantial work, and reusing a file based only on its path does not establish compatibility.

## Decision

Represent the workflow as typed stages with declared inputs, outputs, semantic cache keys, validation, resource estimates, and commit granularity.

- Immutable content-addressed artifacts connect stages.
- Layer and block loop units have atomic commits.
- A run manifest references only committed roots.
- Logical seeds derive from stage/block/layer/attempt identity.
- Resume continues identical semantic inputs; changed inputs create a fork.
- Failure injection around every commit boundary is a release test.

## Consequences

Long runs become resumable, cacheable, auditable, and independently replayable. The cost is an artifact store, transaction protocol, schema evolution, and careful distinction between temporary and committed data.

Components can no longer rely on arbitrary mutable in-memory state surviving across stage boundaries. Any state required for resume needs a safe typed representation or the active unit must be deterministic to replay.

## Alternatives considered

### Save one whole-model checkpoint periodically

Rejected because it is expensive for large models, does not expose stage compatibility, and can still repeat substantial work.

### Serialize the Python process or arbitrary optimizer objects

Rejected for portability, security, and schema-evolution reasons.

### Use a general external workflow engine immediately

Deferred. Stable local stage contracts come first; an external scheduler can later implement the executor port if needed.

## Validation

- forced termination at every commit step resumes to an artifact equivalent to an uninterrupted control;
- semantic input changes invalidate the correct downstream stages;
- corrupt or partial artifacts are never reused;
- a layer/block replay can run from committed inputs alone.

