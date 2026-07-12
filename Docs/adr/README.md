# Architecture Decision Records

Architecture decision records (ADRs) preserve the context and tradeoffs behind decisions that constrain the rewrite. They are short, immutable historical records. If a decision changes, add a new ADR that supersedes the old one instead of rewriting history.

## Status values

- `proposed`
- `accepted`
- `rejected`
- `deprecated`
- `superseded by ADR-NNNN`

## Initial decisions

- [ADR-0001: Use one immutable hierarchical configuration schema](0001-hierarchical-configuration.md)
- [ADR-0002: Execute work as typed resumable stages](0002-resumable-stage-pipeline.md)
- [ADR-0003: Separate trainable and packed runtime representations](0003-separate-training-and-runtime-state.md)
- [ADR-0004: Support large models through block streaming](0004-block-streaming-for-large-models.md)
- [ADR-0005: Retain numbered zero-argument experiment runfiles](0005-numbered-zero-argument-runfiles.md)
- [ADR-0006: DBF remains research-only](0006-dbf-research-only.md)
- [ADR-0007: Calibration and Hessian support tiers](0007-calibration-and-objective-support.md)
- [ADR-0008: Legacy artifact compatibility is conversion-based](0008-legacy-artifact-compatibility.md)

## Template

```markdown
# ADR-NNNN: Title

Status: proposed  
Date: YYYY-MM-DD

## Context

What problem or decision pressure exists?

## Decision

What will be done?

## Consequences

What becomes easier, harder, required, or prohibited?

## Alternatives considered

What credible alternatives were rejected and why?

## Validation

How will we know the decision works?
```
