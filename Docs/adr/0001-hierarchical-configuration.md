# ADR-0001: Use One Immutable Hierarchical Configuration Schema

Status: proposed  
Date: 2026-07-11

## Context

The current application repeats configuration fields and defaults across CLI dataclasses, a model/Hub dataclass, and a dictionary factory. Values are manually copied between representations and then passed through mutable dictionaries that also receive runtime counters and report rows.

This makes default drift likely, weakens validation, obscures semantic cache identity, and makes it difficult to explain which settings a run actually used.

## Decision

Use the nested, frozen `RunConfig` hierarchy specified in [the configuration reference](../03-configuration-reference.md).

- Dataclass fields are the only source of defaults.
- YAML recursively decodes into these dataclasses.
- The CLI applies sparse path/value overrides and declares no algorithm defaults.
- Python callers pass the same `RunConfig` type.
- Input and resolved recipes use the same type.
- Cross-field validation completes before expensive work.
- Derived ranks, resource estimates, progress, retry expenditure, and results use separate typed objects.
- The resolved recipe is serialized completely and becomes an immutable run input.

## Consequences

Benefits:

- one discoverable hierarchy;
- generated CLI/help/reference documentation;
- stable semantic hashing by subtree;
- no hidden mutation during a run;
- explicit migration from legacy fields;
- easier strategy-level tests and recipe comparison.

Costs:

- a schema-aware decoder/override mechanism is required;
- immutable nested updates are more verbose in raw Python, mitigated by builders and `replace` helpers;
- every legacy field must receive an explicit mapping or rejection;
- component APIs must stop accepting arbitrary config dictionaries.

Prohibited:

- new CLI/model-specific configuration dataclasses with duplicated defaults;
- stage functions accepting the entire config when only one subtree is needed;
- inserting keys beginning with `_` or any progress/result values into configuration;
- numeric sentinel values such as zero meaning both a value and disabled state when an explicit optional/bool field is clearer.

## Alternatives considered

### Keep a flat dataclass

Rejected because it preserves name prefixes and weak ownership boundaries as the feature set grows.

### Use untyped YAML dictionaries

Rejected because misspellings and invalid combinations would reach expensive runtime stages and semantic hashing would remain fragile.

### Maintain separate CLI and internal models

Rejected because their defaults and meanings can drift. CLI inputs are sparse overrides, not a second configuration schema.

### Let every component define local defaults

Rejected because a resolved run could no longer state the complete behavior without recreating component construction logic.

## Validation

- minimal YAML, CLI, and Python inputs produce identical canonical serialization;
- schema introspection finds one default for every field;
- every legacy field is mapped or rejected explicitly;
- immutable mutation attempts fail;
- config hash tests distinguish semantic versus presentation changes;
- import checks find no old parallel configuration classes after cutover.

