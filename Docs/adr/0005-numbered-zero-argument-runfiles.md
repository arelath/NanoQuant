# ADR-0005: Retain Numbered Zero-Argument Experiment Runfiles

Status: proposed  
Date: 2026-07-11

## Context

Numbered experiment programs in the current NanoQuant codebase provide two benefits that a run database or generic CLI alone does not:

- directory order records the chronology of research;
- executing a file without parameters makes that file an understandable, repeatable record of the intended run.

The problem is not numbering or runfiles themselves. The problem is that runfiles can grow copied environment loading, logging, orchestration, defaults, and save/evaluation behavior that drifts between experiments.

## Decision

Retain numbered, descriptive, zero-argument Python runfiles as a first-class launcher.

- A runfile constructs the canonical typed `RunConfig` or loads one colocated canonical recipe.
- It declares experiment number, purpose, hypothesis, and baseline.
- It invokes one shared `run_experiment` application entry point.
- It accepts no experiment-defining command-line parameters.
- It contains no copied quantization, logging, checkpoint, or evaluation orchestration.
- Its path and content hash are stored in the run manifest.
- Completed experiment numbers/files are not reused for different semantics.
- YAML and generic CLI execution remain available for automation and sweeps.

## Consequences

Benefits:

- chronology remains visible in source control and directory listings;
- the command used for a run is trivial to record;
- important settings and intent are reviewable together;
- copying the prior experiment to the next number remains a fast research workflow;
- the canonical config and manifest still prevent default drift and preserve exact resolved behavior.

Costs:

- runfile immutability and numbering conventions need review discipline;
- launcher hashes become part of provenance;
- some experiment definitions may have both Python and generated YAML views;
- sweeps need explicit child-run naming and lineage.

## Alternatives considered

### Replace every experiment file with YAML

Rejected as the only workflow because it loses the convenient executable/no-argument record and Python composition for carefully controlled research runs.

### Use only a generic CLI with flags

Rejected because shell history or external notes would become part of reproducibility, and it is easy to omit a flag when recording the run.

### Keep current full orchestration scripts

Rejected because copied mechanics and defaults can drift. Only the thin launcher pattern is retained.

## Validation

- a numbered runfile runs with no arguments;
- its resolved configuration matches canonical serialization tests;
- changing the file changes launcher provenance;
- static checks prohibit CLI parsing and imports of internal orchestration/infra modules in experiment files;
- reports show experiment number, filename, launcher hash, and run ID;
- a completed run can be reproduced from its repository revision and runfile without shell parameters.

