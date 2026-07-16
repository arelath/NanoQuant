# ADR-0009: Colocate experiment definitions outside the library

Status: accepted
Date: 2026-07-16

## Context

Pinned model revisions, evidence paths, experiment intents, numbered outputs, and machine-local reference paths lived
under `src/nanoquant/recipes`. They were campaign definitions rather than reusable library behavior, yet setuptools
included them in the installable `nanoquant` package. Active and legacy chronology numbers also shared one flat
namespace.

## Decision

All Python experiment definitions live in the top-level `experiments/recipes` package. Legacy replay definitions live
in its `legacy` subpackage. Numbered launchers remain thin sibling scripts and import the canonical package as
`recipes`. The `experiments` directory itself is not a package, preventing the second import spelling
`experiments.recipes`. The installable library defines recipe schemas, workflow types, resolution, and validation but
contains no concrete experiment definitions.

## Consequences

- research campaign state is excluded from built `nanoquant` distributions;
- active and legacy numbering spaces are structurally distinct;
- pytest and mypy include `experiments` as an explicit source root;
- standalone tools bootstrap the experiments path before importing legacy definitions;
- contract tests reject a recreated `src/nanoquant/recipes` tree, library imports of `recipes`, or an
  `experiments/__init__.py` file.

## Alternatives considered

A thinned library recipes package was rejected because no current definition is model-agnostic. Inline launcher
definitions were rejected because numbered filenames cannot form a derivation hierarchy. YAML remains a compatible
future representation after wrapper-object codec support exists.

## Validation

Launcher `runpy` identity tests, alternate-working-directory imports, architecture contracts, the full test suite,
Ruff, and strict mypy over both `src/nanoquant` and `experiments/recipes` validate the boundary. Detailed mechanics
and migration inventory are recorded in `Docs/26-experiment-definition-relocation.md`.
