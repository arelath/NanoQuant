# ADR-0009: Colocate concrete experiment definitions with numbered launchers

Status: accepted
Date: 2026-07-17

## Context

Pinned model revisions, evidence paths, experiment intents, numbered outputs, and machine-local reference paths lived
under `src/nanoquant/recipes`. They were campaign definitions rather than reusable library behavior, yet setuptools
included them in the installable `nanoquant` package. Active and legacy chronology numbers also shared one flat
namespace.

## Decision

Every concrete experiment definition lives in its numbered launcher under `experiments/`. The sibling
`experiments/recipes` package contains only generic, unnumbered configuration templates and reusable definition
helpers. The `experiments` directory itself is not a package, preventing the second import spelling
`experiments.recipes`. The installable library defines workflow types, resolution, validation, and publication but
contains no concrete experiment definitions.

Each launcher owns one required `ExperimentIdentity` and one derived `ExperimentDefinition`. Parent experiments are
referenced by `ExperimentRef`, avoiding imports from Python filenames that begin with digits. Legacy recipe modules
are removed; workflow tests use local fixtures where they still exercise historical comparison contracts.

## Consequences

- research campaign state is excluded from built `nanoquant` distributions;
- concrete intent is reviewed next to the command that executes it;
- the recipe package cannot accumulate numbered campaign definitions;
- pytest and mypy include `experiments` as an explicit source root;
- standalone parity tools use the reusable unnumbered parity template;
- contract tests reject a recreated `src/nanoquant/recipes` tree, library imports of `recipes`, or an
  `experiments/__init__.py` file;
- contract tests require every numbered launcher to declare an identity and forbid concrete identities in recipes.

## Alternatives considered

A thinned library recipes package was rejected because campaign definitions are not installable library behavior.
Putting concrete definitions in `experiments/recipes` was implemented first, then superseded: it separated execution
from the experiment specification and let the recipe package become another concrete registry. Derivation now occurs
through generic templates and typed parent references, so numbered filenames do not need to form an import hierarchy.
YAML remains a compatible future representation after wrapper-object codec support exists.

## Validation

Launcher `runpy` identity tests, architecture contracts, the full test suite, Ruff, and strict mypy over `src`,
numbered launchers, and `experiments/recipes` validate the boundary. The implemented identity and layout contract is
recorded in `Docs/29-standardized-experiment-definitions-and-layout.md`; the earlier relocation plan in
`Docs/26-experiment-definition-relocation.md` is superseded.
