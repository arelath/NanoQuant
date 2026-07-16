# Experiment Definition Relocation

Status: accepted and implemented
Date: 2026-07-16

## 1. Problem

Experiment definitions currently live in two places:

- `experiments/` holds the five numbered zero-argument launchers (`001`–`005`);
- `src/nanoquant/recipes/` holds the actual definitions those launchers import, plus
  legacy replay recipes (`experiment008/011/013/018`, `legacy_short_decode`) and the
  shared `base_compression` recipe.

Everything in `src/nanoquant/recipes` is experiment-specific, not generic library code:

- pinned model and dataset revisions (`dcc83ea…`, `093f9f3…`);
- a promoted calibration artifact rooted at `evidence/m3/experiment018-calibration`;
- numbered output layouts (`outputs/NNN-<slug>/…`);
- experiment intents, hypotheses, and baseline-run names;
- a machine-local reference path (`D:\dev\research\llama.cpp`).

That content ships inside the installable `nanoquant` package (`[tool.setuptools.packages.find] where = ["src"]`),
mixes research campaign state into the library, and contradicts both the intended role of a recipes area
(generic reusable configuration) and the configuration design in
[Docs/03-configuration-and-runs.md](03-configuration-and-runs.md) /
[Docs/03-configuration-reference.md](03-configuration-reference.md), which treat a recipe as an input the
application consumes, not a module the application contains.

We accept the divergence from the original YAML-first design — Python definitions stay (see §9 for the YAML
option) — but the definitions must live under `experiments/`, and `src/` must contain no experiment-specific code.

### 1.1 Current inventory

| Module (`src/nanoquant/recipes/`) | Contents | Derives from |
| --- | --- | --- |
| `base_compression.py` | Shared pinned Gemma 3 1B `RunConfig`; `compression_export_recipe()` numbered-output factory | — |
| `experiment001.py` | Config + `CompressionBenchmarkExperiment` | `base_compression` |
| `experiment002.py` | Config + `QualityEvaluationExperiment` | standalone |
| `experiment003.py` | Config + `CompressionQualityExperiment` | `base_compression` |
| `experiment004.py` | Config + `RankExpansionExperiment` | `experiment003` |
| `experiment005.py` | Config + `RankExpansionExperiment` | `experiment003` |
| `experiment008.py` | Legacy replay config | `experiment013` |
| `experiment011.py` | Legacy replay config + `RuntimeBenchmarkExperiment` | standalone |
| `experiment013.py` | Legacy replay config | `experiment018` |
| `experiment018.py` | Legacy parity config | `base_compression` |
| `legacy_short_decode.py` | Retained legacy benchmark fixture | standalone |

Consumers outside the package (all dependencies point *into* the recipes package; nothing inside
`src/nanoquant` imports it, so the move cannot break the library):

- the five launchers in `experiments/`;
- twelve unit test files (`test_base_compression_recipe`, `test_promoted_recipes`, `test_experiment003/004/005`,
  `test_compression_benchmark_workflow`, `test_compression_export_workflow`, `test_compression_quality_workflow`,
  `test_quality_evaluation_workflow`, `test_resident_workflow`, `test_runtime_benchmark_workflow`,
  `test_short_decode_benchmark`);
- `tools/run_gemma_parity.py` and `tools/run_gemma_global_distillation.py`.

## 2. Decision

1. **Single home.** Every experiment definition — configs, experiment/request wrapper objects, shared bases,
   and legacy replay recipes — lives under `experiments/`, colocated with the numbered launchers. This is the
   arrangement ADR-0005 already describes: a runfile "constructs the canonical typed `RunConfig` or loads one
   colocated canonical recipe."
2. **`src/nanoquant` carries no experiment content.** The `nanoquant.recipes` package is deleted, not thinned.
   Nothing generic remains in it today: the one default source for generic configuration is the schema itself
   (`nanoquant/config/schema.py`, per Docs/03 §13), and model-family defaults belong to adapters. If genuinely
   generic, model-agnostic recipe templates are ever wanted, they should be YAML data files (e.g. a top-level
   `recipes/` directory), not Python modules in the library.
3. **Launchers stay thin.** The numbered runfiles keep their exact shape; only the import line changes.
4. **The boundary is enforced by contract tests**, not convention (§7).

## 3. Target layout

```text
experiments/
  001-compress-gemma-3-1b-it.py            # thin numbered launchers, unchanged role
  002-benchmark-gemma-3-1b-it.py
  003-compress-and-benchmark-gemma-3-4b-it.py
  004-gemma-3-4b-it-vproj-plus30.py
  005-gemma-3-4b-it-vproj-double-request.py
  recipes/                                 # importable package: the experiment definitions
    __init__.py                            # re-exports the promoted names (same surface as today)
    base_compression.py                    # shared pinned base + compression_export_recipe()
    experiment001.py
    experiment002.py
    experiment003.py
    experiment004.py
    experiment005.py
    legacy/                                # legacy-chronology replays, separated from the
      __init__.py                          # active chronology so the two numbering spaces
      experiment008.py                     # cannot collide when the new chronology reaches 008
      experiment011.py
      experiment013.py
      experiment018.py
      short_decode.py                      # was legacy_short_decode.py
```

Notes:

- `experiments/` itself deliberately has **no `__init__.py`**. It is a folder of scripts plus one package.
  This guarantees exactly one import spelling for the definitions (`recipes`, never `experiments.recipes`),
  which the identity assertions in the tests depend on (§4).
- The `legacy/` subpackage resolves the latent numbering collision: `experiment008/011/013/018` are *legacy*
  chronology numbers, while `001`–`005` belong to the reset chronology (Docs/03 §10). Today both live flat in
  one package; the split makes the distinction structural. `recipes/__init__.py` continues to re-export all
  promoted names so most importers see one flat surface.
- Derivation chains survive as relative imports: `experiment004/005` import `.experiment003`;
  `legacy/experiment018` imports `..base_compression`; `legacy/experiment013` imports `.experiment018`.
- `legacy/short_decode.py` is used only by tests; it stays with the other legacy replays because it is a
  canonical record of migrated legacy behavior, not merely test scaffolding. (Moving it to `tests/support/`
  was considered and rejected for that reason.)
- The `compression_export_recipe()` factory moves with `base_compression.py`. It encodes research-side
  conventions (the `outputs/NNN-<slug>` layout, the local llama.cpp checkout) and is therefore experiment
  content. The generic types it produces (`CompressionExportRecipe`,
  `resolve_compression_export_recipe`) already live in `nanoquant/compression_export_workflow.py` and stay there.

## 4. Import mechanics

The design must satisfy three execution contexts with **one** canonical import name, because
`test_experiment003/004/005`, `test_promoted_recipes`, and `test_quality_evaluation_workflow` execute launchers
via `runpy.run_path(...)` and assert `namespace["CONFIG"] is EXPERIMENT_00X_CONFIG`. Identity only holds if the
launcher and the test resolve the definitions to the same `sys.modules` entry.

| Context | How `import recipes` resolves |
| --- | --- |
| `python experiments/00X-….py` (any CWD) | `sys.path[0]` is the script's directory, `experiments/`, so the sibling package is found with zero boilerplate. |
| pytest (repo-root CWD) | `pythonpath = [".", "experiments"]` in `pyproject.toml` puts `experiments/` on `sys.path`. `runpy.run_path` does **not** add the script's directory for plain files, so the launcher's `import recipes` hits the entry the test process already created — identity is preserved. |
| `tools/run_gemma_parity.py`, `tools/run_gemma_global_distillation.py` | These scripts run with `sys.path[0] = tools/`, so they need a two-line bootstrap before the recipes import (see below). They already carry standalone-script conventions, so this is acceptable. |

Launcher after the change — the only edit is the import line:

```python
"""Experiment 003: compress and quality-benchmark pinned Gemma 3 4B."""

from recipes import EXPERIMENT_003, EXPERIMENT_003_CONFIG

from nanoquant.compression_quality_workflow import run_compression_quality_experiment

CONFIG = EXPERIMENT_003_CONFIG
EXPERIMENT = EXPERIMENT_003


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(CONFIG, EXPERIMENT, launcher_path=__file__)
    )
```

Tools bootstrap (kept import-only so ruff's E402 does not trigger): add a tiny `tools/_paths.py` whose import
side effect prepends `<repo>/experiments` to `sys.path`, then import normally:

```python
import _paths  # noqa: F401  (prepends <repo>/experiments to sys.path)

from recipes.legacy import EXPERIMENT_018_CONFIG
```

Rejected variant: making `experiments/` a package and importing `experiments.recipes`. It would let tests and
tools import with no `pythonpath` change, but every launcher would need a `sys.path` bootstrap (launchers run
with `experiments/` — not the repo root — as `sys.path[0]`), which pollutes the pristine runfile record
ADR-0005 protects, triggers E402 suppressions, and creates a second import spelling that silently breaks the
`is`-identity tests if any file uses the other one.

## 5. What changes where

### 5.1 `pyproject.toml`

```toml
[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
pythonpath = [".", "experiments", "tools"]  # `tools` supports existing tests that import tool modules

[tool.mypy]
python_version = "3.10"
strict = true
packages = ["nanoquant"]
mypy_path = ["src", "experiments"]     # new: lets mypy resolve both trees
```

`tools` is included because existing unit tests import the standalone launchers as `tools.*`; direct tool execution
still resolves the colocated `_paths.py` through `sys.path[0]`. The moved package must stay under strict mypy (it is
today, as part of `nanoquant`). The validation command in
AGENTS.md becomes:

```powershell
.\.venv\Scripts\python.exe -m mypy src/nanoquant experiments/recipes
```

Passing explicit paths keeps the numbered launcher files (invalid module names like `001-…`) out of mypy's
walk. Verify during implementation that the editable-install/`mypy_path` combination resolves `nanoquant`
imports from `experiments/recipes`; add a `py.typed` marker or adjust `mypy_path` if not.

`[tool.setuptools.packages.find] where = ["src"]` needs no change — deleting `src/nanoquant/recipes` removes
the experiment content from the wheel automatically, which is a goal of this design, not a side effect. The
runtime distribution under `packaging/runtime` is unaffected (it packages only `nanoquant` + `nanoquant.runtime`).

### 5.2 Code moves and import rewrites

| File | Change |
| --- | --- |
| `src/nanoquant/recipes/*` | `git mv` to `experiments/recipes/`, then move the four legacy modules + `legacy_short_decode.py` into `experiments/recipes/legacy/` (rename to `short_decode.py`). Fix relative imports across the `legacy/` boundary (`..base_compression`). |
| `experiments/00[1-5]-*.py` (5 files) | `from nanoquant.recipes import …` → `from recipes import …`. |
| Test files importing `nanoquant.recipes` (12 files, §1.1) | Same mechanical rewrite. Two files use submodule paths: `test_resident_workflow.py` → `from recipes.legacy.experiment018 import EXPERIMENT_018_CONFIG`; `test_short_decode_benchmark.py` → `from recipes.legacy.short_decode import …`. |
| `tools/run_gemma_parity.py`, `tools/run_gemma_global_distillation.py` | Add the `_paths` bootstrap (§4) and rewrite the import. |
| `.venv` | Reinstall / clean the editable install so no stale `nanoquant.recipes` bytecode shadows the move. |

The existing filename assertion in `test_promoted_recipes.py` (`Path("experiments").glob("*.py")`) is
unaffected: the new `recipes/` directory is not matched by a top-level `*.py` glob.

### 5.3 Documentation

- `AGENTS.md`: repository orientation entry for `src/nanoquant/recipes/base_compression.py` →
  `experiments/recipes/base_compression.py`; update the mypy command.
- `Docs/22-base-compression-recipe.md` and any other doc referencing `nanoquant.recipes` paths.
- `Docs/03-configuration-and-runs.md` §10: unchanged in substance; optionally add one sentence noting that
  canonical recipes are colocated under `experiments/recipes/`.
- Record the decision as **ADR-0009: experiment definitions are colocated under `experiments/` and excluded
  from the library package**, referencing this document for mechanics.

## 6. What counts as experiment content (the rule going forward)

A definition belongs under `experiments/`, never `src/`, if it contains any of:

- an experiment number, name, purpose, hypothesis, or baseline-run reference;
- a pinned model, tokenizer, or dataset revision chosen for a research campaign;
- a promoted/prepared artifact hash or `evidence/` path;
- numbered output layouts (`outputs/NNN-…`) or `Results/NNN` conventions;
- machine-local paths (llama.cpp checkout, local caches).

`src/nanoquant` keeps: the schema and its defaults (`config/schema.py` — the single default source), workflow
and experiment *types* (`CompressionBenchmarkExperiment`, `RankExpansionExperiment`, …), resolution/validation
logic, and adapters. In other words, `src` defines what a recipe *is*; `experiments/` holds the recipes that
*exist*.

## 7. Enforcement

Extend `tests/contract/test_architecture.py`:

```python
def test_src_contains_no_experiment_definitions() -> None:
    assert not Path("src/nanoquant/recipes").exists()
    violations = []
    for path in Path("src/nanoquant").rglob("*.py"):
        for imported in _imports(path):
            if imported == "recipes" or imported.startswith("recipes."):
                violations.append(f"{path}: {imported}")
    assert violations == []


def test_recipes_package_has_one_import_spelling() -> None:
    # experiments/ must not become a package; that would create a second
    # spelling (experiments.recipes) and break launcher/test module identity.
    assert not Path("experiments/__init__.py").exists()
```

Optional deeper guard (nice-to-have): an AST check that no module under `src/nanoquant` constructs
`IntentConfig(experiment_number=<int literal>)` or a module-level `RunConfig(...)`, catching re-introduction of
experiment constants under a different package name.

The existing `test_numbered_runfiles_are_thin` continues to police launchers; its forbidden-import set needs no
change (`nanoquant.recipes` no longer exists to forbid).

## 8. Migration order

1. Add `experiments/recipes/` via `git mv src/nanoquant/recipes experiments/recipes`; create `legacy/` and move
   the five legacy modules; fix intra-package relative imports; keep `__init__.py` re-exports intact.
2. Update `pyproject.toml` (pytest `pythonpath`, mypy settings).
3. Rewrite imports in the 5 launchers, 11 test files, and 2 tools scripts; add `tools/_paths.py`.
4. Add the new contract tests (§7); update AGENTS.md and doc references; add ADR-0009.
5. Reinstall the editable package; delete stray `__pycache__` from the old location.
6. Validate:
   - `.\.venv\Scripts\python.exe -m pytest -q` — including the runpy identity tests, which prove launcher and
     test bind the same module objects;
   - `.\.venv\Scripts\python.exe -m ruff check .`;
   - `.\.venv\Scripts\python.exe -m mypy src/nanoquant experiments/recipes`;
   - smoke: `.\.venv\Scripts\python.exe -c "import runpy, sys; sys.path.insert(0, 'experiments'); runpy.run_path('experiments/003-compress-and-benchmark-gemma-3-4b-it.py')"`
     from the repo root, and confirm a launcher imports cleanly when invoked from a different CWD.

The whole migration is mechanical; no numerical behavior, config values, or artifacts change. Run manifests
store launcher paths (`experiments/00X-….py`), which are unchanged, so provenance of completed runs is
unaffected.

## 9. Alternatives considered

### Keep a thinned `nanoquant.recipes` for "generic" bases

Rejected. Nothing in the package today is generic — `base_compression.py` pins a model revision, dataset
revisions, a prepared calibration artifact, and a local llama.cpp path. Keeping an empty-but-present package
invites experiment content to drift back in. Generic defaults already have a home (the schema); generic
templates, if ever needed, should be data (YAML), not library modules.

### Self-contained launchers (definition inline in each numbered file)

Attractive for single-file auditability, but rejected as the general shape: numbered filenames are not valid
module names, so `004`/`005` could not derive from `003` by import, and the legacy replay recipes (008–018)
have no launcher at all yet must remain importable by tests and tools. A launcher whose definition nothing else
derives from *may* inline its config later without violating this design.

### YAML recipes under `experiments/`

Compatible with this design and already half-supported: `nanoquant.config.load_config()` decodes `.yaml` into
`RunConfig` today. What is missing is decoding for the experiment wrapper objects
(`CompressionBenchmarkExperiment`, `RankExpansionExperiment`, `QualityEvaluationExperiment`,
`RuntimeBenchmarkExperiment`, `ShortDecodeBenchmarkExperiment`), which carry export paths, task lists, and
parent-run references outside `RunConfig`. If those gain codec support, a numbered experiment can become
`experiments/00X-name.yaml` plus either a generic loader-launcher or `nanoquant quantize experiments/00X.yaml`,
per ADR-0005's provision for YAML execution. That is an optional follow-up; it does not block this relocation,
and Python definitions remain first-class either way.

## 10. Risks

- **Top-level module name `recipes`.** A same-named third-party package would collide. Risk is low (none in
  the current environment) and `experiments/` precedes site-packages on `sys.path` in every supported context;
  the contract tests pin the single spelling. If it ever bites, the package renames trivially.
- **Dual-import drift.** Someone adds `experiments/__init__.py` or imports `experiments.recipes`, producing
  duplicate module objects and breaking `is`-identity tests. Covered by
  `test_recipes_package_has_one_import_spelling`.
- **mypy resolution across two roots.** Strict mode must keep covering the moved package; verify the
  `mypy_path`/editable-install combination during implementation (§5.1).
- **Stale bytecode.** The old `src/nanoquant/recipes/__pycache__` or a non-editable install could shadow the
  move; step 5 of the migration handles it.

## 11. Implementation result

Implemented on 2026-07-16. All active definitions moved to `experiments/recipes`, legacy chronology definitions moved
to `experiments/recipes/legacy`, and `src/nanoquant/recipes` was removed. Launchers, tests, and standalone tools use
the single canonical `recipes` spelling. Architecture contracts enforce the boundary, ADR-0009 records the decision,
and the editable installation was rebuilt to remove the old package surface.
