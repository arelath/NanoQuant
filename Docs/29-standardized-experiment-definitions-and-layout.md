# Standardized Experiment Definitions and Output Layout

Status: implemented
Date: 2026-07-17

## Summary

Concrete experiments live in their numbered launcher under `experiments/`. The `experiments/recipes/` package
contains only generic, reusable templates and helpers. Every numbered launcher declares one required
`ExperimentIdentity` and builds one `ExperimentDefinition`.

The identity requires:

- an experiment number from 1 through 999;
- an unnumbered lowercase kebab-case name;
- a purpose;
- a falsifiable hypothesis;
- an explicit experiment, external, or no-baseline reference;
- one or more unique tags;
- an optional owner.

From that single declaration, the definition layer derives the canonical name, `IntentConfig`, run root, working
output root, public result root, report names, export locations, and launcher validation rules. A definition cannot
override these conventions.

The standard roots for Experiment 009 are:

```text
evidence/009/   durable run state and provenance
outputs/009/    rebuildable working files, packed files, checkpoints, and logs
Results/009/    public artifacts and reports
```

`Results` remains capitalized because it is the repository's existing publication contract.

## Ownership boundary

The source tree has three distinct responsibilities:

```text
experiments/
  009-some-experiment.py       concrete identity, semantic deltas, workflow choices, launcher
  recipes/
    _delta.py                  generic fail-closed dataclass delta helper
    _experiment.py             generic identity, layout, definition, and workflow builders
    base_compression.py        reusable unnumbered configuration templates
```

A concrete experiment is intentionally colocated with its executable launcher. Numbered recipe modules such as
`recipes/experiment009.py`, registries of concrete experiments, and a `recipes/legacy` namespace are not part of the
design. Python filenames beginning with digits cannot be imported normally, so tests that need a definition load the
launcher by path. Parent experiments are referenced through `ExperimentRef`, not by importing another launcher.

The `src/nanoquant` library continues to own workflow contracts, execution, validation, and publication. It does not
import the research-side `recipes` package or define concrete experiments.

## Object model

### `ExperimentIdentity`

`ExperimentIdentity` is the only place a launcher states its number and descriptive name. Callers provide a name
such as `compress-gemma-3-270m-it`; for Experiment 042 the value object derives
`042-compress-gemma-3-270m-it`.

Construction fails when:

- the number is outside 1–999 or is a boolean;
- the supplied name already has a numeric prefix;
- the name is not one lowercase kebab-case path component;
- purpose or hypothesis is blank;
- no tags are present, tags are blank, or tags repeat;
- an optional owner is blank.

`BaselineRef` makes the comparison choice explicit. `BaselineRef.experiment(ref)` serializes the referenced
experiment's canonical name, `BaselineRef.external(label)` records a non-experiment comparison, and
`BaselineRef.none(reason)` records why a baseline is not meaningful.

### `ExperimentLayout`

`ExperimentLayout(identity)` derives every repository-owned destination:

| Semantic location | Derived path for Experiment 009 |
| --- | --- |
| run root (`RunConfig.output.run_root`) | `evidence/009` |
| concrete resident run | `evidence/009/009-<name>` |
| working root | `outputs/009` |
| public root | `Results/009` |
| logical runtime | `outputs/009/logical` |
| packed runtime | `outputs/009/packed` |
| converter checkpoint | `outputs/009/llamacpp-checkpoint` |
| final GGUF | `Results/009/<release>-nanoquant.gguf` |
| summary | `outputs/009/009-<name>-summary.json` |
| benchmark | `outputs/009/009-<name>-benchmark.json` |
| quality JSON staging file | `outputs/009/009-<name>-quality.json` |
| quality Markdown public file | `Results/009/009-<name>-quality.md` |
| rank expansion report | `outputs/009/009-<name>-expansion.json` |

GGUFs, paired mmproj files, their export summaries, and their export/upload receipts are created directly in
`Results/009`. Final JSON statistics and other publishable files are validated and hard-linked there by the existing
publication service. Quality Markdown is also written directly to its derived public path; there is no recipe-facing
`quality_markdown_output` choice.

Intermediate logs belong below `outputs/NNN`, conventionally `outputs/NNN/logs` when a workflow needs a dedicated
log directory. Durable journals, manifests, commits, and resumable numerical state remain below `evidence/NNN`.

### `ExperimentDefinition`

`ExperimentDefinition` groups:

- the required identity;
- the fully materialized `RunConfig`;
- the typed workflow object;
- the derived layout.

Its constructor verifies that the config intent exactly equals the identity-derived intent and that
`config.output.run_root` exactly equals the layout run root. This makes path and identity drift an import-time error,
before expensive model work starts.

### Workflow builders

The reusable builders correspond to the workflow families currently used by numbered experiments:

- `define_compression_benchmark_experiment`;
- `define_quality_evaluation_experiment`;
- `define_compression_quality_experiment`;
- `define_rank_expansion_experiment`.

Builders accept only semantic choices such as expected block count, quality backend, WDDM guard, parent experiment,
rank multiplier, or export policy. They materialize all owned paths from the layout.

`CompressionExportPolicy` contains actual export choices: release name, runtime family, token-embedding type,
llama.cpp root, and optional Hugging Face publication. It does not contain output paths or an experiment number.

`ExperimentRef(number, name)` provides typed parent paths for rank-expansion workflows:

```text
parent run       evidence/NNN/NNN-<name>
parent packed    outputs/NNN/packed
parent quality   Results/NNN/NNN-<name>-quality.json
```

## Current experiment inventory

All ten active launchers now own their concrete definitions:

| Experiment | Workflow | Reusable template | Explicit semantic choices |
| --- | --- | --- | --- |
| 001 | compression + export + benchmark | `BASE_COMPRESSION_TEMPLATE` | BF16 baseline and common benchmark protocol |
| 002 | standalone quality evaluation | local delta of `BASE_COMPRESSION_TEMPLATE` | accepted candidate run and evaluation protocol |
| 003 | compression + export + quality | `GEMMA_3_4B_COMPRESSION_TEMPLATE` | 34 blocks, dense quality, WDDM guard |
| 004 | v_proj rank expansion | `GEMMA_3_4B_COMPRESSION_TEMPLATE` | parent 003, +30% bits, release name |
| 005 | v_proj rank expansion | `GEMMA_3_4B_COMPRESSION_TEMPLATE` | parent 003, doubled request, release name |
| 006 | compression + export + quality | `BASE_COMPRESSION_TEMPLATE` | 26 blocks and factorized quality |
| 007 | compression + export + quality | `GEMMA_3_270M_COMPRESSION_TEMPLATE` | 270M model pin and 18 blocks |
| 008 | large-model compression + quality | local delta of `LARGE_MODEL_COMPRESSION_TEMPLATE` | 12B model pin, 48 blocks, CPU-offload guards |
| 009 | compression + quality + Hugging Face publication | `GEMMA_3_270M_COMPRESSION_TEMPLATE` | 18 blocks, factorized quality, quality-gated public GGUF repository |
| 010 | compression + quality, without external publication | local delta of `GEMMA_3_270M_COMPRESSION_TEMPLATE` | Experiment 009 with 1,600-iteration cubic ADMM and no Hugging Face upload |

The reusable recipe package exports four unnumbered templates:

- `BASE_COMPRESSION_TEMPLATE`;
- `GEMMA_3_270M_COMPRESSION_TEMPLATE`;
- `GEMMA_3_4B_COMPRESSION_TEMPLATE`;
- `LARGE_MODEL_COMPRESSION_TEMPLATE`.

Templates deliberately retain `IntentConfig(experiment_number=None, name="unnamed-run", ...)`. A template becomes
a runnable experiment only through a builder and a complete identity.

Experiment 002's candidate run is an external input, not owned output, so its historical `evidence/m4/...` path
remains explicit. Similarly, a machine-local llama.cpp checkout and an optional Hugging Face destination are
external/tooling choices and are not derived from the experiment number.

## Launcher contract

For a standardized experiment:

```text
launcher filename stem == identity.canonical_name
launcher numeric prefix == identity.number
config.intent           == identity.to_config()
config.output.run_root  == identity-derived evidence/NNN
```

The launcher validator checks both the numeric prefix and, for canonical numbered names, the complete filename stem.
Ad hoc non-numbered fixtures may continue to use the lower-level `RunConfig` API, but numbered launchers cannot drift
from their identity.

A launcher follows this shape:

```python
EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=9,
        name="some-experiment",
        purpose="...",
        hypothesis="...",
        baseline=BaselineRef.external("..."),
        tags=("compression", "quality"),
    ),
    SOME_REUSABLE_TEMPLATE,
    expected_blocks=26,
)

if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
```

No `CONFIG`, `EVALUATION`, export path factory, numbered recipe module, or duplicated output constant is needed.

## Retention and publication rules

The three roots represent different lifecycle classes:

| Root | Contents | Retention rule |
| --- | --- | --- |
| `evidence/NNN` | manifests, journals, commits, metrics, durable artifacts, resumable state | authoritative run evidence; clean only with store-aware tools |
| `outputs/NNN` | logical/packed conversion, checkpoints, reports, logs | rebuildable/intermediate; never delete while a run is active |
| `Results/NNN` | final GGUF/mmproj, receipts, final statistics, Markdown reports | stable public output and zero-copy publication view |

Publication continues to use validated hard links for artifacts originating outside `Results/NNN`, so files are not
duplicated. Pre-convention GGUFs already present under `outputs/NNN` are validated and adopted into the final path by
hard link on retry. A workflow is not complete merely because a GGUF exists; required receipts, evaluation, and
publication must also succeed.

## Validation

Contract and unit tests enforce that:

- `experiments/recipes` contains only the four generic Python modules;
- no recipe module constructs a concrete `ExperimentIdentity`;
- every numbered launcher constructs an `ExperimentIdentity`;
- all ten launcher stems equal their canonical identity names;
- every active config uses the derived intent and run root;
- templates are unnumbered;
- derived active config hashes are distinct;
- quality Markdown paths are below `Results/NNN`;
- parent experiment paths are derived through `ExperimentRef`;
- launcher name and number mismatches fail before execution.

## Consequences

Adding an experiment now requires choosing identity and semantics, not coordinating several path strings. The number
appears once, the canonical prefix cannot be forgotten, and all workflows share one directory convention.

The deliberate tradeoff is that numbered launcher files are no longer tiny import shims: they are executable,
reviewable experiment specifications. Reuse happens through unnumbered templates and typed builders in `recipes`,
while concrete campaign intent stays next to the command that runs it.
