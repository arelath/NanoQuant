# Configuration, Recipes, and Run Identity

> The complete nested dataclass design, defaults, YAML, validation rules, runtime-state separation, and current-field migration map are specified in [Hierarchical Configuration Reference](03-configuration-reference.md). This document focuses on recipe lifecycle, identity, hashing, and run organization.

## 1. One canonical schema

The rewrite has one `RunConfig` schema. CLI flags, Python calls, and YAML or JSON recipes are input adapters that construct this schema; none owns separate defaults.

The concrete hierarchy is not merely a list of names. It is defined field-by-field in the [configuration reference](03-configuration-reference.md#11-complete-runconfig), including nested `ADMMConfig`, Hessian sampling and regularization, rank retry, each tuning phase, resource limits, activation storage, checkpointing, packing, evaluation, and observability.

```python
@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig

    schema_version: int = 1
    intent: IntentConfig = field(default_factory=IntentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    allocation: RankAllocationConfig = field(default_factory=RankAllocationConfig)
    factorization: FactorizationConfig = field(default_factory=FactorizationConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    block_tuning: BlockTuningConfig = field(default_factory=BlockTuningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
```

Configuration is immutable after resolution. Runtime counters, discovered tensor shapes, retry expenditure, cached Hessians, and output rows belong to `RunState` or stage artifacts, never to `RunConfig`.

## 2. Recipe format

A recipe contains both executable settings and the reason for the run:

```yaml
schema_version: 1

intent:
  experiment_number: 20
  name: gemma-3-4b-residual-outlier-ablation
  purpose: Determine whether residual-selected outliers reduce block loss enough to justify their bit cost.
  hypothesis: Residual selection will improve v_proj and o_proj more than Fisher selection at equal effective BPW.
  baseline_run: run_01J...
  owner: research
  tags: [gemma3, outliers, ablation]

model:
  source: google/gemma-3-4b-it
  revision: 0123456789abcdef
  sequence_length: 2048
  load_dtype: bfloat16

dataset:
  sources:
    - name: ultrachat_200k
      revision: 89abcdef
      weight: 0.75
    - name: wikitext2
      revision: 456789ab
      weight: 0.25
  selection_seed: 0

reproducibility:
  seed: 0
  deterministic: true

calibration:
  method: online_fisher
  sample_count: 256
  shrinkage: 0.6
  objective:
    kind: diagonal

allocation:
  strategy: sensitivity
  target_bpw: 1.0
  bounds:
    multiple: 32

factorization:
  implementation: nanoquant_admm
  admm:
    outer_iterations: 800
    inner_iterations: 5
    penalty_schedule: cubic
    transpose_wide: false

outliers:
  selector: residual
  fraction: 0.001
  storage_dtype: bfloat16
  charge_to_bit_budget: true

runtime:
  executor: auto
  resources:
    gpu_memory_gib: 44
    cpu_memory_gib: 64
    temporary_disk_gib: 500

evaluation:
  default_tier: quick

output:
  run_root: runs
```

The source revision and dataset revisions are mandatory for a reproducible run. A friendly model name without a revision may be accepted during interactive exploration, but the resolved recipe must pin the discovered revision and mark the input as initially floating.

## 3. Resolution phases

Recipe processing has four phases:

1. **Parse:** read the input without applying defaults.
2. **Resolve:** apply schema defaults, aliases, model-adapter defaults, and environment-independent derived settings.
3. **Validate:** check individual fields and cross-field constraints.
4. **Plan:** inspect source metadata and resources to produce shapes, storage estimates, and an execution plan.

`nanoquant inspect-recipe` performs all four without starting calibration or quantization.

Examples of cross-field validation:

- dense Hessian requested for a dimension whose estimated workspace exceeds the declared limit;
- global KD enabled under a single-GPU streaming plan without a supported teacher-target cache;
- outlier storage charged as FP16 while configured for INT8;
- target BPW smaller than unavoidable metadata and embedding storage;
- a runtime backend that does not support the selected rank alignment;
- full evaluation requested without a tokenizer revision or task revision;
- resume requested with a recipe that changes an upstream stage input.

## 4. Semantic identities and hashes

Not every recipe field should invalidate every cache. The rewrite defines stage-specific semantic identities:

```text
dataset_key     = hash(dataset sources, revisions, selection, tokenizer, sequence length)
calibration_key = hash(dataset_key, source model revision, calibration method/config)
plan_key        = hash(calibration_key, source shapes, allocation/outlier budget config)
layer_key       = hash(source tensor hash, layer plan, objective artifact, factorizer config, seed)
packed_key      = hash(frozen model state, packed format version, backend layout config)
evaluation_key  = hash(model artifact, evaluator/task revisions, evaluation spec)
```

Fields such as run title, notes, console verbosity, or report theme do not invalidate numerical stages. Changing a factorizer iteration count invalidates factorization and all downstream stages but reuses compatible calibration and planning inputs.

Every cache hit records the producing run and verifies schema, content hash, and semantic compatibility. File existence alone is not a cache contract.

## 5. Run identity and lifecycle

A run receives a sortable unique ID at creation. The ID identifies the attempt; content hashes identify reusable computation. This permits two attempts with identical recipes to remain separately auditable while sharing immutable stage artifacts.

Run states are:

```text
created → planned → running → completed
                    ↘ failed
                    ↘ cancelled
                    ↘ interrupted
```

`running` also records the active stage and loop unit. A process heartbeat lease distinguishes an active process from an abandoned run without rewriting historical events.

## 6. Run directory

```text
runs/<run-id>/
  manifest.json
  recipe.input.yaml
  recipe.resolved.yaml
  plan.json
  environment.json
  source.json
  launcher.json
  events.jsonl
  status.json

  refs/
    calibration.json
    quantization-plan.json
    model-artifact.json
    evaluations/

  checkpoints/
    progress.json

  reports/
    summary.md
    comparison.md
    diagnostics.md

  logs/
    console.txt
```

Large immutable tensors are held in the artifact store and referenced by hash; they are not duplicated in every run directory. The run directory remains useful even if cache garbage collection later removes optional temporary artifacts.

## 7. Manifest

The manifest is the authoritative self-description of a run. It contains:

- run ID and parent/fork run ID;
- experiment number plus launcher kind, repository-relative path, and content hash;
- purpose, hypothesis, owner, and tags;
- input and resolved recipe hashes;
- source model identity and file hashes;
- dataset and tokenizer identities;
- code revision and dirty-tree patch hash when applicable;
- package, PyTorch, CUDA, driver, compiler, and kernel versions;
- executor and resource plan;
- stage inputs, outputs, statuses, and timings;
- warnings and fallback decisions;
- evaluation and benchmark references;
- terminal status and conclusion.

Launcher provenance is a separate typed value, not part of `RunConfig` defaults:

```python
@dataclass(frozen=True, slots=True)
class LauncherProvenance:
    kind: str  # numbered_runfile, yaml, cli, python_api
    repository_root: Optional[str]
    repository_relative_path: Optional[str]
    content_hash: Optional[str]
    experiment_number: Optional[int]
    command_arguments: tuple[str, ...]
```

For a numbered zero-argument runfile, `command_arguments` is empty and the parsed filename number must agree with `config.intent.experiment_number`. The exact local absolute path may be included as diagnostic environment data, but repository-relative path plus revision/hash provides portable provenance.

Environment capture uses an allowlist. It must never dump all environment variables because tokens and secrets are commonly present there.

## 8. Forking a run

Experiments frequently change only downstream behavior. A fork declares a parent run and a recipe patch:

```text
nanoquant fork runs/<parent> \
  --set factorization.outer_iterations=1000 \
  --purpose "Test whether late ADMM convergence improves export error"
```

The planner shows which artifacts remain reusable and why others are invalidated:

```text
REUSE dataset
REUSE calibration
REUSE quantization plan
INVALIDATE layer factorization: factorization.outer_iterations changed
INVALIDATE packing: upstream frozen state changed
INVALIDATE evaluation: model artifact changed
```

No configuration is edited inside an existing historical run.

## 9. Configuration evolution

Schema migration rules are explicit:

- readers support the current schema and a documented window of prior schemas;
- migrations are pure transformations with golden tests;
- removed fields either map unambiguously or fail with a remediation message;
- defaults are applied according to the source schema version before migration;
- resolved recipes are never silently rewritten in place.

## 10. Experiment numbering and naming

Sequential numbers are retained as a useful chronological research record. The active chronology was reset after
legacy migration lessons moved into shared code and evidence; its first zero-argument launcher is
`experiments/001-compress-gemma-3-1b-it.py`.

The number is a human chronology label, not the immutable run identity:

- a run ID identifies one attempt/resume lineage;
- content hashes identify reusable artifacts;
- the experiment number shows research order;
- the filename slug summarizes intent;
- `IntentConfig` records purpose, hypothesis, and baseline;
- the manifest records launcher path/hash and resolved configuration.

Completed experiment files are not edited into a different experiment. A changed hypothesis or semantic setting receives the next number. The file remains thin and uses the canonical configuration/application API; it does not duplicate orchestration. Full conventions are in [Lessons Carried Forward](12-lessons-carried-forward.md#2-preserve-numbered-experiment-files) and [ADR-0005](adr/0005-numbered-zero-argument-runfiles.md).
