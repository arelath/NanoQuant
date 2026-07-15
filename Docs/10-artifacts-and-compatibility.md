# Artifact Formats and Compatibility

## 1. Artifact principles

Artifacts are durable interfaces between expensive stages. They must be:

- safe to load without arbitrary code execution;
- versioned and self-describing;
- content-addressed and checksummed;
- valid independently of the process that wrote them;
- streamable for large models;
- inspectable without allocating all tensors;
- explicit about logical versus backend-packed representation;
- atomically committed;
- migratable when semantics permit.

A Python pickle or a full-model `torch.save` file is not the primary interchange format.

## 2. Artifact classes

| Artifact | Produced by | Consumed by |
| --- | --- | --- |
| `DatasetSelection` | dataset preparation | calibration/evaluation |
| `ActivationCapture` | model prefix/block execution | calibration/replay/quantization |
| `CalibrationStats` | calibrator | objective builder/planner |
| `ObjectiveArtifact` | objective builder | factorizer/scale fitter |
| `QuantizationPlan` | allocator/planner | block quantizer |
| `LayerResult` | factorization/tuning | block commit/replay/reporting |
| `FrozenBlock` | block workflow | next-block propagation/packer |
| `PackedModelArtifact` | packer/exporter | runtime/evaluation |
| `EvaluationResult` | evaluator | comparison/reporting |
| `RunManifest` | application services | resume/audit/reporting |

Every artifact has a small JSON descriptor and zero or more immutable tensor/data files.

## 3. Common descriptor

```json
{
  "artifact_type": "calibration_stats",
  "schema_version": 1,
  "artifact_id": "sha256:...",
  "created_at": "2026-07-11T18:42:31Z",
  "producer": {
    "component": "online-fisher-calibrator",
    "version": "1.0.0",
    "code_revision": "..."
  },
  "semantic_inputs": [],
  "files": [
    {
      "path": "stats-00001.safetensors",
      "size": 123456,
      "sha256": "..."
    }
  ],
  "metadata": {},
  "validation": {
    "status": "passed",
    "validator_version": 1
  }
}
```

The artifact ID is calculated from canonical semantic metadata and file hashes, not timestamps or filesystem paths.

## 4. Storage layout

```text
artifact-store/
  objects/
    sha256/
      ab/
        abcdef.../
          artifact.json
          tensors-00001.safetensors
          data-00001.parquet
  temp/
    <writer-id>/
  leases/
  indexes/
```

Runs reference objects by artifact ID. The store may expose friendly indexes by model, run, or stage, but indexes are rebuildable and never define content identity.

Large tensor artifacts are sharded at useful boundaries. Model artifacts prefer block-aligned shards so a 70B runtime or converter can locate and load blocks without scanning or materializing unrelated tensors.

## 5. Calibration and objective artifacts

`CalibrationStats` records:

- source model and adapter identity;
- dataset selection and tokenizer identity;
- layer identity and shape;
- statistic type and mathematical definition;
- sample/token counts;
- clipping, shrinkage, damping, and accumulation dtype;
- forward/backward/streaming execution strategy;
- numerical summaries and warnings;
- tensors such as input/output norms or covariance representation.

An `ObjectiveArtifact` declares whether it is diagonal, block-diagonal, low-rank-plus-diagonal, or dense and provides a common operator interface. Consumers do not infer the objective type from tensor filenames.

## 6. Layer and block checkpoints

A committed `LayerResult` contains:

- layer plan and all attempt summaries;
- accepted trainable/frozen logical factors;
- source tensor and objective identities;
- deterministic seed identity;
- reconstruction, export, scale-fit, and tuning metrics;
- bit and outlier accounting;
- component versions and elapsed/resource metrics.

A committed `FrozenBlock` contains accepted layer references, required unquantized block tensors, block-level metrics, and the identity of next-block activation streams.

Checkpoint descriptors reference immutable objects. Resume state does not depend on deserializing an arbitrary Python call stack or optimizer object. When an optimizer must resume inside a long tuning unit, its tensors and scalar state use a versioned, safe stage-specific artifact; otherwise the unit restarts deterministically from its last accepted boundary.

## 7. Packed model artifact

```text
packed-model/
  nanoquant-model.json
  model-config.json
  generation-config.json
  tokenizer/
  weights/
    shared.safetensors
    block-000.safetensors
    block-001.safetensors
    ...
    head.safetensors
  layouts/
    cuda-binary-v1/
      block-000.safetensors
      ...
  evaluation-summary.json
  README.md
```

`nanoquant-model.json` records:

- format and minimum runtime version;
- source model/revision/license metadata;
- model adapter/family;
- logical tensor inventory;
- block/layer shape, rank, padding, scales, outliers, bias, and BPW accounting;
- packed layout inventory and backend requirements;
- file hashes and total sizes;
- tokenizer/config identities;
- recipe/run references;
- experiment number and numbered-launcher path/content hash when present;
- validation results;
- optional quality/performance summary.

The runtime can inspect this descriptor and choose a layout before opening large weight shards.

## 8. Tensor names and layouts

Logical tensor names are canonical and independent of source checkpoint spelling. Example:

```text
blocks.12.self_attn.v_proj.factor_left
blocks.12.self_attn.v_proj.factor_right
blocks.12.self_attn.v_proj.scale_pre
blocks.12.self_attn.v_proj.scale_mid
blocks.12.self_attn.v_proj.scale_post
blocks.12.self_attn.v_proj.outlier_indices
blocks.12.self_attn.v_proj.outlier_values
```

Backend-packed names include a declared layout namespace and never masquerade as logical tensors. Padding and original shape are metadata, not encoded only in tensor shape conventions.

Implementation status (2026-07-15): the deployment runtime now defines logical artifact descriptor schema 1 and
logical format `nanoquant-v1`. `nanoquant-model.json` records pinned model identity, runtime compatibility, the
complete quantized-layer specification and tensor inventory, contiguous block indexes, shard paths, sizes, and
SHA-256 hashes. Logical tensor roles are `factor_left`, `factor_right`, `scale_pre`, `scale_mid`, `scale_post`, and
optional `bias`, `outlier_indices`, `outlier_values`, and `outlier_scales`; tensor keys remain canonical dotted
layer names plus those role suffixes. Each block is one safetensors shard. Creation is atomic and refuses overwrite;
inspection bounds descriptor size, rejects path traversal/future schemas, verifies file hashes and safetensors
headers without loading payloads, and loading opens only the shard containing the requested layer. This is the
backend-independent logical artifact, not the still-open packed CUDA layout or complete deployable model export.

## 9. Exact size and BPW accounting

Reports distinguish:

- logical binary factor bits;
- scale, bias, outlier, index, embedding, norm, and head bytes;
- packing padding;
- required runtime metadata;
- container/index overhead;
- optional duplicate backend layouts.

Two BPW values may be useful:

```text
core_bpw      = quantized target-layer logical bits / target source-weight count
artifact_bpw  = complete deployable artifact bytes * 8 / source model parameter count
```

The names and denominator are always shown. Actual serialized bytes are the authority for deployable size.

## 10. Validation

Artifact validation proceeds without executing model code where possible:

- descriptor schema and canonical encoding;
- all referenced files present, sized, and hashed;
- tensor names unique and inventory complete;
- shapes, dtypes, ranks, padding, and alignment consistent;
- bit accounting reconciles with components;
- no non-finite scale/bias values;
- outlier indices sorted/in-range according to format contract;
- backend layout compatible with declared version/capabilities;
- source/config/tokenizer relationships valid;
- optional sampled or complete reference numerical parity.

Validation status and validator version are stored, but consumers may revalidate under a newer validator.

## 11. Compatibility policy

Versions exist at different layers:

- artifact descriptor schema;
- logical NanoQuant representation;
- packed backend layout;
- model adapter mapping;
- component configuration/result schemas.

A descriptor schema change does not automatically imply a new binary layout. The runtime publishes a compatibility table:

| Runtime | Descriptor | Logical format | CUDA layout |
| --- | --- | --- | --- |
| `1.x` | `1` | `nanoquant-v1` | `cuda-binary-v1` |

Readers:

- reject unsupported future major versions;
- ignore documented optional fields only when the schema permits it;
- produce precise migration guidance;
- never guess a layout from missing metadata.

## 12. Migration

Migrations are standalone, pure where possible, and never overwrite the source artifact:

```text
nanoquant migrate-artifact old/ --to logical-v2 --output new/
```

Migration may be:

- metadata-only;
- logical tensor rename/restructure;
- repacking from retained logical state;
- full re-quantization when semantics cannot be preserved.

The resulting descriptor records its parent artifact and migration tool/version. Golden and numerical parity tests cover every supported migration.

## 13. Hub and distribution export

Publishing is an infrastructure operation over an already validated packed artifact. It may create repository-specific metadata and shard indexes, but cannot alter numerical tensors without creating a new artifact identity and revalidation.

Remote loading:

- pins a revision;
- verifies hashes;
- downloads only selected compatible packed layouts where supported;
- does not require remote arbitrary Python code for standard supported model adapters;
- preserves license and source attribution metadata.

## 14. Security and privacy

- No general pickle loading in the deployment path.
- Tensor containers are safetensors or another reviewed non-executable format.
- JSON fields have size/depth limits.
- Paths in descriptors are relative and cannot escape the artifact root.
- Dataset selections and fixtures do not embed credentials.
- Replay fixtures are marked public, internal, restricted, or derived-sensitive.
- Token/prompt samples are excluded from reports by default; hashes and indices provide lineage.

## 15. Garbage collection

Objects are collectible only when unreachable from:

- retained run manifests;
- named baselines/releases;
- pinned fixtures;
- active leases;
- child-artifact lineage required by policy.

Garbage collection first produces a dry-run inventory with sizes and references. Temporary activation streams have short retention unless a diagnostic fixture explicitly pins them.

The measured retention classes, split block/activation format, store-aware root rules, migration protocol, and disk
acceptance bounds are specified in [Artifact Retention and Disk Usage](14-artifact-retention-and-disk-usage.md).
