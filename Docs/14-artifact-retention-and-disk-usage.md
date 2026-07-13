# Artifact Retention and Disk Usage

Status: rolling block-result v2 implemented; shared-store and store-aware GC phases remain proposed

Audience: artifact-store, orchestration, resume, replay, and scaling maintainers

Implementation note (2026-07-12): resident runs now write small durable `block-result` v2 objects and separate
`activation-generation` objects. The default `rolling` policy retires the predecessor generation only after the
successor block and journal record commit. V1 readers remain supported, and frozen loading does not require retired
generations. The workspace-level shared store, reader leases, selected pins, migration, and store-aware GC described
below are later rollout phases.

## 1. Problem statement

NanoQuant must preserve enough state to resume expensive work, reproduce accepted results, and audit parity without
turning every transient tensor into permanent evidence. The current resident implementation commits both full
boundary activation streams inside every immutable `block-result`. That makes correctness robust, but it couples a
small durable block result to approximately 2.4 GB of data whose useful resume lifetime normally ends at the next
block commit.

The same audit also found multiple standalone content-addressed stores containing identical artifact IDs. A content
hash deduplicates objects only within one store; it does not prevent the same bytes from being copied into several
run-local stores.

The design therefore separates these concerns:

1. semantic results that must remain durable;
2. rolling state required only to resume the next unit of work;
3. explicitly pinned replay or diagnostic fixtures;
4. scratch and rejected-attempt payloads that may expire;
5. evidence documents, which garbage collection never deletes.

## 2. Measured baseline

The pinned `google/gemma-3-1b-it` audit used 256 sequences of 2,048 tokens and hidden width 1,152. The evidence tree
contained 158.8 GB of logical files while the tuned run had completed 11 of 26 blocks.

| Storage class | Measured logical size | Finding |
| --- | ---: | --- |
| Shared full-Fisher artifact store | 108.1 GB | Contains the complete untuned run and partial tuned run |
| Five earlier standalone Gemma stores | 50.7 GB | Superseded experiments retained separately |
| 37 `block-result` artifacts | 89.4 GB | 26 untuned plus 11 tuned block checkpoints |
| Unreferenced source/outlier stage artifacts | 6.1 GB | 522 objects; safe only after lease/age checks |
| Cross-store duplicate content-addressed objects | 26.5 GB | 1,367 IDs occur in more than one physical store |

Each block result contains two BF16 tensors:

```text
256 samples * 2048 tokens * 1152 hidden * 2 bytes = 1,207,959,552 bytes per stream
```

The teacher and compressed streams therefore require about 2.416 GB per committed block including container
overhead. Retaining all 26 boundaries costs about 62.8 GB per run. Retaining one rolling pair costs about 2.4 GB.

The tuned-run `artifacts` paths are NTFS junctions to the full-Fisher store and do not consume a second physical
copy. Hashing 18.6 GB of non-activation payloads in the shared store found no exact payload duplicates. Teacher
streams from the untuned and tuned runs are not identical and cannot be content-deduplicated. The primary savings
must come from retention policy and cross-store topology, not compression or optimistic hash matching.

## 3. Goals

- Preserve atomic layer/block commits and deterministic resume from the latest valid boundary.
- Keep frozen model assembly, evaluation, reporting, and accepted-result audit independent of transient activations.
- Make replay retention explicit instead of retaining every boundary accidentally.
- Bound normal activation evidence to a constant number of generations rather than the model block count.
- Deduplicate equal objects across runs on the same workspace and filesystem.
- Preserve old artifact readers and journals; migrations never overwrite source evidence.
- Make garbage-collection roots store-aware, typed, auditable, and safe in the presence of active writers.
- Estimate durable, rolling, scratch, and worst-case disk separately before a run starts.

## 4. Non-goals

- This design does not weaken content hashing or make committed objects mutable.
- It does not delete journals, reports, manifests, logs, evaluation results, or user-authored evidence documents.
- It does not require lossy activation compression. Activation dtype or compression changes are separate numerical
  experiments with their own parity evidence.
- It does not make every historical run replayable forever. Replayability is a named retention policy with a visible
  storage cost.
- It does not garbage-collect the Hugging Face source snapshot or external dataset caches.

## 5. Retention classes

Every artifact descriptor declares one retention class. Retention class affects root policy, never content identity
or validation.

| Class | Examples | Default lifetime |
| --- | --- | --- |
| `durable_result` | plan, accepted layer result, frozen block, evaluation, tuned model state | While a retained run/release references it |
| `resume_state` | next-block teacher/compressed activations, in-epoch optimizer state | Latest committed generation plus a grace period |
| `pinned_fixture` | selected difficult layer/block captures, golden replay tensors | Until the explicit pin is removed |
| `diagnostic` | rejected attempt tensors, optional detailed captures | Configured TTL or explicit pin |
| `scratch` | materialized source slices, temporary concatenations, writer staging | Lease lifetime only |

Metrics and provenance for rejected attempts remain durable as small JSON summaries even when their tensor payloads
expire. A source layer normally references the pinned model revision, canonical tensor name, shape, dtype, and
content hash instead of copying the source tensor. `capture-layer` and `capture-block` create `pinned_fixture`
objects when self-contained source tensors are required.

## 6. Block result version 2

`block-result-v2` is small and durable. It contains or references:

- block identity and semantic input hashes;
- accepted `LayerResult` references;
- frozen block state or its immutable reference;
- block loss snapshots, bit accounting, timing, and resource metrics;
- teacher and compressed boundary generation references;
- the replay/retention policy applied at commit time.

The full boundary tensors move into separate `activation-generation-v1` objects:

```json
{
  "artifact_type": "activation-generation",
  "schema_version": 1,
  "retention_class": "resume_state",
  "stream": "compressed",
  "boundary_after_block": 10,
  "shape": [256, 2048, 1152],
  "dtype": "bfloat16",
  "sample_selection": "sha256-...",
  "parent_generation": "sha256-...",
  "producer_block_result": "sha256-...",
  "files": ["activations.safetensors"]
}
```

Teacher and compressed streams are separate objects because their consumers and final lifetimes differ. Large
objects may be block/sample sharded, but a generation descriptor commits the complete ordered shard inventory
atomically.

The durable block result may continue to reference an expired activation generation. Such a reference is recorded as
lineage, not as a strong reachability edge. Typed references therefore declare their role:

```text
strong     required to load or validate the durable result
resume     required only while it is the active resume boundary
lineage    identity/provenance; payload may be absent under policy
```

Validators report an expired lineage payload as `not_retained`, not corruption. Missing strong or currently rooted
resume payloads remain corruption.

## 7. Commit and resume protocol

For block `N`:

1. retain the predecessor boundary generations under the active run lease;
2. commit all accepted layer results and frozen block state;
3. produce and validate teacher/compressed generations for boundary `N`;
4. commit `block-result-v2` with strong frozen-state references and resume-generation references;
5. append the block journal record;
6. atomically advance the run's `active-resume-boundary` pointer;
7. release the predecessor generation from the active root set;
8. collect the predecessor only after no reader lease exists and the configured grace period expires.

A crash before step 5 leaves the old boundary rooted. A crash between steps 5 and 6 is repaired by deriving the
expected pointer from the validated journal. A crash after step 6 resumes from the new generation. No ordering may
leave the journal pointing at a generation eligible for collection.

Layer-level resume within a block continues to use the predecessor block-entry activations. Those generations stay
rooted until the block commit is durable. Deterministic replay of already accepted layers remains unchanged.

At the final block, the teacher stream normally expires immediately after final metrics are committed. The final
compressed stream may remain rooted through suffix execution/evaluation and then expire unless explicitly pinned.
Frozen model loading never depends on either stream.

## 8. Replay policies

The run recipe declares one policy before execution:

| Policy | Retained activation generations | Intended use |
| --- | --- | --- |
| `rolling` | Current resume boundary only | Normal full runs; default |
| `final` | Current boundary plus final compressed generation | Evaluation handoff |
| `selected` | Rolling boundary plus configured block IDs | Regression fixtures and difficult layers |
| `all` | Every teacher/compressed boundary | Short diagnostics only |

`all` requires an explicit disk estimate and acknowledgement when planned activation bytes exceed the configured
diagnostic budget. Promotion evidence should use `selected` fixtures rather than making `all` the implicit default.

Replay commands fail clearly when an unpinned activation payload has expired and offer deterministic regeneration
from the nearest retained predecessor when the required source/frozen state is available. Reports distinguish direct
replay from regenerated replay.

## 9. Store topology and cross-run deduplication

New runs in one workspace use a workspace-level content-addressed store:

```text
evidence/
  artifacts/
    store.json
    objects/...
    leases/...
    pins/...
  m4/<run>/
    manifest.json
    state/journal.jsonl
    events.jsonl
    report.json
```

Each store has an immutable `store_id`. A typed artifact reference contains both `store_id` and `artifact_id`.
Friendly run-local paths may be junctions, but the manifest is authoritative. Regex discovery of a bare hash in an
unrelated evidence file is not sufficient to root an object.

When importing an object from another local store, the importer:

1. validates descriptor and payload hashes;
2. reuses the object when it already exists in the destination;
3. otherwise attempts a same-volume hard link or filesystem clone for immutable payload files;
4. falls back to a verified copy;
5. records import provenance without changing semantic identity.

This removes the measured 26.5 GB of repeated physical objects across the audited standalone stores while retaining
their replayability.

## 10. Store-aware garbage collection

Garbage collection is mark-and-sweep over typed references. Its roots are:

- retained run manifests whose `store_id` matches the target store;
- active resume-boundary pointers;
- named release/baseline pins;
- explicit replay fixture pins;
- active writer and reader leases;
- migration/import transactions that have not committed or expired.

Evidence files remain untouched even when their referenced payloads are deliberately retired. The evidence records
the retention decision, time, policy, tool version, and deleted logical/allocated bytes.

The apply protocol is:

1. acquire the store GC lease;
2. snapshot the inventory generation and root-set generation;
3. mark strong and active-resume reachability transitively;
4. apply TTL/minimum-age rules to otherwise unreachable objects;
5. emit a dry-run plan with object IDs, kinds, owners, logical bytes, and allocated bytes;
6. on `--apply`, reject a stale plan if either generation changed;
7. rename candidates atomically into store-local quarantine;
8. update validation/index caches;
9. delete quarantined objects after the recovery window.

GC never follows a junction into a second store and never assumes that the same artifact ID in two independent
stores gives one store ownership of the other store's references. The current conservative text scanner remains a
legacy fallback only; scoped manifests supersede it.

## 11. Disk planning

The resource plan reports four independent quantities:

```text
durable_result_bytes
peak_resume_state_bytes
peak_scratch_bytes
optional_pinned_fixture_bytes
```

It also reports `worst_case_before_gc_bytes`, because an interrupted process or grace period can temporarily retain
both predecessor and successor generations. For rolling activation storage:

```text
activation_generation_bytes = samples * sequence_length * hidden_width * dtype_bytes
steady_state_resume_bytes    = 2 streams * activation_generation_bytes
commit_peak_resume_bytes     = 4 streams * activation_generation_bytes
```

For the pinned Gemma workload, steady state is approximately 2.416 GB and commit peak is approximately 4.832 GB,
independent of 26-block depth. Disk admission includes a safety margin and refuses to start when the store cannot
accommodate commit peak, scratch, quarantine, and the durable result estimate.

Logical and allocated bytes are reported separately so sparse files, hard links, filesystem compression, and
junctions do not produce misleading totals.

## 12. Backward compatibility and migration

Readers support both formats:

- `block-result-v1`: embedded `teacher-activations.safetensors` and
  `compressed-activations.safetensors` remain strong files inside the immutable object;
- `block-result-v2`: frozen result plus typed external generation references.

Existing v1 objects are never edited or partially pruned because doing so would invalidate their content identity.
A migration creates new v2 objects and records the v1 parent. Where supported, activation payloads are hard-linked or
cloned into generation objects; otherwise migration copies and verifies them. After all retained manifests point to
v2, the old v1 objects become ordinary GC candidates.

Retiring a standalone legacy store is a separate explicit operation. It may preserve journals/reports while removing
all payloads, but the resulting run is marked `metadata_only` and is no longer claimed to be directly replayable.

## 13. Rollout plan

1. Add store identity and typed strong/resume/lineage references without changing v1 readers.
2. Add `activation-generation-v1` and `block-result-v2` writers behind a recipe version.
3. Add the atomic active-resume-boundary pointer and crash-injection coverage.
4. Make the workspace-level shared store the default for new runs.
5. Replace global hash-regex rooting with manifest/store-aware GC; retain the scanner for legacy evidence.

The implemented `tools/cleanup_run_activations.py` closes the reused-run-directory case that ordinary reachability
GC cannot reclaim: historical block records intentionally root their old activation generations. The command keeps
the latest block generation for the active (by default, latest) config identity and retires only activation-generation
artifacts referenced by superseded identities or older active blocks. It is dry-run by default and leaves journals,
block/layer results, frozen tensors, metrics, logs, and other evidence files untouched.

`tools/validate_resident_run.py` provides the corresponding pre-consumption audit. It validates the journal as a
strict, single-identity sequence; checks every layer and block envelope against its journal position; follows every
transitive typed artifact reference; and re-hashes descriptor members without reading or updating the persistent
validation cache. Missing predecessor `activation-generation` objects are reported as intentionally retired under
rolling retention, while a missing latest generation or any missing durable factor/result artifact fails the audit.
`--require-complete` additionally requires a contiguous zero-based prefix equal to `--expected-blocks` (26 by
default). Run this before global KD, final evaluation, or cleanup of a release candidate.
6. Add v1-to-v2 migration and verified hard-link/clone import.
7. Migrate or retire older Gemma stores only after dry-run reports and explicit retention decisions.
8. Enforce rolling disk estimates in resident and streaming execution plans.

No step requires rewriting the currently running Gemma evidence. Its v1 artifacts remain valid inputs to the
backward-compatible reader and migration tests.

## 14. Acceptance criteria

### Correctness and recovery

- Uninterrupted and interrupted/resumed tiny runs remain bit-exact under rolling retention.
- Crash injection before and after generation, block, journal, and pointer commits always discovers a valid boundary.
- A reader lease prevents collection; expired leases are recoverable without retaining objects forever.
- Frozen model assembly and evaluation succeed after all non-pinned activation generations are collected.
- Selected replay fixtures remain loadable; an expired unpinned fixture produces a precise error/regeneration plan.
- v1 and v2 frozen models produce identical outputs for the same accepted state.

### Store and GC safety

- Two stores containing the same artifact ID do not root or delete one another's objects.
- Shared-store import reuses an existing object and does not allocate a second physical payload.
- Dry-run performs no writes; apply rejects stale inventory/root generations.
- GC never deletes evidence documents, active resume state, releases, baselines, or explicit fixture pins.
- Corrupt, missing, junction-escaped, or noncanonical candidates are rejected before deletion.

### Measured disk bounds

- A 26-block pinned Gemma run using `rolling` retains no more than one teacher/compressed resume pair in steady state.
- Activation commit peak stays within two generations per stream and returns to steady state after grace/GC.
- Retained activation bytes are independent of completed block count.
- The planner's durable, peak-resume, and scratch estimates are compared with observed logical and allocated bytes;
  unexplained error beyond the configured tolerance fails the scaling gate.
- A migrated shared-store replay preserves required old-run artifacts while eliminating physical duplicate copies.

## 15. Operational guidance before implementation

- Do not manually remove files from a committed v1 artifact; delete only whole validated objects through GC.
- Do not apply GC to the shared full-Fisher store while the tuned Gemma writer is active.
- Use the existing cleanup command in dry-run mode and retain the 24-hour guard for current runs.
- Retiring old standalone stores requires a named decision about whether each run stays replayable or becomes
  metadata-only.
- Preserve at least the pinned source model, calibration selection, accepted frozen state, evaluation result, and a
  small selected replay set until the parity gate is approved.
