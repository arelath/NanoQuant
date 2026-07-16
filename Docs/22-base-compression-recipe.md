# Base compression recipe and mandatory GGUF export

All numbered compression experiments derive their numerical configuration from
`src/nanoquant/recipes/base_compression.py`. The visible `BASE_COMPRESSION_CONFIG` replaces the previous implicit
practice of treating legacy Experiment 018 as the base recipe. Experiment 018 now derives from the base like every
other numbered run.

The same module exposes `compression_export_recipe(experiment_number, model_slug)`. It assigns canonical outputs:

```text
outputs/NNN-model-slug/
  logical/
  packed/
  llamacpp-checkpoint/
  model-slug-nanoquant.gguf
  model-slug-nanoquant.gguf.export.json
  model-slug-nanoquant.export-summary.json
```

## Completion contract

Compression experiment workflows call `execute_complete_compression`. A high-level compression experiment is not
complete until all of these stages succeed:

1. the resident compression and optional global tuning have durable complete commits;
2. the complete run passes a fresh transitive artifact validation while streaming into the logical runtime format;
3. logical-to-packed conversion validates every tensor exactly;
4. the pinned modified llama.cpp converter produces a non-empty GGUF and a hash-bound receipt;
5. the GGUF, export summary, receipt, and experiment statistics are hard-linked into `Results/NNN`.

Each stage is resumable. Existing logical, packed, checkpoint, and GGUF outputs are hash-validated and reused. A
partial or provenance-mismatched output fails closed rather than being treated as complete.

## Exporting an older completed run

`execute_compression_export` performs only stages 2–4 and never recompresses the model. This is the supported
backfill path for a resident run that predates the mandatory export contract. After export, use
`tools/publish_results.py` to add its GGUF, export summary, and receipt to the experiment's Results directory.

Experiment 003 v5 was the first backfill through this contract. Its 34-block globally tuned state passed validation,
produced a 1,757,547,456-byte GGUF, and published SHA-256
`96e58f79fadba6c3f684979991f66269f7d1ccb300d32b68b198c0c3ecb9eb38` under `Results/003`.
