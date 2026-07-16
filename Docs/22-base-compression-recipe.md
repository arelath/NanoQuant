# Base compression recipe and mandatory GGUF export

All numbered compression experiments derive their numerical configuration from
`src/nanoquant/recipes/base_compression.py`. The visible `BASE_COMPRESSION_CONFIG` replaces the previous implicit
practice of treating legacy Experiment 018 as the base recipe. Experiment 018 now derives from the base like every
other numbered run.

The base allocation promotes every `self_attn.v_proj` layer to its physical maximum rank. The promotion happens
after the ordinary sensitivity allocation, so other layers retain their target-BPW ranks and the reported physical
BPW includes the additional `v_proj` storage. Numbered experiments whose results or resumable state predate this
decision explicitly pin an empty override; newly authored compression experiments inherit the maximum-rank policy.

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
4. the pinned modified llama.cpp converter produces a non-empty GGUF shell, then `llama-quantize` quantizes
   `token_embd.weight` to Q8_0 by default and verifies the material tensor type;
5. the GGUF, export summary, receipt, and experiment statistics are hard-linked into `Results/NNN`.

The embedding level is part of the export recipe and receipt identity. Use
`compression_export_recipe(..., token_embedding_type="q4_k")` for a Q4_K embedding; Q4/Q5/Q6/Q8 llama.cpp
variants accepted by the recipe are supported. The second pass uses F16 as its base type because llama.cpp's `COPY`
mode disables per-tensor overrides. On NanoQuant GGUFs, the F16 base leaves the existing BF16/F16/I32/F32 sidecars alone
and changes only the BF16 token embedding.

Each stage is resumable. Existing logical, packed, checkpoint, and GGUF outputs are hash-validated and reused. A
partial or provenance-mismatched output fails closed rather than being treated as complete.

## Exporting an older completed run

`execute_compression_export` performs only stages 2–4 and never recompresses the model. This is the supported
backfill path for a resident run that predates the mandatory export contract. After export, use
`tools/publish_results.py` to add its GGUF, export summary, and receipt to the experiment's Results directory.

Experiment 003 v5 was the first backfill through this contract. Its 34-block globally tuned state passed validation.
The initial export incorrectly retained `token_embd.weight` as BF16; it was superseded by the verified Q8_0 export
recorded in the current receipt and `Results/003` publication.
