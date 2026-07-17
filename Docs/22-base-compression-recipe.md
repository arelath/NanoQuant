# Base compression recipe and mandatory GGUF export

Numbered compression experiments start from an unnumbered reusable template in
`experiments/recipes/base_compression.py`. The package exposes `BASE_COMPRESSION_TEMPLATE`,
`GEMMA_3_1B_PARITY_TEMPLATE`, `GEMMA_3_4B_COMPRESSION_TEMPLATE`, and
`LARGE_MODEL_COMPRESSION_TEMPLATE`. Concrete identities and experiment-specific deltas live in the numbered
launchers in `experiments/`, not in `recipes`.

The base allocation promotes every `self_attn.v_proj` and `self_attn.k_proj` layer to its physical maximum rank and
adds 25% to each `self_attn.q_proj` packed-factor budget. Promotions happen after ordinary sensitivity allocation,
so other layers retain their target-BPW ranks and reported physical BPW includes the additional projection storage.
Numbered experiments whose results or resumable state predate these decisions explicitly pin empty overrides;
newly authored compression experiments inherit both policies.

Every recipe definition uses `config_delta(parent, ...)` at each nested dataclass boundary. The shared compression
recipe is itself a delta from the canonical schema defaults, standalone benchmark recipes use the same schema
baseline, and derived experiments inherit from their direct recipe parent. The helper rejects an explicit value
equal to its parent during module import, so recipe files state only material differences while their fully resolved
`RunConfig` remains complete and hash-stable.

The generic experiment builder derives export locations from `ExperimentIdentity`. For Experiment `NNN`, it assigns:

```text
outputs/NNN/
  logical/
  packed/
  llamacpp-checkpoint/
  model-slug-nanoquant.gguf
  model-slug-nanoquant.gguf.export.json
  model-slug-nanoquant.gguf.huggingface.json  # only when a Hub upload is configured
  mmproj-BF16.gguf                 # multimodal snapshots only
  mmproj-BF16.gguf.export.json     # multimodal snapshots only
  NNN-canonical-name-summary.json
```

## Completion contract

Compression experiment workflows call `execute_complete_compression`. A high-level compression experiment is not
complete until all of these stages succeed:

1. the resident compression and optional global tuning have durable complete commits;
2. the complete run passes a fresh transitive artifact validation while streaming into the logical runtime format;
3. logical-to-packed conversion validates every tensor exactly;
4. the pinned modified llama.cpp converter produces a non-empty GGUF shell, then `llama-quantize` quantizes
   `token_embd.weight` to Q8_0 by default and verifies the material tensor type;
5. when the source snapshot declares a non-empty `vision_config`, the pinned upstream converter exports the vision
   tower and projector as `mmproj-BF16.gguf`, verifies `general.type=mmproj`, `MOSTLY_BF16`, a non-empty tensor
   inventory, and a receipt bound to the source config and converter;
6. when the export recipe declares a Hugging Face destination, the validated language GGUF and optional mmproj are
   uploaded in one model-repository commit and a local token-free receipt records its exact commit and file hashes;
7. the GGUF files, export summaries, receipts, and experiment statistics are hard-linked into `Results/NNN`.

The embedding level is part of `CompressionExportPolicy` and receipt identity. Set
`CompressionExportPolicy(token_embedding_type="q4_k")` for a Q4_K embedding; Q4/Q5/Q6/Q8 llama.cpp variants
accepted by the export contract are supported. The second pass uses F16 as its base type because llama.cpp's `COPY`
mode disables per-tensor overrides. On NanoQuant GGUFs, the F16 base leaves the existing BF16/F16/I32/F32 sidecars alone
and changes only the BF16 token embedding.

The mmproj remains independent of NanoQuant language-weight compression and is generated directly from the pinned
Hugging Face vision stack. Text-only snapshots, including Gemma 3 1B, do not produce a placeholder mmproj.

Each stage is resumable. Existing logical, packed, checkpoint, language GGUF, and mmproj outputs are hash-validated
and reused. A partial or provenance-mismatched output fails closed rather than being treated as complete.

## Optional Hugging Face upload

Hugging Face publication is an explicit experiment-recipe choice. A newly authored compression experiment can add
the destination to its export declaration:

```python
from recipes import CompressionExportPolicy, HuggingFaceUploadConfig

export = CompressionExportPolicy(
    release_name="gemma-3-1b-it",
    huggingface=HuggingFaceUploadConfig(
        "owner/gemma-3-1b-it-nanoquant-GGUF",
        private=True,
        commit_message="Publish NanoQuant Experiment 008",
    ),
)
```

Do not put a token in the recipe. The uploader uses the standard cached Hugging Face login or `HF_TOKEN`; environment
capture continues to exclude that secret. Before making any Hub request, it opens each exported model file, verifies
the byte count and SHA-256 from the GGUF export result, rewinds the same open handle, and gives that handle to the Hub
client. The language GGUF and optional mmproj therefore enter one commit without a save or conversion step that could
change numerical content.

On success, `<model>.gguf.huggingface.json` records the canonical repository ID and URL, commit OID and URL, requested
visibility, commit message, and each uploaded filename, byte count, and SHA-256. High-level compression experiments
also publish this receipt under `Results/NNN` and include it in their schema-2 summary. Upload failures propagate, but
the completed local compression and validated exports remain reusable; rerunning retries publication without
recompressing. Experiments whose export policy omits `huggingface` remain fully offline.

## Exporting an older completed run

`execute_compression_export` performs only stages 2–6 and never recompresses the model. This is the supported
backfill path for a resident run that predates the mandatory export contract. After export, use
`tools/publish_results.py` to add its GGUF files, export summary, and receipts to the experiment's Results directory.

Experiment 003 v5 was the first backfill through this contract. Its 34-block globally tuned state passed validation.
The initial export incorrectly retained `token_embd.weight` as BF16; it was superseded by the verified Q8_0 export
recorded in the current receipt and `Results/003` publication.

Experiment 003's Gemma 3 4B snapshot also exports and publishes the paired vision artifact:

- path: `Results/003/mmproj-BF16.gguf`;
- bytes: 851,251,776;
- tensor count: 439;
- material tensor types: BF16, F16, and F32 (the latter two are converter-required exceptions);
- SHA-256: `78a2097ec69ed696a6463201fd1333b0f0086836c869bbaf0b4511680b1787b5`.
