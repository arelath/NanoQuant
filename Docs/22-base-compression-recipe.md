# Base compression recipe and mandatory GGUF export

Numbered compression experiments start from an unnumbered reusable template in
`experiments/recipes/base_compression.py`. The package exposes `BASE_COMPRESSION_TEMPLATE`,
`GEMMA_3_270M_COMPRESSION_TEMPLATE`, `GEMMA_3_4B_COMPRESSION_TEMPLATE`, and
`LARGE_MODEL_COMPRESSION_TEMPLATE`. Concrete identities and experiment-specific deltas live in the numbered
launchers in `experiments/`, not in `recipes`.

The base allocation promotes every `self_attn.v_proj` and `self_attn.k_proj` layer to its physical maximum rank and
adds 25% to each `self_attn.q_proj` packed-factor budget. Promotions happen after ordinary sensitivity allocation,
so other layers retain their target-BPW ranks and reported physical BPW includes the additional projection storage.
All compression templates inherit both policies; there is no compatibility template that clears them for old runs.

Every recipe definition uses `config_delta(parent, ...)` at each nested dataclass boundary. The shared compression
recipe is itself a delta from the canonical schema defaults, standalone benchmark recipes use the same schema
baseline, and derived experiments inherit from their direct recipe parent. The helper rejects an explicit value
equal to its parent during module import, so recipe files state only material differences while their fully resolved
`RunConfig` remains complete and hash-stable.

The generic experiment builder derives export locations from `ExperimentIdentity`. Intermediate runtime artifacts
remain rebuildable under `outputs/NNN`, while final deployment files are created directly in `Results/NNN`:

```text
outputs/NNN/
  logical/
  packed/
  llamacpp-checkpoint/
  NNN-canonical-name-summary.json

Results/NNN/
  model-slug-nanoquant.gguf
  model-slug-nanoquant.gguf.export.json
  model-slug-nanoquant.export-summary.json
  model-slug-nanoquant.gguf.huggingface.json  # only when a Hub upload is configured
  mmproj-BF16.gguf                 # multimodal snapshots only
  mmproj-BF16.gguf.export.json     # multimodal snapshots only
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
6. workflows with a quality protocol complete that protocol and write its machine-readable JSON and rendered
   Markdown before any external publication;
7. when the export recipe declares a Hugging Face destination, the validated language GGUF, optional mmproj, and
   completed quality artifacts are uploaded in one model-repository commit, and a local token-free receipt records
   its exact commit and file hashes;
8. final GGUFs, export summaries, and export/upload receipts already reside in `Results/NNN`; remaining validated
   experiment statistics are hard-linked there without copying large artifacts.

The embedding level is part of `CompressionExportPolicy` and receipt identity. Set
`CompressionExportPolicy(token_embedding_type="q4_k")` for a Q4_K embedding; Q4/Q5/Q6/Q8 llama.cpp variants
accepted by the export contract are supported. The second pass uses F16 as its base type because llama.cpp's `COPY`
mode disables per-tensor overrides. On NanoQuant GGUFs, the F16 base leaves the existing BF16/F16/I32/F32 sidecars alone
and changes only the BF16 token embedding.

The mmproj remains independent of NanoQuant language-weight compression and is generated directly from the pinned
Hugging Face vision stack. Text-only snapshots, including Gemma 3 1B, do not produce a placeholder mmproj.

The NanoQuant-specific language converter is vendored at
`tools/llamacpp/convert_nanoquant_to_gguf.py`, with its upstream license and provenance beside it. Portable setup may
copy that hash-pinned file into the pinned upstream llama.cpp conversion toolchain; the NanoQuant llama.cpp fork is
not required to create a GGUF. Upstream `conversion.py`, `convert_hf_to_gguf.py`, `gguf-py`, and the standard
`llama-quantize` executable are still required, so vendoring this converter does not make GGUF export independent of
all llama.cpp tooling. The modified fork remains the reference implementation for llama.cpp NanoQuant inference.
The checkpoint bridge supports the shared canonical projection layout used by Gemma 3 and Llama model families;
the upstream converter selects the final GGUF architecture from the pinned Hugging Face model configuration.

Each stage is resumable. Existing logical, packed, checkpoint, language GGUF, and mmproj outputs are hash-validated
and reused. A complete pre-convention GGUF under `outputs/NNN` is validated and hard-linked into `Results/NNN` on
the first retry, so the layout transition does not repeat conversion or duplicate model bytes. A partial or
provenance-mismatched output fails closed rather than being treated as complete.

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

The low-level compression export never contacts Hugging Face. Quality and benchmark workflows defer the configured
upload until evaluation succeeds and its document exists. Compression-quality commits expose the rendered report as
`README.md` and its machine-readable measurements as `quality.json`; benchmark workflows also provide their
machine-readable measurements as `quality.json`.
The GGUF, optional mmproj, and quality files therefore share one commit identity.

Do not put a token in the recipe. The shared resident launcher loads the repository-root `.env` with override
semantics before resolving Hugging Face inputs, so a corrected local `HF_TOKEN` takes precedence over an inherited
token; the uploader also supports the standard cached Hugging Face login. Environment capture continues to exclude
that secret. Before making any Hub request, it opens every model and quality file, verifies its byte count and
SHA-256, rewinds the same open handle, and gives that handle to the Hub client. No save or conversion step can change
the validated content between evaluation and upload.

On success, `<model>.gguf.huggingface.json` records the canonical repository ID and URL, commit OID and URL, requested
visibility, commit message, and each uploaded filename, byte count, and SHA-256. High-level compression experiments
also publish this receipt under `Results/NNN` and include it in their schema-2 summary. Upload failures propagate, but
the completed local compression and validated exports remain reusable; rerunning retries publication without
recompressing. Experiments whose export policy omits `huggingface` do not publish; source-model and evaluation
resolution may still contact the Hub when a pinned local file is missing.

## Exporting an older completed run

`execute_compression_export` performs only stages 2–5, never recompresses the model, and never contacts Hugging Face.
This is the supported local backfill path for a resident run that predates the mandatory export contract. Its GGUF
and receipt are written directly to the experiment's Results directory; complete the quality workflow before Hub
publication, and use `tools/publish_results.py` to add remaining summaries and statistics.

Experiment 003 v5 was the first backfill through this contract. Its 34-block globally tuned state passed validation.
The initial export incorrectly retained `token_embd.weight` as BF16; it was superseded by the verified Q8_0 export
recorded in the current receipt and `Results/003` publication.

Experiment 003's Gemma 3 4B snapshot also exports and publishes the paired vision artifact:

- path: `Results/003/mmproj-BF16.gguf`;
- bytes: 851,251,776;
- tensor count: 439;
- material tensor types: BF16, F16, and F32 (the latter two are converter-required exceptions);
- SHA-256: `78a2097ec69ed696a6463201fd1333b0f0086836c869bbaf0b4511680b1787b5`.
