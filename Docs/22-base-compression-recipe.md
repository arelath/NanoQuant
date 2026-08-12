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
  model-slug-nanoquant.model-card.md       # validated source for Hub README.md
  model-slug-nanoquant.export-summary.json
  NNN-experiment-name-gguf-quality.json    # when llama.cpp deployment quality is enabled
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
   `token_embd.weight` and, when it exists independently, `output.weight` to Q8_0 by default and verifies both
   material tensor types;
5. when the source snapshot declares a non-empty `vision_config`, the pinned upstream converter exports the vision
   tower and projector as `mmproj-BF16.gguf`, verifies `general.type=mmproj`, `MOSTLY_BF16`, a non-empty tensor
   inventory, and a receipt bound to the source config and converter;
6. workflows with a quality protocol complete the BF16-versus-packed protocol and write its machine-readable JSON;
   recipes with llama.cpp deployment quality enabled then run the exported GGUF through the NanoQuant fork on the
   identical prepared token IDs and target positions, write an identity-bound `gguf-quality.json`, and include both
   comparisons in the rendered Markdown before any external publication;
7. when the export recipe declares a Hugging Face destination, the validated language GGUF, optional mmproj, and
   completed quality artifacts are uploaded in one model-repository commit, and a local token-free receipt records
   its exact commit and file hashes;
8. final GGUFs, export summaries, and export/upload receipts already reside in `Results/NNN`; remaining validated
   experiment statistics are hard-linked there without copying large artifacts.

The embedding and output levels are independent parts of `CompressionExportPolicy` and receipt identity. Set
`CompressionExportPolicy(token_embedding_type="q4_k", output_tensor_type="q6_k")` to override either default;
Q4/Q5/Q6/Q8 llama.cpp variants accepted by the export contract are supported. Models with tied embeddings may omit
the separate `output.weight`; its absence is recorded and accepted. Source `lm_head.weight` and `output.weight`
tensors are both treated as independent output projections and must map to canonical GGUF `output.weight`; Qwen3
therefore follows the same quantization contract as Llama-family models. The second pass uses F16 as its base type
for floating outlier sidecars because llama.cpp's `COPY` mode disables per-tensor overrides. On NanoQuant GGUFs,
that base leaves existing BF16/F16/I32/F32 sidecars alone and changes the token embedding plus the independent
output tensor when present. When salient weights use I8 storage, the pass instead uses a Q8_0 base plus an exact
`.nq_salient_weight=I8` tensor override. llama.cpp only honors that preservation override for a quantized base; the
explicit embedding and output policies still determine those two tensors. Export receipt schema 6 records the
selected base and override list, while validated schema-5 floating-sidecar exports remain reusable.

The mmproj remains independent of NanoQuant language-weight compression and is generated directly from the pinned
Hugging Face vision stack. Text-only snapshots, including Gemma 3 1B, do not produce a placeholder mmproj.

The NanoQuant-specific language converter is vendored at
`tools/llamacpp/convert_nanoquant_to_gguf.py`, with its upstream license and provenance beside it. Portable setup may
copy that hash-pinned file into the pinned upstream llama.cpp conversion toolchain; the NanoQuant llama.cpp fork is
not required to create a GGUF. Upstream `conversion.py`, `convert_hf_to_gguf.py`, `gguf-py`, and the standard
`llama-quantize` executable are still required, so vendoring this converter does not make GGUF export independent of
all llama.cpp tooling. The modified fork remains the reference implementation for llama.cpp NanoQuant inference.
The checkpoint bridge supports the shared canonical projection layout used by Gemma 3, Llama, and dense Qwen 3
model families; the upstream converter selects the final GGUF architecture from the pinned Hugging Face model
configuration.

Each stage is resumable. Existing logical, packed, checkpoint, language GGUF, and mmproj outputs are hash-validated
and reused. A complete pre-convention GGUF under `outputs/NNN` is validated and hard-linked into `Results/NNN` on
the first retry, so the layout transition does not repeat conversion or duplicate model bytes. A partial or
provenance-mismatched output fails closed rather than being treated as complete.
When a source has an independent output head and its otherwise bound GGUF receipt predates the output-tensor
quantization contract, export keeps the old GGUF in place while rebuilding and validating a replacement, then
atomically supersedes the GGUF and receipt. Current-schema tensor mismatches still fail closed as corruption.

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
upload until evaluation succeeds and its document exists. Before upload, the reusable
`tools/render_huggingface_model_card.py` generator creates and validates a separate model card with source-model,
quantized-derivative, task, format, language, and license metadata. Compression-quality cards retain the rendered
quality report as their Markdown body; benchmark cards receive a generated summary body. Both workflows expose the
card as `README.md` and their machine-readable measurements as `quality.json`.
Compression-quality reports distinguish deployable storage from evaluation workspace. Experiments that do not use
llama.cpp may reconstruct or expand packed factors into ordinary PyTorch tensors as a correctness backend; its CUDA
allocator peak is reference-backend workspace rather than deployed NanoQuant VRAM. Experiments 027 through 029 disable
that duplicate candidate pass: Transformers evaluates only the BF16 reference, while the exported NanoQuant GGUF is
evaluated only by llama.cpp. Their primary comparison, quality gate, report, and publication therefore use the actual
deployment artifact rather than a reconstructed dense or factorized candidate.
The llama.cpp quality subprocess runs the GGUF through the pinned custom branch and samples that child process
independently. On Windows, its table reports WDDM dedicated/shared GPU peaks plus the child working-set peak; these
are the packed deployment-runtime measurements and remain separate from PyTorch allocator counters.
The GGUF, optional mmproj, packed quality, optional llama.cpp GGUF quality, and report files therefore share one
commit identity. The GGUF result is reusable only when the GGUF hash, prepared-input hash, scorer binary, llama.cpp
commit and runtime-library identity, GPU-layer policy, and parallelism all still match.

The protocol-matched runner is built from `tools/llamacpp/quality_runner` with
`tools/build_llamacpp_quality.py`. It calls the llama.cpp C API directly: WikiText scores every shifted target after
the per-window BOS token when the tokenizer defines one; for tokenizers such as Qwen3 that intentionally omit BOS,
the first raw token in each window supplies context instead. Both policies score 127 targets in a 128-token window.
Multiple-choice tasks score the same retained context and continuation targets used by
the PyTorch evaluator. It does not reconstruct text or substitute generated answer letters. llama.cpp exposes F32
logits, so the runner records its F64 host log-sum-exp policy explicitly; close-choice results need not be bit-identical
to the packed PyTorch backend. Each parallel batch is validated immediately. If a batch produces an incomplete or
non-finite score, the runner discards all parallel results and restarts the complete benchmark serially so one report
never mixes execution modes. A persistent serial failure reports the exact sequence index, expected and observed
target counts, and accumulated negative log likelihood. Experiment 029 requests serial scoring from the outset
because Qwen3 8B demonstrated repeated non-finite results under multi-sequence NanoQuant CUDA execution.

The same generator can be run independently without contacting the Hub:

```powershell
.\.venv\Scripts\python.exe tools\render_huggingface_model_card.py `
  --base-model owner/source-model --base-revision pinned-revision `
  --model-name published-model --source-snapshot path\to\snapshot `
  --body path\to\quality.md --output path\to\model-card.md
```

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

Publication progress is appended to the compression run's `events.jsonl`, rendered into `run.log`, and shown on the
console. Events cover model-card generation, each artifact's validation, 256 MiB hash-progress checkpoints for large
files, repository access, a 30-second heartbeat during the blocking Hub commit, commit completion/failure, receipt
creation, and export-summary refresh. They include only non-secret repository and artifact metadata; `HF_TOKEN` is
never logged.

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

## Publishing an accepted derived run

An analysis checkpoint can be materialized as an isolated derived run while
retaining the immutable source resident manifest. Promotion does not rewrite
that source identity. Instead, the export recipe must explicitly set both
`publication_experiment_number` and `use_active_global_tuning=True`, and the
terminal workflow loader must opt into relocated-run loading. Current-schema
relocated loading removes only the material `output` path from the manifest
comparison. Historical terminal loading first decodes the persisted canonical
`RunConfig` with current schema defaults and requires it to equal the requested
canonical config, then compares every field represented by the older request
shape. One-shot interruption controls and the WDDM guard are non-semantic after
completion; relocated output and artifact-registry paths may differ when
relocation was explicitly authorized. The journal identity, model identity,
plan, and transitive artifact graph must still match and pass fresh validation.
Active resume never uses this historical compatibility path.

This route is only for a terminal derived run whose active global-tuning
artifact has already been committed through the normal artifact store. It does
not permit an incomplete run to resume under a different path, infer a Results
namespace, or silently include tuning absent from the recipe. Experiment 040 is
the first retained use of this promotion path.
