# Frozen Legacy Support Inventory

This inventory describes behavior present at the Milestone 0 capture, not an implicit promise that architecture guessing
will continue. New support is adapter-, dataset-, evaluator-, format-, and backend-version specific.

## Model families and variants

| Legacy identifier | Frozen status | Rewrite disposition |
| --- | --- | --- |
| `llama` | Factorized seven projection decoder block | Llama-compatible adapter, productized |
| `mistral`, `mixtral`, `mobilellm` | Routed through the Llama-shaped branches | Explicit variant validation; no automatic compatibility claim |
| `qwen3` | Factorized seven projection decoder block | Qwen adapter, productized after contract suite |
| `gemma*` | Gemma and Gemma 3 text-stack handling | Gemma/Gemma 3 adapter, productized after contract suite |
| `opt` | Attention plus `fc1`/`fc2` factorization | OPT adapter retained |
| `gpt2` | Decoder traversal exists, but factorized layer list does not | Unsupported with explicit diagnostic |

Checkpoint sources are Hugging Face/Transformers directories or Hub snapshots, including sharded safetensors and legacy
PyTorch weights. The rewrite product path requires pinned revisions and safetensors metadata/hash verification; remote
model code remains off by default.

## Data and evaluation

- Calibration sources: Salesforce Wikitext-2 raw, AllenAI C4 English, and HuggingFaceH4 UltraChat 200k `train_sft`.
- Legacy perplexity sources: Wikitext-2 and C4.
- Legacy task defaults: BoolQ, PIQA, HellaSwag, WinoGrande, ARC Easy, and ARC Challenge through `lm_eval`.
- The frozen Experiment 019 mixture is UltraChat plus Wikitext-2. The legacy string did not record mixture weights or
  dataset revisions, so the rewrite must pin those before claiming dataset identity parity.

## Artifacts and packing

- Pickled `.pt` compressed state dictionaries with binary factors, scales, salient outliers, and row-wise int8 embedding
  storage.
- Custom PyTorch extension layouts for binary GEMV and Marlin-style GEMM/fused packing.
- Modified llama.cpp NanoQuant GGUF representation produced by `convert_nanoquant_to_gguf.py`.
- Markdown/CSV reconstruction reports and rank-utility CSV are reporting evidence, not model artifact formats.

Compatibility follows ADR 0008. No executable legacy artifact is a native rewrite artifact.

## CUDA architectures

The legacy builder specializes for the active GPU when available. Its no-GPU fallback explicitly emits `sm_80`, `sm_89`,
and `sm_90`. The captured development GPU is compute capability 8.9. These are build targets, not parity evidence; each
optimized backend still needs the Milestone 6 shape/dtype/rank/outlier matrix on every declared architecture.

## Configuration

The frozen `NanoQuantConfigDataclass` has 95 fields. `nanoquant.config.migration.migration_inventory()` gives every field
exactly one mapped or explicitly removed disposition, and `migrate_legacy()` rejects any field outside that inventory.
The total-migration and unknown-field behavior are enforced by `tests/unit/test_config.py`.

