# Building local imatrix datasets

`tools/build_imatrix_dataset.py` creates llama.cpp activation-importance data for
an LLM and stores it under `Datasets/`. This is a calibration input for subsequent
compression, not a compression experiment or a model quality benchmark.

For the new Gemma 3 text mixture, see [the corpus research and build report](gemma-calibration-corpus.md).
Use `--calibration-text PATH --parse-special` to collect statistics from its local
UTF-8 text. The local bytes and parsing setting are pinned in the request and
receipt; missing special-token parsing is rejected for Gemma-formatted input.

## First dataset: Gemma 3 270M IT

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe tools/build_imatrix_dataset.py
```

Defaults are pinned to:

- Model: `google/gemma-3-270m-it`, revision
  `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`.
- Calibration: `eaddario/imatrix-calibration`, revision
  `e87ed55dcba9d9c3a3e41539f3e728e981b1daa4`,
  [`combined_all_huge.parquet`](https://huggingface.co/datasets/eaddario/imatrix-calibration/blob/e87ed55dcba9d9c3a3e41539f3e728e981b1daa4/combined_all_huge.parquet).
- Destination: `Datasets/gemma-3-270m-it-imatrix-eaddario-huge/`.
- F16 GGUF conversion, 512-token contexts, every complete chunk, GPU offload,
  and no perplexity calculation. Short trailing token remainders are discarded
  by llama.cpp. It inserts/replaces BOS according to its tokenizer policy.

The Parquet has one `content` string containing the entire corpus. Extraction
preserves that string byte-for-byte as UTF-8, including embedded newlines, with
no CSV quoting, dataframe formatting, chat template, sampling or truncation.
For other Parquet sources, nonempty strings are joined by two LF characters;
null/empty/whitespace-only records are skipped and counted. Invalid types fail.

## Prerequisites and options

Use the repository Python environment, with the `imatrix` extra (`pyarrow`) and
the dependencies required by the selected llama.cpp converter installed. Access
to gated model weights must already be configured through Hugging Face's normal
authentication and model access process. Tokens are never passed on the command
line or written into receipts.

Build llama.cpp's `llama-imatrix` target with CUDA for GPU execution. The default
checkout is `D:\dev\research\llama.cpp`; override it with
`NANOQUANT_LLAMA_CPP_ROOT` or `--llama-cpp-root`. The script discovers Windows
Release and Unix build binaries; `--imatrix-exe` overrides that discovery.
The binary must support GGUF imatrix output, `--no-ppl`, and chunk metadata.

```powershell
# Prepare downloads, full calibration text and model GGUF without inference.
.\.venv\Scripts\python.exe tools/build_imatrix_dataset.py --prepare-only

# Another model; arbitrary revisions are resolved to a snapshot and recorded.
.\.venv\Scripts\python.exe tools/build_imatrix_dataset.py `
  --model google/gemma-3-1b-it `
  --revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 `
  --dataset-dir Datasets/gemma-3-1b-it-imatrix-eaddario-huge

# A deliberately limited dataset, with its own destination and recorded limit.
.\.venv\Scripts\python.exe tools/build_imatrix_dataset.py `
  --chunks 100 --dataset-dir Datasets/gemma-3-270m-it-imatrix-eaddario-huge-100
```

`--gpu-layers 0` selects CPU execution. `--model-gguf` accepts an existing
full-precision GGUF that the caller has verified corresponds to the selected
model/revision; its hash is recorded. The script still downloads the HF snapshot
for source provenance. `--local-files-only` prevents Hugging Face downloads.
`--calibration-repo`, `--calibration-revision`, `--calibration-file`,
`--text-column` and `--context-size` customize the calibration protocol.

Before GPU work, inspect existing workers and `nvidia-smi` as described in
`AGENTS.md`. During inference the script holds NanoQuant's `cuda:0` lease. This
coordinates with other NanoQuant workers; external programs do not honor it.

## Output and completion contract

The dataset directory contains:

| File | Purpose |
| --- | --- |
| `imatrix.gguf` | Validated importance sums/counts, usable by llama.cpp or NanoQuant |
| `source.json` | Completion receipt, source revisions/hashes, settings, command and tensor inventory |
| `calibration.txt`, `calibration.json` | Exact extracted corpus and provenance |
| `model-f16.gguf`, `model.json` | Reusable converted model and conversion identity |
| `request.json` | Requested settings; incompatible reuse fails |
| `conversion.log`, `imatrix.log` | Tool output and progress |

`source.json` is published only after the child succeeds, the existing NanoQuant
imatrix loader accepts all sums/counts, and GGUF chunk metadata matches the
expected count logged by llama.cpp. An explicit `--chunks N` must produce
exactly N chunks. `context_tokens_processed` counts full context slots; it is
not a count of unique source tokens or an evaluation metric.

Repeated invocations with the same settings audit hashes and reuse a completed
dataset. Preparation is reusable after `--prepare-only`. A failed/interrupted
inference leaves `imatrix.partial.gguf`, periodically saved every 10 chunks,
and never publishes a completion receipt. Preserve this evidence. Automatic
mid-inference resume is not implemented; use a separate destination for a fresh
run. Do not treat a periodic save as the completed full-corpus dataset.

The hashes include the calibration, model source files, model GGUF and executable;
the llama.cpp checkout revision and converter entry-point hash are also recorded.
The checkout revision alone does not establish that a locally modified binary
was built from that exact checkout; the executable hash identifies what ran.
Model/cache files stay local and are not uploaded. Generated datasets are ignored
by Git under the repository's existing `Datasets/*/` policy.

Inspect a completed artifact or obtain its recipe configuration:

```powershell
.\.venv\Scripts\python.exe tools/inspect_imatrix.py `
  Datasets/gemma-3-270m-it-imatrix-eaddario-huge/imatrix.gguf
```

The `calibration.imatrix` object printed by that command pins the path and SHA-256.
Keep the model-specific directory together when moving the dataset. NanoQuant's
existing `resolve_imatrix` validates mapping and widths against the target model
inventory when the imatrix is consumed.

## Verified first build (2026-11-15)

The default Gemma 3 270M IT run completed locally and its completed-dataset reuse
path passed a fresh hash audit with `--local-files-only`.

| Measurement | Result |
| --- | --- |
| Calibration text | 32,303,694 UTF-8 bytes, extracted from the complete Parquet content |
| Completed contexts | 13,301 × 512 = 6,810,112 token slots |
| Model coverage | All 126 projections across 18 blocks; no unused entries |
| Observations | Exactly 6,810,112 per projection |
| Imatrix size | 471,072 bytes |
| Collection command wall time, including tokenization | 2,698.343 seconds |
| Imatrix SHA-256 | `73bc8cbe4cc5941d224d43ce419a96b9f285fe4c7a126290f8db83d9ff4f2241` |
| Calibration text SHA-256 | `786cb7d7adfe5c0e957b5206e6cf522892bee9922746add7d55c5b7368e1ab2d` |
| Source Parquet SHA-256 | `696861a9d942de9612e69a561f99445fe21576c00e7f1ff8f2019982ac36cf31` |

The local dataset's `source.json` contains complete generation provenance;
`validation.json` records the safetensors-inventory mapping and the uniform
observation-count check. This validates the artifact and its model coverage;
downstream quantization quality was not measured in this build.
