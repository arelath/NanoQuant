# Downloaded Gemma 3 1B IQ2_M quality profile

Date: 2026-09-13. Experiment 069 is an evaluation-only comparison of a published
Hugging Face quantization with the retained Experiment 060 best-methods model.
No quantization, compression, fitting, or tuning is performed.

The published IQ2_M file improves six-task average accuracy by **7.50 percentage
points** and lowers perplexity by **3.50%** relative to the retained 060
best-methods result, while using **34.13% more deployment storage**.

| Metric | BF16 retained | NanoQuant 060 retained | Published IQ2_M |
| --- | ---: | ---: | ---: |
| WikiText-2 perplexity, lower is better | 96.459609 | 115.668625 | 111.619040 |
| PIQA normalized accuracy | 73.0% | 65.9% | 72.9% |
| ARC-Easy normalized accuracy | 63.8% | 46.1% | 58.4% |
| ARC-Challenge normalized accuracy | 37.6% | 24.1% | 36.1% |
| HellaSwag normalized accuracy | 52.8% | 44.7% | 49.6% |
| Winogrande accuracy | 59.4% | 53.6% | 59.2% |
| BoolQ accuracy | 76.8% | 66.9% | 70.1% |
| Six-task mean | 60.57% | 50.22% | 57.72% |

**Execution qualification:** 060 uses its completed PyTorch factorized quality
reference; IQ2_M is freshly scored in llama.cpp. Inputs and protocol match, but
execution backends differ. The separately included completed 048 GGUF result
uses the same runtime binaries and exact scorer inputs as IQ2_M: perplexity
153.709446 and six-task average accuracy 46.12%, at 417,340,544 bytes. It is the
strongest completed retained NanoQuant GGUF benchmark found, but weaker than
the newer 060 reference. These results do not prove equal-budget superiority.

## Inputs and storage

The external model is
[`bartowski/google_gemma-3-1b-it-GGUF`](https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/tree/116f76234503685a98f572982177b11d44ec8ff1),
file `google_gemma-3-1b-it-IQ2_M.gguf`, revision
`116f76234503685a98f572982177b11d44ec8ff1`. The completed download was verified
against Hugging Face's published size and SHA-256 before scoring:

```text
669,783,808 bytes
553ab5fff3b64918f5c805769e7f06b7898ff33ba0dd7eda076df47ad25fca36
```

The reference is `Results/060/gemma-3-1b-it-nanoquant.gguf`, verified against its
retained export receipt:

```text
499,358,944 bytes
5e445cab6c93f03492ca0c9cea4ce7977fc780b9dbd75ad60578aef609dad02c
```

IQ2_M names a mixed quantization recipe. Inspection found 18.32% of original
parameters in IQ2_S, 5.46% in IQ3_S, 45.24% in IQ4_NL, 0.77% in Q5_0, 30.20%
in Q8_0, and the remaining 0.01% in F32. The complete external file costs
5.3589 bits per original weight versus 3.9953 for NanoQuant 060, using the same
999,885,952-parameter denominator independently checked against the source
checkpoint headers. The published file is 34.13% larger. This is not an
equal-storage-budget comparison. NanoQuant's historical 1.9634 charged factor
BPW excludes storage that the whole-file rate includes.

The two GGUFs have matching model architecture, token vocabulary/type mapping,
special token IDs, and byte-identical Q8_0 embedding tensors. The external file
omits the sliding-window rope frequency; the pinned runtime's default of 10000
matches the reference's explicit value. Its publisher names the same
`google/gemma-3-1b-it` source model but does not specify the upstream weight
revision. The reference uses `dcc83ea841ab6100d6b47a070329e1ba4cf78752`;
tokenizer and embedding agreement do not prove every upstream weight identical.

## Frozen scoring protocol

The original plan was to evaluate both deployment files with the same llama.cpp
build and exact prepared token inputs. IQ2_M completed in 175.42 seconds.
The additional fresh 060 GGUF rescore was deliberately stopped after 276 of
4,515 batches in 892.58 seconds: the observed pace implied several hours.
The larger I8 salient sidecars and slower custom kernel path are a likely
explanation; no completed 060 GGUF quality result was obtained or claimed.

The final comparison explicitly uses 060's completed PyTorch quality reference
and 048's completed GGUF benchmark. The interrupted scorer's original failure
exit, logs, frozen plan, and a separate `baseline-rescore-stop.json` receipt are
preserved. Its descendants were confirmed stopped and the device lease was
confirmed available. Partial scores are not used. The BF16 column reuses 060's
retained BF16 result.

- WikiText-2 test: 64 independent 128-token windows, 8,128 scored targets.
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.
- PIQA, ARC-Easy, ARC-Challenge, HellaSwag, Winogrande, and BoolQ: the first
  1,000 examples of each task, with retained prompts and normalization.
- Combined input: 18,060 sequences, SHA-256
  `b85a84dc6e2c9a7c9d57fb7a78be58382f4d2eb717e51f5c304b7f3a9c6bc0c1`.
- llama.cpp commit: `2927569f318311aded148d76be6b1109664066f5`.
- Scorer SHA-256: `cd035fa27d23ad8f3a8dca97ab2efb51a23dcfb41ef389b053e8b288c87e55ce`.
- Runtime SHA-256: `35cc485edf7a32e3f281704c239c9cf97c497d5ce7b4fd8503d14ea53555d8b4`.
- All layers on CUDA, parallel 4, runtime-default CPU thread settings.
- Causal target scoring with F32 logits and F64 host log-sum-exp; finite and
  complete per-token scoring is required.

The protocol, input identities, download receipt, model hashes, and launcher
hash are frozen before scoring in `Results/069/frozen-protocol.json`. Rendering
requires the 060 reference file hash and protocol to match that frozen identity.
The 048 deployment comparison must match IQ2_M's runtime and complete input
bytes, allowing only the non-semantic dataset-cache fingerprint to differ. Per-task
uncertainty uses 10,000 paired bootstrap resamples with fixed seed 6900; this
does not account for earlier model selection or multiple comparisons.

These are the retained diagnostic slices, not a fresh held-out evaluation gate.
The short-context perplexity values should only be compared under this exact
protocol. The evaluation-slice registry is not changed by this experiment.

## Reproduction and evidence

The downloaded file and model card are retained in
`outputs/external-2bit-profile/download/`, with the published metadata and
verified `download-receipt.json` in its parent. Given those inputs and the
retained Experiment 060 artifacts, the original optional full same-runtime plan
can be rerun with the first command below. It reuses the completed external
result and starts the expensive baseline rescore again. The second command
alone reproduces the final report using completed retained references:

```powershell
.\.venv\Scripts\python.exe tools\profile_downloaded_gguf.py
.\.venv\Scripts\python.exe tools\render_downloaded_gguf_profile.py
```

The evaluation runner validates the model/download receipts and prepared input
identities, uses the existing cross-process CUDA lease, and runs the two models
sequentially. Exact matching completed evaluator results can be reused; a change
to the frozen experiment identity is rejected.

`Results/069` retains the external raw per-example quality JSON, prepared input
binary, GGUF inventory, frozen protocol, event journal, interrupted-run status,
stop receipt, comparison summary, report, and PNG/SVG/PDF chart. The final
summary identifies its completed retained references by path and hash; the
scorer status correctly remains failed for the deliberately interrupted extra
rescore. IQ2_M resource peaks were 1,436.2 MiB dedicated GPU and 1,580.2 MiB host
working set. They describe this quality workload, not decode throughput.

The first preflight stopped before GPU use because Experiment 048's dataset
cache fingerprint differed from 060's. Their combined prepared input hash was
identical. The guard was corrected to require the current fingerprint to match
060, semantic protocol fields to match 048, and scorer inputs to match 048
byte for byte. Failed-attempt logs and the successful attempt's separate logs
are retained in `outputs/069/`.

Validation: the focused comparison/llama.cpp tests passed (12 tests); the full
default suite passed (1,516 tests, 49 deselected); Ruff and mypy passed (217
source files). The rendered chart was visually inspected. These checks validate
the reporting and existing software contracts; only the completed real-model
evaluations support the quality measurements above.
