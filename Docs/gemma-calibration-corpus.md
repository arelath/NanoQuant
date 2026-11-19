# Gemma 3 imatrix calibration corpus candidate v1

This is a new **text calibration corpus**, not a computed importance matrix.
The local artifact is `Datasets/gemma-3-imatrix-calibration-v1/calibration.txt`.
It targets general text use of Gemma 3 1B IT and 4B IT. No model forward pass,
compression, or quality evaluation is part of corpus construction.

## Research and design

Calibration examples determine which activations are observed; more tokens alone
do not establish better downstream quality. [Williams and Aletras, ACL 2024](https://arxiv.org/abs/2311.09755)
find that sensitivity depends on the compression method, model, task, and data.
[Liu et al., 2024](https://arxiv.org/abs/2406.12928) show that matching the
calibration distribution to a particular test distribution is not always optimal.
These results motivate a diverse candidate and a controlled comparison; they do
not establish optimal weights for NanoQuant or Gemma 3.

The [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3)
describes web, code, mathematics, and multilingual training inputs. Google's
[formatting documentation](https://ai.google.dev/gemma/docs/core/prompt-structure)
describes its user/model turn markers. We render prompts through the actual
pinned tokenizer template, rather than substituting ChatML or hand-writing roles.
[llama.cpp's imatrix documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
requires `--parse-special` to interpret these control tokens.

[Bartowski's August 2026 study](https://huggingface.co/blog/bartowski/imatrix-dataset)
also motivates mixing prose and templated conversation data. Its strongest
coverage findings involve other models, particularly MoE models; they are not
proof of gains for dense Gemma 3 or NanoQuant. We do not adopt its model results
as acceptance evidence here.

The old Eaddario `combined_all_huge` corpus is a single mixed text stream with
100,001 lines. Inspection found 31,912 literal backslash-n sequences and no
Gemma turn markers. Its multilingual fragments and prompt lengths vary widely.
The new candidate adds coherent training prose and native prompt formatting,
sets explicit token budgets, deduplicates text, and excludes known lexical
overlaps with cached evaluation splits.

The initial target is **1,000,000 calibration tokens plus 65,536 held-out tokens**.
Whole records are retained, so actual totals slightly exceed those budgets.
The proportions below are an engineering starting point, not optimized weights.

| Stratum | Token budget | Source and rendering |
| --- | ---: | --- |
| Prose | 30% | WikiText-2 raw **train** paragraphs; BOS/text/EOS |
| Instructions | 20% | harmless_alpaca **train** prompts; Gemma user turns |
| Code | 25% | Eaddario code-like prompts; Gemma user turns |
| Math | 15% | Eaddario math-like prompts; Gemma user turns |
| Multiscript | 10% | Eaddario non-ASCII-heavy text; BOS/text/EOS |

Code/math are heuristic labels. Multiscript means more than 15% non-ASCII
alphabetic characters, with at least five such characters; it is **not** a
language classifier or a claim of balanced coverage of Gemma's languages.
The English training sources dominate this general-purpose candidate. A
multilingual-heavy deployment needs its own measured mixture.

## Sources, provenance, and limitations

All input files already existed in the local Hugging Face cache. Bulk downloads
were unavailable during this build. The tool is deliberately offline and pins:

| Source | Revision | Input |
| --- | --- | --- |
| [Eaddario](https://huggingface.co/datasets/eaddario/imatrix-calibration) | `e87ed55dcba9d9c3a3e41539f3e728e981b1daa4` | `combined_all_huge.parquet` |
| [harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca) | `02c6a92cfcf11bb0c387334f8146d149d65b587f` | `data/train-00000-of-00001.parquet` |
| [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) | `b08601e04326c79dfdd32d625aee71d232d685c3` | `wikitext-2-raw-v1/train-00000-of-00001.parquet` |

Retain source attribution and upstream terms. Eaddario's card declares MIT and
lists its contributing corpora; that declaration does not replace upstream text
and code licenses. WikiText's card declares CC BY-SA 3.0 and GFDL. The cached
harmless_alpaca card has no license declaration, so this local research mixture
does not have a blanket redistribution license. The generated data stays local
under the existing ignored `Datasets/*/` policy.

The cached instruction source has **prompts only**. No assistant answers were
invented, and prose was not mislabelled as assistant speech. Thus this candidate
exercises user-turn syntax but lacks assistant-response and multi-turn coverage.
A future revision should add licensed complete conversations, for example the
Apache-2.0 new subsets described in the [SmolTalk card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
The cached UltraChat snapshot contained only its README. C4's cached data was
validation data and was excluded as an input.

Literal source escapes are preserved. Blanket unescaping could corrupt code
string literals, so the candidate still contains some flattened code snippets.
This remains a data-quality limitation, not a resolved formatting defect.

## Filtering and split contract

- Normalize NFKC/case/whitespace for duplicate identity, while preserving source
  content for rendering. Drop blank, very short, oversized, NUL/replacement
  character, and embedded special-token records.
- Use 16–2,048 rendered tokens per record; do not truncate, repeat, or pad records.
- Build an exclusion index from locally cached test/validation/dev Arrow files
  in named benchmark datasets. Reject exact normalized matches and shared
  13-word spans after punctuation/case normalization. Hash every indexed file.
- Split before selection by seeded group hash. WikiText paragraphs from one
  article remain together. Reserve approximately 10% of groups as the candidate
  held-out pool, then select independently by stratum.
- Exclude any calibration record sharing a 13-word span with the selected
  held-out records. Interleave categories by emitted token counts so a short
  prefix is not a single-domain block.

The overlap audit is limited to cached evaluation files. Short questions,
paraphrases, uncached benchmarks, and pre-existing model-training contamination
are not ruled out. Internal near-duplicates that are not normalized exact
matches may remain. The candidate held-out subset is disjoint from this build;
some of its underlying text appeared in the earlier full Eaddario calibration
run. It is **not a fresh independent parity gate**, and must not replace the
project's reserved evaluation protocol or slice registry.

## Rebuild and use

```powershell
.\.venv\Scripts\python.exe tools/build_gemma_calibration_corpus.py `
  --output Datasets/gemma-3-imatrix-calibration-v1-rebuild `
  --verify-tokenizer C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-4b-it\snapshots\093f9f388b31de276ce2de164bdc2081324b9767
```

The default tokenizer is `google/gemma-3-1b-it` at
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`. The optional second tokenizer check
compares every rendered record and both full token streams exactly. Matching
text/tokenization does **not** make computed 1B importance matrices reusable for
4B: collect statistics separately for each exact model.

Outputs are `calibration.txt`, `heldout.txt`, their record/provenance JSONL files,
and `manifest.json`. The manifest pins source/tokenizer hashes, builder hash,
seed, exclusions, lengths, mixture counts, and compatibility results. An existing
output directory is refused. Keep the manifest and source inputs for reproduction.

The imatrix builder now accepts a local corpus. To collect importance later,
first check process ownership, the device lease, and `nvidia-smi` per `AGENTS.md`:

```powershell
.\.venv\Scripts\python.exe tools/build_imatrix_dataset.py `
  --model google/gemma-3-1b-it `
  --revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 `
  --calibration-text Datasets/gemma-3-imatrix-calibration-v1/calibration.txt `
  --parse-special --context-size 512 --chunks -1 `
  --dataset-dir Datasets/gemma-3-1b-it-imatrix-calibration-v1 `
  --local-files-only
```

This command performs model conversion and GPU inference; it was **not run**
as part of corpus preparation. Use `--prepare-only` if only conversion is wanted.
The text is not pre-padded: llama.cpp discards the final incomplete context.
The recorded Hugging Face token counts do not establish llama.cpp's exact
processed count; its BOS and chunking policy still applies.

## Validation and remaining acceptance

### Corpus validation on 2026-11-15

| Measurement | Calibration | Held-out |
| --- | ---: | ---: |
| Records | 15,449 | 989 |
| Gemma tokens | 1,000,294 | 65,933 |
| UTF-8 bytes | 4,597,925 | 301,709 |
| Unknown tokens / padding tokens | 0 / 0 | 0 / 0 |
| Complete 512-token contexts, before llama.cpp policy | 1,953 | 128 |
| Complete 2,048-token contexts, before llama.cpp policy | 488 | 32 |

The exclusion index covered 31 cached evaluation files and 19,346,915 unique
13-word hashes. It rejected 77 candidate records for evaluation overlap.
Validation re-read every selected record against its original source and checked
source/output hashes, unique IDs, record lengths, and exact reconstruction of
text from JSONL. Calibration and held-out data have zero shared record IDs,
source groups, or 13-word spans. Both pinned Gemma tokenizers match exactly.
A second independent build reproduced all four data files **and the manifest
byte-for-byte**. Local `validation.json` records these checks.

Calibration text SHA-256:
`3d8fa1e99d56ca7ba8455a1aabd5b6d10a82ca1a7d502f7fb15450cba290e301`.

Software validation: 23 focused tests passed; repository-wide Ruff and mypy
passed. With Hugging Face offline mode enabled and a workspace-local pytest
temporary directory, the full suite completed with 1,808 passed, 54 deselected,
and four environment-dependent failures. Three existing export/checkpoint tests
failed because Windows denied `os.link`; the fourth benchmark-provenance test
expects its temporary launcher to be outside the repository, whereas the
workspace-local `--basetemp` placed it inside. These failures do not exercise
the new corpus builder. The default sandbox temporary directory also denies
atomic file replacement, which is why workspace-local temporary paths were used.
The full output is retained locally as `Datasets/gemma-3-imatrix-calibration-v1/software-tests.log`.

### Model-level comparison

Compare old and new corpora using the same model revision, imatrix executable,
token budget, context length, quantization recipe, and evaluation protocol.
Keep corpus changes separate from context-length changes: start with matched
512-token contexts, then test 2,048 separately. Compare held-out KL, WikiText/C4
perplexity, task accuracy, instruction behavior, runtime, and memory. Include
multiple corpus seeds where affordable. Report regressions as well as gains.
Only that controlled model-level comparison can establish improved quality.

The subsequent [270M imatrix comparison](Experiments/077-078-imatrix-calibration-comparison.md)
records the generated row-bearing matrix, equal-budget Eaddario control, native
GGUF perplexity comparison, and the separate NanoQuant 077/078 acceptance status.
