# Reusable Teacher-Response Dataset Builder

Status: implemented; first real dataset generation and Hugging Face publication pending

## Purpose

Thinking-model compression should not spend hours regenerating the same teacher answers inside every experiment.
`tools/build_teacher_dataset.py` creates a small, pinned, reusable conversational dataset first. Compression runs can
then consume the uploaded `messages` records without loading a teacher model or repeating generation.

The builder initially supports Qwen3-family chat templates. It can use a larger member of the same family—such as
Qwen3-8B instead of Qwen3-0.6B—as the teacher. That is a deliberate teacher-transfer target, not an exact
source-model reproduction, and the teacher model and revision remain visible in every row and in the manifest.

## Default scope

The default source is pinned `HuggingFaceH4/ultrachat_200k`, but the builder does not process all 200,000 rows. It
stops after accepting **512 complete responses per requested mode**:

- dual mode: 512 thinking plus 512 non-thinking records, 1,024 total;
- thinking only: 512 records;
- non-thinking only: 512 records.

The count is configurable. The source is streamed and deterministically shuffled, so the full dataset is not loaded
into memory or sent through the teacher.

For every selected source record, the original final assistant response is discarded. The teacher generates the
complete replacement response. Thinking records retain a coherent reasoning span and final answer from the same
generation; non-thinking records retain the teacher's complete non-thinking answer.

## Interactive menu

Run:

```powershell
.\.venv\Scripts\python.exe tools\build_teacher_dataset.py
```

The menu proceeds in this order:

1. **Resume or start new.** On the second invocation, continuing the newest persisted run is the default.
2. **Prompt dataset.**
   - UltraChat 200K is the default.
   - A custom Hugging Face conversational dataset can supply its ID, revision, split, configuration, and messages
     column.
3. **Teacher family.** The ordered Qwen family list is read from
   `experiments/recipes/interactive_recommended_models.yaml`.
4. **Teacher size.** Qwen3-8B is the default Qwen3 teacher; larger or smaller listed variants and a custom model ID
   are available.
5. **Response modes.**
   - thinking and non-thinking, default;
   - thinking only;
   - non-thinking only.
6. **Accepted responses per mode.** Default: 512.
7. **Maximum complete sequence length.** Default: 2,048 tokens. A response that cannot finish within the limit is
   rejected rather than truncated mid-reasoning.
8. **Generation backend.**
   - llama.cpp server, default;
   - Transformers greedy generation.
9. **Generation device.** Default: `cuda`.
10. **Hugging Face upload.** Upload after local completion is the default. The user chooses the dataset repository
    and visibility; private is the default.
11. **Confirmation.** The menu shows pinned revisions, mode/count, backend, destination, and upload repository before
    writing settings and starting.

Settings are written before model loading. An interruption preserves accepted/rejected attempt journals, and the
next invocation continues without skipping or regenerating committed responses.

## Parameterized use

Example using Qwen3-8B to produce a small dual-mode dataset:

```powershell
.\.venv\Scripts\python.exe tools\build_teacher_dataset.py `
  --output evidence\teacher-datasets\qwen3-8b-ultrachat-512 `
  --teacher-model Qwen/Qwen3-8B `
  --mode both `
  --samples-per-mode 512 `
  --backend llamacpp `
  --device cuda `
  --hub-repo OWNER/qwen3-8b-ultrachat-teacher-responses
```

Resume it with:

```powershell
.\.venv\Scripts\python.exe tools\build_teacher_dataset.py `
  --resume evidence\teacher-datasets\qwen3-8b-ultrachat-512
```

Useful parameters include:

| Parameter | Meaning | Default |
| --- | --- | --- |
| `--teacher-model` | Teacher model ID | required outside the menu |
| `--teacher-revision` | Pinned teacher commit | current commit is resolved once |
| `--source-dataset` | Conversational prompt dataset | UltraChat 200K |
| `--source-revision` | Pinned dataset commit | pinned UltraChat revision or resolved commit |
| `--source-config` | Dataset configuration/subset | none |
| `--source-split` | Dataset split | `train_sft` |
| `--messages-column` | Conversation column | `messages` |
| `--mode` | `both`, `thinking`, or `non-thinking` | `both` |
| `--samples-per-mode` | Accepted records per mode | 512 |
| `--sequence-length` | Prompt plus complete response limit | 2,048 |
| `--maximum-new-tokens` | Response generation cap | 1,536 |
| `--minimum-new-tokens` | Minimum accepted response length | 16 |
| `--maximum-attempt-multiplier` | Rejection/attempt budget | 20 |
| `--seed` | Deterministic source shuffle | 0 |
| `--backend` | `llamacpp` or `transformers` | `llamacpp` |
| `--device` | Generation device | `cuda` |
| `--hub-repo` | Dataset repository; enables upload | no upload in parameter mode |
| `--public` | Publish a public repository | private |

## Durable layout and resume

```text
evidence/teacher-datasets/<run>/
  settings.yaml
  .active-lease.json                 # present only while a process owns the run
  state/teacher-traces/*.jsonl       # append-only accepted/rejected attempts
  state/teacher-traces/*.json        # active trace artifact receipts
  artifacts/                         # content-addressed complete response turns
  dataset/
    README.md                        # Hugging Face dataset card/configuration map
    manifest.json                    # identities, counts, ordered IDs, file hashes
    data/thinking.jsonl
    data/non_thinking.jsonl
  huggingface-upload.json            # repository and commit receipt, when uploaded
  completion.json
```

An immutable settings hash covers the source, teacher, modes, generation limits, device/backend, and upload request.
The reusable dataset identity excludes only its publication destination and creation timestamp. Existing local data
is revalidated by size and SHA-256 before reuse. Two processes cannot own the same run directory concurrently.

If upload fails after local generation, rerunning only retries publication. A completed upload receipt prevents a
second commit.

## Published dataset schema

The Hugging Face payload exposes separate `thinking` and `non_thinking` configurations and an `all` configuration
when both modes were generated. Each uses split `train`.

Every row includes:

- `messages`, compatible with NanoQuant's `ultrachat_messages` record format;
- `mode`;
- prompt source/revision/split/subset and source-record hash;
- teacher model/revision and generation implementation;
- prompt, response, and complete-token hashes;
- prompt/response token counts and stop reason.

The uploaded revision must be pinned when used for compression. A static behavior slice looks like:

```python
BehaviorSliceConfig(
    "thinking",
    ReasoningMode.THINKING,
    DatasetSourceConfig(
        "OWNER/qwen3-8b-ultrachat-teacher-responses",
        revision="<uploaded dataset commit>",
        subset="thinking",
        split="train",
    ),
    "ultrachat_messages",
    0.50,
)
```

There is no `teacher_trace_generation` setting in the consuming slice. Preparation reads the already completed
teacher turn directly, so repeated compression runs share the exact same behavior targets.

## Validation and publication policy

A response is accepted only when it:

- terminates with a recognized EOS/end-of-turn token;
- preserves the rendered prompt prefix;
- satisfies the requested thinking or non-thinking delimiter invariant;
- has a non-empty final answer and, for thinking mode, a non-empty reasoning span;
- fits as a complete record within the sequence limit;
- round-trips through the pinned teacher chat template.

The builder uploads only after all requested local mode files and their manifest have been atomically published.
Public upload is never the default because prompt and model licensing and generated content must be reviewed first.
