"""Full standard MMLU with five same-subject examples and answer-letter scoring."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nanoquant.infrastructure.io_utils import hash_file

MMLU_REPO = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_QUESTIONS = 14_042
MMLU_SUBJECTS = 57
_LETTERS = "ABCD"
_MAX_BATCH = 8
_MAX_BATCH_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class PreparedMmlu:
    tokens: tuple[tuple[int, ...], ...]
    subjects: tuple[str, ...]
    gold: tuple[int, ...]
    labels: tuple[int, ...]
    identity: dict[str, Any]


def mmlu_files(*, local_files_only: bool) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {
        split: Path(hf_hub_download(
            MMLU_REPO, f"all/{split}-00000-of-00001.parquet", repo_type="dataset",
            revision=MMLU_REVISION, local_files_only=local_files_only,
        ))
        for split in ("dev", "test")
    }


def _question(row: dict[str, Any]) -> str:
    options = row["choices"]
    if len(options) != len(_LETTERS) or row["answer"] not in range(len(_LETTERS)):
        raise ValueError("MMLU row must have four choices and a valid answer")
    return "\n".join([
        row["question"].strip(),
        *(f"{letter}. {value.strip()}" for letter, value in zip(_LETTERS, options, strict=True)),
        "Answer:",
    ])


def prepare_mmlu(tokenizer: Any, *, local_files_only: bool) -> PreparedMmlu:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    paths = mmlu_files(local_files_only=local_files_only)
    demonstrations: dict[str, list[str]] = defaultdict(list)
    for row in pq.read_table(paths["dev"]).to_pylist():
        demonstrations[row["subject"]].append(f"{_question(row)} {_LETTERS[row['answer']]}\n\n")
    rows = pq.read_table(paths["test"]).to_pylist()
    subjects = tuple(str(row["subject"]) for row in rows)
    if (len(rows) != MMLU_QUESTIONS or len(demonstrations) != MMLU_SUBJECTS
            or set(subjects) != demonstrations.keys() or any(len(v) != 5 for v in demonstrations.values())):
        raise ValueError("MMLU requires the full pinned test split and five dev questions per subject")
    prefixes = {
        subject: f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        + "".join(items)
        for subject, items in demonstrations.items()
    }
    prompts = [prefixes[subject] + _question(row) for subject, row in zip(subjects, rows, strict=True)]
    tokens = tuple(tuple(ids) for ids in tokenizer(prompts, add_special_tokens=True, verbose=False).input_ids)
    prefix = tokenizer.encode("Answer:", add_special_tokens=False)
    label_ids = []
    for letter in _LETTERS:
        joined = tokenizer.encode(f"Answer: {letter}", add_special_tokens=False)
        if joined[:-1] != prefix:
            raise ValueError("MMLU answer letters must be single-token continuations after Answer:")
        label_ids.append(int(joined[-1]))
    if len(set(label_ids)) != len(_LETTERS) or not all(tokens):
        raise ValueError("MMLU tokenization produced ambiguous labels or an empty prompt")
    return PreparedMmlu(tokens, subjects, tuple(int(row["answer"]) for row in rows), tuple(label_ids), {
        "dataset": MMLU_REPO, "revision": MMLU_REVISION, "samples": len(rows), "subjects": len(demonstrations),
        "split_sha256": {name: hash_file(path) for name, path in paths.items()},
        "prompt_token_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        "answer_label_ids": label_ids, "few_shot": 5, "scoring": "argmax-answer-letter-next-token-logit-v1",
        "max_prompt_tokens": max(map(len, tokens)), "max_batch": _MAX_BATCH, "max_batch_tokens": _MAX_BATCH_TOKENS,
    })


@torch.inference_mode()
def evaluate_mmlu(
    model: nn.Module, prepared: PreparedMmlu, *, device: str, pad_token_id: int,
    after_batch: Callable[[], None],
) -> dict[str, Any]:
    ordered = sorted(range(len(prepared.tokens)), key=lambda i: len(prepared.tokens[i]))
    predictions = [-1] * len(ordered)
    labels = torch.tensor(prepared.labels, device=device)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and end - start < _MAX_BATCH:
            if (end - start + 1) * len(prepared.tokens[ordered[end]]) > _MAX_BATCH_TOKENS:
                break
            end += 1
        batch = ordered[start:end]
        rows = [prepared.tokens[i] for i in batch]
        width = len(rows[-1])
        ids = torch.tensor([[pad_token_id] * (width - len(row)) + list(row) for row in rows], device=device)
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows], device=device)
        logits = model(
            input_ids=ids, attention_mask=mask, position_ids=mask.cumsum(dim=1).sub(1).clamp_min(0),
            use_cache=False, logits_to_keep=1,
        ).logits[:, -1].index_select(1, labels)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("MMLU scorer received non-finite answer logits")
        for index, answer in zip(batch, logits.argmax(dim=1).tolist(), strict=True):
            predictions[index] = answer
        after_batch()
        start = end
    counts = Counter(prepared.subjects)
    hits = Counter(subject for subject, pred, gold in zip(
        prepared.subjects, predictions, prepared.gold, strict=True,
    ) if pred == gold)
    correct = hits.total()
    return {
        "task_name": "mmlu", "sample_count": len(ordered), "raw_correct_count": correct,
        "primary_metric": "acc", "primary_value": correct / len(ordered),
        "subjects": {s: {"count": counts[s], "correct": hits[s]} for s in sorted(counts)},
        "examples": [{"id": i, "subject": subject, "predicted": pred, "gold": gold}
                     for i, (subject, pred, gold) in enumerate(zip(prepared.subjects, predictions, prepared.gold,
                                                                  strict=True))],
    }
