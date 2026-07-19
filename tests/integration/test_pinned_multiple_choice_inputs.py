from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from nanoquant.application.task_evaluation import (
    pinned_legacy_multiple_choice_tasks,
    render_pinned_task_document,
    tokenize_multiple_choice_example,
)
from nanoquant.config.codec import canonical_json
from nanoquant.infrastructure.hf_task_evaluation import (
    HFCausalPairTokenizer,
    hash_hf_tokenizer_snapshot,
    load_pinned_multiple_choice_documents,
)

GEMMA_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
TOKENIZER_CONTENT_HASH = "sha256:19317db471b30f6cfa877d781ecac1db28de6628e44e3751df0c44344444a811"
RETAINED_TEXT_HASHES = {
    "piqa": "5691f43910a69e5b4d6e65e651fb02b3639a047ecaad5832cefd553ee9d4c3c8",
    "arc_easy": "3b4a2d1d4eb07403cfae855007043ced09a53e4d27b34c3795c289375b83cbeb",
    "arc_challenge": "ffd2a64c115a7936fe980d220b5b802e43cd601a81ccaa5ade30b91c7ae6dbe2",
    "hellaswag": "82739219844b2767b187d46f9f2ec5f78b37bb77ae9434ba6b7b9b45233473ee",
    "winogrande": "955679b0eec2f6a60d31427e925b61165ecb2ff0dfbbf0ed6a15a3b1b66716a1",
    "boolq": "981ea7040c683647479f281f31b71b27ba7209dbde83e5972fbfd40cfab8e2a8",
}
RETAINED_TOKEN_HASHES = {
    "piqa": "e18822cc67c4b8cbf8869747ec7ae12ed02b9e3fa08066d9bb79ae26fe1fb561",
    "arc_easy": "3fde10483e911935e98575fd1216b81880c692d7799e6981327a74fcd8baf26f",
    "arc_challenge": "4b5ec3f6a52f178ec735b797d6eba7756e9540702b5745370fd7ad257ecabde7",
    "hellaswag": "0aa361be563c2334ebb79e89bece49a29906f8b2ab9704ef8ec5ac27d32e6bb8",
    "winogrande": "df558e4279c33bf9a0a245c1e6bfd7daa0b46292c8e0f09befc1da1566c1c2d5",
    "boolq": "7a8724e2c513112780bc8e5950de4471436d5b99b958c23dd45ebe752ab22f80",
}


def _snapshot() -> Path:
    configured = os.environ.get("NANOQUANT_GEMMA_SNAPSHOT")
    if configured:
        return Path(configured)
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--google--gemma-3-1b-it"
        / "snapshots"
        / GEMMA_REVISION
    )


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@pytest.mark.external_data
def test_local_pinned_datasets_and_gemma_tokenizer_match_retained_legacy_samples() -> None:
    snapshot = _snapshot()
    if not snapshot.is_dir():
        pytest.skip("pinned Gemma tokenizer snapshot is not available locally")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encode_pair = HFCausalPairTokenizer(tokenizer, add_special_tokens=True)

    assert tokenizer.add_bos_token is True
    assert hash_hf_tokenizer_snapshot(snapshot) == TOKENIZER_CONTENT_HASH
    for task in pinned_legacy_multiple_choice_tasks():
        try:
            document = load_pinned_multiple_choice_documents(
                task,
                maximum_samples=1,
                local_files_only=True,
            )[0]
        except FileNotFoundError as exc:  # pragma: no cover - depends on an optional local cache
            pytest.skip(f"pinned {task.task_name} dataset is not available locally: {exc}")
        text = render_pinned_task_document(task, document, sample_id="retained-0")
        tokenized = tokenize_multiple_choice_example(text, encode_pair)

        assert _hash((text.contexts, text.continuations, text.correct_choice)) == RETAINED_TEXT_HASHES[task.task_name]
        assert (
            _hash((tokenized.contexts, tokenized.continuations, tokenized.correct_choice))
            == RETAINED_TOKEN_HASHES[task.task_name]
        )
