from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nanoquant.config.codec import canonical_json, semantic_hash
from nanoquant.domain.linear_math import chunk_slices, chunked_reduce, parse_torch_dtype
from nanoquant.infrastructure.memory_cleanup import explicit_memory_cleanup
from nanoquant.infrastructure.safetensors_io import load_tensors
from nanoquant.runtime.codec import RuntimeDecodeError, decode_dataclass


def test_semantic_hash_matches_the_existing_canonical_identity() -> None:
    payload = {"b": (2, 3), "a": 1}
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    assert semantic_hash(payload) == f"sha256:{expected}"


def test_chunked_reduction_covers_the_leading_dimension() -> None:
    value = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    assert tuple(chunk_slices(5, 2)) == (slice(0, 2), slice(2, 4), slice(4, 5))
    assert chunked_reduce(value, 2, lambda chunk: chunk.square().sum()) == value.square().sum()
    with pytest.raises(ValueError, match="positive"):
        tuple(chunk_slices(1, 0))


def test_dtype_parser_has_one_consistent_failure_mode() -> None:
    assert parse_torch_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="unsupported torch dtype"):
        parse_torch_dtype("float128")


def test_explicit_memory_cleanup_runs_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("nanoquant.infrastructure.memory_cleanup.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    with pytest.raises(RuntimeError, match="fixture"), explicit_memory_cleanup("cuda:0"):
        raise RuntimeError("fixture")
    assert calls == ["gc", "cuda"]


def test_safetensors_loader_validates_keys_and_places_tensors(tmp_path: Path) -> None:
    path = tmp_path / "fixture.safetensors"
    save_file({"left": torch.arange(3), "right": torch.arange(2)}, path)
    loaded = load_tensors(path, ("right", "left"))
    assert tuple(loaded) == ("right", "left")
    assert torch.equal(loaded["left"], torch.arange(3))
    with pytest.raises(KeyError, match="missing"):
        load_tensors(path, ("absent",))


@dataclass(frozen=True)
class _Child:
    value: int


@dataclass(frozen=True)
class _Manifest:
    name: str
    children: tuple[_Child, ...]
    enabled: bool = True


def test_runtime_decoder_handles_nested_tuples_defaults_and_paths() -> None:
    decoded = decode_dataclass(_Manifest, {"name": "fixture", "children": [{"value": 3}]})
    assert decoded == _Manifest("fixture", (_Child(3),))
    with pytest.raises(RuntimeDecodeError, match=r"manifest\.children\[0\]\.value"):
        decode_dataclass(_Manifest, {"name": "fixture", "children": [{"value": "bad"}]})
