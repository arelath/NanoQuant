"""Metadata-first, hash-verifying sharded safetensors model source."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import cast

import torch
from safetensors import safe_open

from nanoquant.domain.models import CheckpointInventory, CheckpointTensorMetadata, SourceTensor, TensorSpec
from nanoquant.infrastructure.io_utils import hash_file


class SourceIntegrityError(IOError):
    pass


def _safe_relative_shard(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
        raise SourceIntegrityError(f"unsafe shard path in index: {name!r}")
    return path.as_posix()


class SafetensorsModelSource:
    def __init__(self, snapshot: str | Path, *, source: str, revision: str, verify_hashes: bool = True) -> None:
        self.snapshot = Path(snapshot).resolve()
        self.source = source
        self.revision = revision
        self.verify_hashes = verify_hashes
        if not self.snapshot.is_dir():
            raise FileNotFoundError(self.snapshot)
        self._key_to_shard = self._discover_shards()
        self._shard_hashes: dict[str, str] = {}
        self._verified_signatures: dict[str, tuple[int, int, int]] = {}
        self._inventory: CheckpointInventory | None = None

    def _discover_shards(self) -> dict[str, str]:
        index = self.snapshot / "model.safetensors.index.json"
        if index.is_file():
            try:
                raw = json.loads(index.read_text(encoding="utf-8"))
                weight_map = cast(dict[str, str], raw["weight_map"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise SourceIntegrityError("invalid safetensors shard index") from exc
            indexed = {str(key): _safe_relative_shard(str(shard)) for key, shard in weight_map.items()}
            if len(indexed) != len(weight_map):
                raise SourceIntegrityError("duplicate tensor key in shard index")
            for shard in set(indexed.values()):
                if not (self.snapshot / shard).is_file():
                    raise SourceIntegrityError(f"indexed shard is missing: {shard}")
            return indexed
        shards = sorted(self.snapshot.glob("*.safetensors"))
        if not shards:
            raise SourceIntegrityError("snapshot contains no safetensors weights")
        discovered: dict[str, str] = {}
        for shard_path in shards:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in discovered:
                        raise SourceIntegrityError(f"tensor appears in multiple shards: {key}")
                    discovered[key] = shard_path.name
        return discovered

    def _hash(self, shard: str) -> str:
        if shard not in self._shard_hashes:
            path = self.snapshot / shard
            self._shard_hashes[shard] = hash_file(path)
            self._verified_signatures[shard] = self._signature(path)
        return self._shard_hashes[shard]

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _verify_unchanged(self, shard: str) -> None:
        path = self.snapshot / shard
        expected = self._hash(shard)
        signature = self._signature(path)
        if self._verified_signatures.get(shard) == signature:
            return
        if hash_file(path) != expected:
            raise SourceIntegrityError(f"source shard changed after inventory: {shard}")
        self._verified_signatures[shard] = signature

    def tensor_metadata(self) -> tuple[CheckpointTensorMetadata, ...]:
        grouped: dict[str, list[str]] = {}
        for key, shard in self._key_to_shard.items():
            grouped.setdefault(shard, []).append(key)
        metadata: list[CheckpointTensorMetadata] = []
        for shard in sorted(grouped):
            content_hash = f"sha256:{self._hash(shard)}" if self.verify_hashes else None
            with safe_open(self.snapshot / shard, framework="pt", device="cpu") as handle:
                actual = set(handle.keys())
                expected = set(grouped[shard])
                if actual != expected:
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    raise SourceIntegrityError(f"shard index mismatch for {shard}: missing={missing}, extra={extra}")
                for key in sorted(expected):
                    tensor_slice = handle.get_slice(key)
                    metadata.append(
                        CheckpointTensorMetadata(
                            key,
                            shard,
                            TensorSpec(tuple(tensor_slice.get_shape()), str(tensor_slice.get_dtype()).lower()),
                            content_hash,
                        )
                    )
        return tuple(metadata)

    def inventory(self) -> CheckpointInventory:
        if self._inventory is not None:
            return self._inventory
        config_path = self.snapshot / "config.json"
        try:
            config = cast(dict[str, object], json.loads(config_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceIntegrityError("missing or invalid config.json") from exc
        tokenizer_names = tuple(sorted(path.name for path in self.snapshot.glob("tokenizer*.*") if path.is_file()))
        tokenizer_digest = hashlib.sha256()
        for name in tokenizer_names:
            tokenizer_digest.update(name.encode("utf-8") + b"\0")
            tokenizer_digest.update(bytes.fromhex(hash_file(self.snapshot / name)))
        tensors = self.tensor_metadata()
        shards = {metadata.shard for metadata in tensors}
        self._inventory = CheckpointInventory(
            1,
            self.source,
            self.revision,
            config,
            tokenizer_names,
            f"sha256:{tokenizer_digest.hexdigest()}",
            tensors,
            sum((self.snapshot / shard).stat().st_size for shard in shards),
        )
        return self._inventory

    @contextmanager
    def read_tensor(self, key: str | SourceTensor, device: str = "cpu") -> Iterator[torch.Tensor]:
        source_key = key.source_key if isinstance(key, SourceTensor) else key
        try:
            shard = self._key_to_shard[source_key]
        except KeyError as exc:
            raise KeyError(f"tensor is not in source inventory: {source_key}") from exc
        if self.verify_hashes:
            self._verify_unchanged(shard)
        with safe_open(self.snapshot / shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(source_key)
            yield tensor if device == "cpu" else tensor.to(device)
