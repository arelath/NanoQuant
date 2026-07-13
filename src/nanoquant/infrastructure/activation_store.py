"""CUDA, pinned-host, and pageable-host activation stores."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch

from nanoquant.domain.resources import ResourcePlan


class MemoryActivationStore:
    def __init__(self, kind: str, device: str | None = None) -> None:
        if kind not in {"cuda", "pinned_ram", "ram"}:
            raise ValueError(f"unsupported in-memory activation tier: {kind}")
        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA activation storage requested without CUDA")
        self.kind = kind
        self.device = device or ("cuda:0" if kind == "cuda" else "cpu")
        self._values: dict[str, torch.Tensor] = {}

    def put(self, key: str, value: torch.Tensor) -> None:
        if not key or key in self._values:
            raise ValueError("activation key must be non-empty and unique")
        if self.kind == "cuda":
            stored = value.detach().to(self.device).clone()
        else:
            stored = value.detach().to("cpu").clone()
            if self.kind == "pinned_ram":
                stored = stored.pin_memory()
        self._values[key] = stored

    @contextmanager
    def read(self, key: str, device: str | None = None) -> Iterator[torch.Tensor]:
        try:
            value = self._values[key]
        except KeyError as exc:
            raise KeyError(f"activation is not stored: {key}") from exc
        target = (
            value if device is None or str(value.device) == device else value.to(device, non_blocking=value.is_pinned())
        )
        yield target

    def remove(self, key: str) -> None:
        del self._values[key]

    def clear(self) -> None:
        self._values.clear()


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class MmapGenerationWriter:
    def __init__(self, store: MmapActivationStore, key: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        if not key or not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("mmap activation generation requires a key and positive shape")
        if dtype not in _DTYPES.values():
            raise ValueError(f"unsupported mmap activation dtype: {dtype}")
        if store._descriptor(key).exists():
            raise ValueError(f"activation key already exists: {key}")
        self.store = store
        self.key = key
        self.shape = shape
        self.dtype = dtype
        self._written = torch.zeros(shape[0], dtype=torch.bool)
        descriptor, temporary = tempfile.mkstemp(prefix="activation-", suffix=".tmp", dir=store.temporary)
        os.close(descriptor)
        self.temporary = Path(temporary)
        self.numel = 1
        for dimension in shape:
            self.numel *= dimension
        self.temporary.write_bytes(b"")
        with self.temporary.open("r+b") as output:
            output.truncate(self.numel * torch.empty((), dtype=dtype).element_size())
        self._mapping = torch.from_file(str(self.temporary), shared=True, size=self.numel, dtype=dtype).reshape(shape)
        self._committed = False

    def write(self, selection: slice, values: torch.Tensor) -> None:
        start, stop, step = selection.indices(self.shape[0])
        if step != 1 or stop <= start:
            raise ValueError("activation writes require a non-empty contiguous batch slice")
        expected = (stop - start, *self.shape[1:])
        if tuple(values.shape) != expected:
            raise ValueError(f"activation batch shape {tuple(values.shape)} does not match {expected}")
        if self._written[start:stop].any():
            raise ValueError("activation batch overlaps a prior write")
        self._mapping[start:stop].copy_(values.detach().to(device="cpu", dtype=self.dtype))
        self._written[start:stop] = True

    def commit(self) -> str:
        if self._committed:
            raise RuntimeError("activation generation was already committed")
        if not self._written.all():
            missing = torch.where(~self._written)[0].tolist()
            raise ValueError(f"activation generation has unwritten batches: {missing[:8]}")
        del self._mapping
        content_hash = _hash_file(self.temporary)
        destination = self.store._data(self.key)
        os.replace(self.temporary, destination)
        metadata = {
            "schema_version": 1,
            "key": self.key,
            "shape": list(self.shape),
            "dtype": str(self.dtype).removeprefix("torch."),
            "bytes": destination.stat().st_size,
            "content_hash": content_hash,
        }
        descriptor, temporary = tempfile.mkstemp(prefix="descriptor-", suffix=".tmp", dir=self.store.temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(metadata, output, sort_keys=True, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.store._descriptor(self.key))
        self._committed = True
        return content_hash

    def close(self) -> None:
        if hasattr(self, "_mapping"):
            del self._mapping
        if not self._committed and self.temporary.exists():
            self.temporary.unlink()

    def __enter__(self) -> MmapGenerationWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class MmapActivationStore:
    kind = "mmap"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.temporary = self.directory / ".tmp"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _name(key: str) -> str:
        if not key:
            raise ValueError("activation key must be non-empty")
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _data(self, key: str) -> Path:
        return self.directory / f"{self._name(key)}.bin"

    def _descriptor(self, key: str) -> Path:
        return self.directory / f"{self._name(key)}.json"

    def begin_generation(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> MmapGenerationWriter:
        return MmapGenerationWriter(self, key, shape, dtype)

    def put(self, key: str, value: torch.Tensor) -> None:
        with self.begin_generation(key, tuple(value.shape), value.dtype) as writer:
            writer.write(slice(0, value.shape[0]), value)
            writer.commit()

    def _metadata(self, key: str) -> dict[str, Any]:
        try:
            value = json.loads(self._descriptor(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KeyError(f"activation is not stored: {key}") from exc
        if value.get("key") != key:
            raise OSError("ACT001 activation descriptor identity mismatch")
        data = self._data(key)
        if (
            not data.is_file()
            or data.stat().st_size != value.get("bytes")
            or _hash_file(data) != value.get("content_hash")
        ):
            raise OSError("ACT001 activation generation is corrupt")
        return cast(dict[str, Any], value)

    @contextmanager
    def read(self, key: str, device: str | None = None, selection: slice | None = None) -> Iterator[torch.Tensor]:
        metadata = self._metadata(key)
        dtype_name = str(metadata["dtype"])
        try:
            dtype = _DTYPES[dtype_name]
        except KeyError as exc:
            raise OSError(f"ACT001 unsupported activation dtype: {dtype_name}") from exc
        shape = tuple(int(value) for value in metadata["shape"])
        numel = 1
        for dimension in shape:
            numel *= dimension
        mapped = torch.from_file(str(self._data(key)), shared=False, size=numel, dtype=dtype).reshape(shape)
        value = mapped if selection is None else mapped[selection]
        target = value if device is None or device == "cpu" else value.to(device)
        yield target

    def remove(self, key: str) -> None:
        descriptor = self._descriptor(key)
        data = self._data(key)
        if not descriptor.exists():
            raise KeyError(f"activation is not stored: {key}")
        descriptor.unlink()
        data.unlink(missing_ok=True)

    def clear(self) -> None:
        for descriptor in self.directory.glob("*.json"):
            descriptor.unlink()
        for data in self.directory.glob("*.bin"):
            data.unlink()

    def cleanup_uncommitted(self) -> int:
        removed = 0
        for path in self.temporary.glob("*.tmp"):
            path.unlink()
            removed += 1
        committed_data = {
            self._data(json.loads(path.read_text(encoding="utf-8"))["key"]) for path in self.directory.glob("*.json")
        }
        for path in self.directory.glob("*.bin"):
            if path not in committed_data:
                path.unlink()
                removed += 1
        return removed


def activation_store_for_plan(
    plan: ResourcePlan, directory: str | Path, *, device: str = "cuda:0"
) -> MemoryActivationStore | MmapActivationStore:
    if plan.activation_tier == "mmap":
        return MmapActivationStore(directory)
    return MemoryActivationStore(plan.activation_tier, device if plan.activation_tier == "cuda" else None)
