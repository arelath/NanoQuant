"""Safetensors I/O helpers for research infrastructure."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_ACTIVATION_DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


def activation_tensor_chunks(name: str, tensor: torch.Tensor) -> Iterable[bytes | memoryview] | None:
    """Return a byte-identical single-tensor safetensors stream for common activations."""
    dtype = _ACTIVATION_DTYPES.get(tensor.dtype)
    if dtype is None or sys.byteorder != "little" or tensor.is_neg():
        return None
    payload = tensor.reshape(-1).view(torch.uint8).numpy().data
    header = json.dumps(
        {name: {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [0, len(payload)]}},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)

    def chunks() -> Iterator[bytes | memoryview]:
        yield len(header).to_bytes(8, "little") + header
        chunk_bytes = 16 * 1024 * 1024
        for start in range(0, len(payload), chunk_bytes):
            yield payload[start : start + chunk_bytes]

    return chunks()


class SafetensorsManager:
    """Single research-side entry point for bounded safetensors I/O."""

    @contextmanager
    def open(
        self,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> Iterator[Any]:
        with safe_open(Path(path), framework="pt", device=str(torch.device(device))) as handle:
            yield handle

    def load(
        self,
        path: str | Path,
        keys: Iterable[str] | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        requested = None if keys is None else tuple(keys)
        if requested is not None and (not requested or len(requested) != len(set(requested))):
            raise ValueError("safetensors keys must be non-empty and unique")
        destination = torch.device(device)
        result: dict[str, torch.Tensor] = {}
        with self.open(path) as handle:
            available = tuple(handle.keys())
            selected = available if requested is None else requested
            missing = tuple(key for key in selected if key not in available)
            if missing:
                raise KeyError(f"safetensors keys are missing: {missing}")
            for key in selected:
                value = handle.get_tensor(key)
                result[key] = value if destination.type == "cpu" else value.to(destination)
        return result

    def save(
        self,
        tensors: Mapping[str, torch.Tensor],
        path: str | Path,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        if not tensors:
            raise ValueError("cannot write an empty safetensors file")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(dict(tensors), Path(path), metadata=None if metadata is None else dict(metadata))


SAFETENSORS = SafetensorsManager()


def load_tensors(
    path: str | Path,
    keys: Iterable[str],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load requested tensors through CPU and place them on ``device``."""

    return SAFETENSORS.load(path, keys, device=device)


__all__ = ["SAFETENSORS", "SafetensorsManager", "load_tensors"]
