"""Read llama.cpp activation importance without modifying its source file.

Legacy sums/ncall and GGUF in_sum2/counts follow llama.cpp common/imatrix-loader
and tools/quantize. Multi-expert entries are retained, never averaged into a
single dense projection. The supported resident path consumes dense entries.
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nanoquant.config.schema import ImatrixConfig, SharedInputGroupConfig
from nanoquant.domain.models import LayerId, ModelInventory


@dataclass(frozen=True)
class ImatrixData:
    sha256: str
    format: str
    entries: dict[str, torch.Tensor]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def imatrix_config(path: str | Path) -> ImatrixConfig:
    """Pin a local file for a recipe. Full format validation occurs here too."""
    source = Path(path).resolve()
    data = load_imatrix(source)
    return ImatrixConfig(str(source), data.sha256)


def _normalized(sums: np.ndarray[Any, Any], counts: np.ndarray[Any, Any], name: str) -> torch.Tensor:
    values = np.asarray(sums, dtype=np.float64).reshape(-1)
    count = np.asarray(counts, dtype=np.float64).reshape(-1)
    if (not values.size or not count.size or values.size % count.size
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.isfinite(count).all() or (count <= 0).any()
            or not np.equal(count, np.rint(count)).all()):
        raise ValueError(f'invalid imatrix sums/counts for {name}; positive integral observation counts required')
    means = values.reshape(count.size, -1) / count[:, None]
    if not np.isfinite(means).all() or (means > np.finfo(np.float32).max).any() or (means.sum(axis=1) <= 0).any():
        raise ValueError(f'invalid or all-zero imatrix importance for {name}')
    return torch.from_numpy(means.astype(np.float32))


def _legacy(payload: bytes) -> dict[str, torch.Tensor]:
    stream = io.BytesIO(payload)

    def read(size: int) -> bytes:
        if size < 0 or size > len(payload) - stream.tell():
            raise ValueError('truncated or invalid legacy imatrix length')
        return stream.read(size)

    def integer() -> int:
        return int(struct.unpack('<i', read(4))[0])

    count = integer()
    if not 0 < count <= len(payload) // 17:
        raise ValueError('invalid legacy imatrix entry count')
    result = {}
    for _ in range(count):
        size = integer()
        if not 0 < size <= 65536:
            raise ValueError('invalid legacy imatrix tensor name length')
        name = read(size).decode('utf-8')
        calls, length = integer(), integer()
        if length <= 0 or name in result or '\x00' in name:
            raise ValueError(f'invalid or duplicate imatrix entry: {name}')
        result[name] = _normalized(np.frombuffer(read(4 * length), dtype='<f4'), np.array([calls]), name)
    if stream.tell() < len(payload):
        if integer() < 0:
            raise ValueError('negative imatrix chunk count')
        if stream.tell() < len(payload):
            read(integer()).decode('utf-8')
        if stream.tell() != len(payload):
            raise ValueError('unexpected trailing imatrix bytes')
    return result


def _gguf(path: Path) -> dict[str, torch.Tensor]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path), 'r')
    kind = reader.get_field('general.type')
    if kind is None or kind.contents() != 'imatrix':
        raise ValueError('GGUF general.type must be imatrix (not a model GGUF)')
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name in tensors:
            raise ValueError(f'duplicate imatrix tensor: {tensor.name}')
        if tensor.data.dtype.kind != 'f' or tensor.data.dtype.itemsize != 4:
            raise ValueError(f'imatrix tensor must be F32: {tensor.name}')
        tensors[tensor.name] = tensor.data
    names = {name.removesuffix('.in_sum2') for name in tensors if name.endswith('.in_sum2')}
    counts = {name.removesuffix('.counts') for name in tensors if name.endswith('.counts')}
    if not names or names != counts:
        raise ValueError('imatrix GGUF requires matching .in_sum2 and .counts tensors')
    return {name: _normalized(tensors[name + '.in_sum2'], tensors[name + '.counts'], name) for name in sorted(names)}


def load_imatrix(path: Path, expected_sha256: str | None = None) -> ImatrixData:
    before = file_sha256(path)
    if expected_sha256 is not None and before != expected_sha256:
        raise ValueError('imatrix SHA-256 mismatch; the pinned source has changed')
    with path.open('rb') as stream:
        magic = stream.read(4)
    try:
        entries = _gguf(path) if magic == b'GGUF' else _legacy(path.read_bytes())
    except (IndexError, KeyError, OverflowError, UnicodeError, struct.error) as error:
        raise ValueError(f'malformed imatrix: {error}') from error
    if file_sha256(path) != before:
        raise ValueError('imatrix changed while being read')
    return ImatrixData(before, 'gguf' if magic == b'GGUF' else 'legacy', entries)


_LLAMA_NAMES = {
    'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k', 'self_attn.v_proj': 'attn_v',
    'self_attn.o_proj': 'attn_output', 'mlp.gate_proj': 'ffn_gate',
    'mlp.up_proj': 'ffn_up', 'mlp.down_proj': 'ffn_down',
}


def resolve_imatrix(
    config: ImatrixConfig, inventory: ModelInventory, groups: tuple[SharedInputGroupConfig, ...] = (),
) -> tuple[dict[LayerId, torch.Tensor], dict[str, object]]:
    data = load_imatrix(Path(config.path), config.sha256)
    explicit = {item.layer: item.tensor for item in config.tensor_map}
    if len(explicit) != len(config.tensor_map):
        raise ValueError('duplicate imatrix layer mappings')
    resolved: dict[LayerId, torch.Tensor] = {}
    mappings: dict[str, str] = {}
    used: set[str] = set()
    for block in inventory.blocks:
        for layer in block.quantizable_layers:
            key = f'{block.block.index}:{layer.layer.path}'
            stem = _LLAMA_NAMES.get(layer.layer.path)
            aliases = [layer.weight.source_key]
            if stem is not None:
                aliases.extend((f'blk.{block.block.index}.{stem}.weight', f'blk.{block.block.index}.{stem}'))
            candidates = [explicit[key]] if key in explicit else list(dict.fromkeys(
                name for name in aliases if name in data.entries
            ))
            if len(candidates) != 1 or candidates[0] not in data.entries:
                raise ValueError(f'missing or ambiguous imatrix entry for {key}; set tensor_map explicitly')
            name = candidates[0]
            value = data.entries[name]
            if tuple(value.shape) != (1, layer.in_features):
                raise ValueError(f'imatrix shape mismatch for {key}: {tuple(value.shape)}; '
                                 f'expected one dense input vector of width {layer.in_features}')
            if name in used:
                raise ValueError(f'imatrix entry mapped to multiple layers: {name}')
            used.add(name)
            # These are mean squared activations, not their square roots.
            # The ADMM solver takes square roots when preconditioning weights.
            resolved[layer.layer] = value[0].contiguous()
            mappings[key] = name
    if set(explicit) - set(mappings):
        raise ValueError('imatrix tensor_map contains layers outside the quantized inventory')
    for block in inventory.blocks:
        for group in groups:
            members = [resolved[LayerId(block.block, member)] for member in group.members]
            if any(not torch.allclose(members[0], value, rtol=1e-4, atol=1e-6) for value in members[1:]):
                raise ValueError(f'imatrix shared-input vectors disagree in block {block.block.index}, {group.name}')
    return resolved, {'format': data.format, 'sha256': data.sha256, 'path': config.path,
                      'normalization': 'llama-cpp-sums-over-counts', 'tensor_map': mappings,
                      'unused_entries': sorted(set(data.entries) - used)}
