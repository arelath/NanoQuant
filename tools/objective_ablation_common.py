"""GGUF-backed, fixed-sign scale QAT helpers for objective-only diagnostics."""

from __future__ import annotations

from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import torch
from probe_residual_stage_functional import PROJECTIONS
from weight_image_common import _float_tensor, load_gguf_module, tensor_map, unpack_sign_words


def bf16_ste(value: torch.Tensor) -> torch.Tensor:
    """Use persisted BF16 values with an FP32 master-scale derivative."""
    return value + (value.bfloat16().float() - value).detach()


class ScaleUnit(torch.nn.Module):
    def __init__(self, left: torch.Tensor, right: torch.Tensor, pre: torch.Tensor,
                 mid: torch.Tensor, post: torch.Tensor, indices: torch.Tensor, salient: torch.Tensor):
        super().__init__()
        self.register_buffer("left", left.to(torch.int8))
        self.register_buffer("right", right.to(torch.int8))
        self.register_buffer("indices", indices.long())
        self.register_buffer("salient", salient.float())
        self.pre = torch.nn.Parameter(pre.float().clone())
        self.mid = torch.nn.Parameter(mid.float().clone())
        self.post = torch.nn.Parameter(post.float().clone())

    def dense_weight(self) -> torch.Tensor:
        pre = bf16_ste(self.pre).index_fill(0, self.indices, 0)
        left = self.left.float() * bf16_ste(self.post)[:, None]
        right = self.right.float() * bf16_ste(self.mid)[:, None] * pre[None, :]
        return (left @ right).index_copy(1, self.indices, self.salient)

    def scales(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().cpu().bfloat16() for name in ("pre", "mid", "post")}


class UnitSlice(torch.nn.Module):
    def __init__(self, unit: ScaleUnit, start: int, end: int):
        super().__init__()
        self.unit = unit
        self.start, self.end = start, end

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.unit.dense_weight()[self.start:self.end].to(value.dtype))


def read_units(path: Path, llama_root: Path) -> dict[str, ScaleUnit]:
    gguf = load_gguf_module(llama_root)
    tensors = tensor_map(gguf.GGUFReader(str(path)))
    units = {}
    for key in sorted(tensors):
        if not key.endswith(".nq_v"):
            continue
        base = key.removesuffix(".nq_v")

        def value(suffix: str, base: str = base) -> torch.Tensor:
            return torch.from_numpy(_float_tensor(tensors[base + "." + suffix], gguf).copy())

        pre, mid, post = (value("nq_scale_" + name).reshape(-1) for name in ("pre", "mid", "post"))
        if any(getattr(tensors[base + ".nq_scale_" + name].tensor_type, "name", "") != "BF16"
               for name in ("pre", "mid", "post")):
            raise ValueError("pinned 060 factor-scale storage is expected to be BF16")
        left = torch.from_numpy(unpack_sign_words(np.asarray(tensors[base + ".nq_u"].data), len(post), len(mid)))
        right = torch.from_numpy(unpack_sign_words(np.asarray(tensors[key].data), len(mid), len(pre)))
        indices = value("nq_salient_idx").long().reshape(-1)
        salient = value("nq_salient_weight").float() * value("nq_salient_scale").float().reshape(1, -1)
        units[base] = ScaleUnit(left, right, pre, mid, post, indices, salient)
    if len(units) != 130:
        raise ValueError("expected all 130 pinned 060 shared/single factor units")
    return units


def bindings(model: torch.nn.Module, units: dict[str, ScaleUnit]):
    """Yield unique projection ownership while preserving shared QKV scales."""
    for block_index, block in enumerate(model.model.layers):
        qkv = units[f"blk.{block_index}.attn_qkv"]
        start = 0
        for name in ("q", "k", "v"):
            module = block.get_submodule(f"self_attn.{name}_proj")
            end = start + module.out_features
            yield block, f"self_attn.{name}_proj", qkv, start, end
            start = end
        if start != len(qkv.post):
            raise ValueError("shared QKV dimensions changed")
        for source, module_path in PROJECTIONS.items():
            if source in {"attn_q", "attn_k", "attn_v"}:
                continue
            unit = units[f"blk.{block_index}.{source}"]
            yield block, module_path, unit, 0, len(unit.post)


def install_trainable(model: torch.nn.Module, units: dict[str, ScaleUnit]) -> None:
    for block, path, unit, start, end in tuple(bindings(model, units)):
        parent, _, name = path.rpartition(".")
        setattr(block.get_submodule(parent), name, UnitSlice(unit, start, end))


def install_dense(model: torch.nn.Module, units: dict[str, ScaleUnit]) -> None:
    # The model must contain dense nn.Linear projections here. Shared units are
    # materialized once, and every arm is evaluated after BF16 persistence.
    cached = {}
    for block, path, unit, start, end in bindings(model, units):
        if id(unit) not in cached:
            cached[id(unit)] = unit.dense_weight().detach().bfloat16()
        module = block.get_submodule(path)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("dense evaluation requires restored linear modules")
        module.weight.copy_(cached[id(unit)][start:end].to(module.weight.device))
    del cached


def load_scales(units: dict[str, ScaleUnit], values: dict[str, torch.Tensor]) -> None:
    expected = {base + "." + name for base in units for name in ("pre", "mid", "post")}
    if set(values) != expected:
        raise ValueError("scale checkpoint does not match all shared/single units")
    with torch.no_grad():
        for base, unit in units.items():
            for name in ("pre", "mid", "post"):
                source, target = values[base + "." + name], getattr(unit, name)
                if source.shape != target.shape or not torch.isfinite(source).all():
                    raise ValueError("scale checkpoint shape/finiteness mismatch")
                target.copy_(source)


def scale_state(units: dict[str, ScaleUnit], *, persisted: bool = True) -> dict[str, torch.Tensor]:
    return {base + "." + name: (getattr(unit, name).detach().cpu().bfloat16() if persisted
                                else getattr(unit, name).detach().cpu().clone())
            for base, unit in units.items() for name in ("pre", "mid", "post")}


def normalized_hidden_loss(student: tuple[torch.Tensor, ...], teacher: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if len(student) != len(teacher) or len(student) < 2:
        raise ValueError("teacher/student hidden-state coverage mismatch")
    # Exclude embedding state; include states after each decoder layer, with
    # final norm applied to the last one as returned by Transformers.
    return torch.stack([(s.float() - t.float()).square().mean() / t.float().square().mean().clamp_min(1e-8)
                        for s, t in zip(student[1:], teacher[1:], strict=True)]).mean()
