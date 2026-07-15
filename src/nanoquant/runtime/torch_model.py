"""PyTorch model-shell bindings for prepared NanoQuant execution plans."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from nanoquant.runtime.backend import WorkloadKind
from nanoquant.runtime.planning import PreparedExecutionPlans

_ACTIVE_WORKLOAD: ContextVar[WorkloadKind | None] = ContextVar(
    "nanoquant_runtime_active_workload", default=None
)


@contextmanager
def execution_workload(kind: WorkloadKind) -> Iterator[None]:
    """Select the already prepared plan for one complete model-shell forward."""

    if kind not in ("prefill", "decode"):
        raise ValueError(f"unsupported runtime execution workload: {kind}")
    token = _ACTIVE_WORKLOAD.set(kind)
    try:
        yield
    finally:
        _ACTIVE_WORKLOAD.reset(token)


class PreparedLinear(nn.Module):
    """Parameter-free module dispatching one canonical layer through paired plans."""

    def __init__(
        self,
        plans: PreparedExecutionPlans,
        layer_index: int,
        *,
        output_dtype: Literal["input", "backend"] = "input",
    ) -> None:
        super().__init__()
        if output_dtype not in ("input", "backend"):
            raise ValueError(f"unsupported prepared linear output dtype policy: {output_dtype}")
        self._plans = plans
        self._layer_index = layer_index
        self._output_dtype = output_dtype
        prefill = plans.prefill.dispatches[layer_index].layer.spec
        decode = plans.decode.dispatches[layer_index].layer.spec
        if prefill != decode:
            raise ValueError(f"paired prepared layer specifications differ: {prefill.name}")
        self.in_features = prefill.in_features
        self.out_features = prefill.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        kind = _ACTIVE_WORKLOAD.get()
        if kind is None:
            raise RuntimeError("NanoQuant prepared linear executed outside an execution_workload context")
        plan = self._plans.prefill if kind == "prefill" else self._plans.decode
        output = plan.linear_at(self._layer_index, value)
        return output.to(dtype=value.dtype) if self._output_dtype == "input" else output


class PreparedRMSNorm(nn.Module):
    """Inference-only Gemma RMSNorm using the fused native F32 kernel."""

    scale: torch.Tensor

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        if weight.ndim != 1 or not weight.dtype.is_floating_point:
            raise ValueError("prepared RMSNorm weight must be a floating-point vector")
        if eps <= 0.0:
            raise ValueError("prepared RMSNorm epsilon must be positive")
        self.register_buffer("scale", (1.0 + weight.detach().float()).contiguous())
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.scale.numel():
            raise ValueError("prepared RMSNorm input width differs from its scale")
        if value.dtype == torch.float32:
            return F.rms_norm(value, (self.scale.numel(),), self.scale, self.eps)
        # Preserve the pinned Gemma3 implementation outside the F32 performance
        # protocol instead of changing its cast/reduction behavior implicitly.
        output = value.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.scale).type_as(value)


def bind_prepared_rms_norms(model: nn.Module) -> int:
    """Replace every pinned Transformers Gemma3 RMSNorm after shell loading."""

    replacements: list[tuple[nn.Module, str, PreparedRMSNorm, bool]] = []
    for path, module in tuple(model.named_modules()):
        if module.__class__.__name__ != "Gemma3RMSNorm":
            continue
        parent_path, separator, attribute = path.rpartition(".")
        if not separator or not parent_path or not attribute:
            raise ValueError(f"Gemma3 RMSNorm module path must be dotted: {path!r}")
        weight = getattr(module, "weight", None)
        eps = getattr(module, "eps", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(eps, float):
            raise ValueError(f"Gemma3 RMSNorm contract differs: {path}")
        parent = model.get_submodule(parent_path)
        replacements.append(
            (parent, attribute, PreparedRMSNorm(weight, eps), module.training)
        )
    for parent, attribute, replacement, training in replacements:
        replacement.train(training)
        setattr(parent, attribute, replacement)
    return len(replacements)


def bind_prepared_linears(
    model: nn.Module,
    plans: PreparedExecutionPlans,
    module_paths: Mapping[str, str],
    *,
    output_dtype: Literal["input", "backend"] = "input",
) -> int:
    """Replace an exact planned linear inventory with parameter-free runtime modules."""

    names = tuple(item.plan.layer_name for item in plans.prefill.dispatches)
    if set(module_paths) != set(names):
        missing = sorted(set(names) - set(module_paths))
        unexpected = sorted(set(module_paths) - set(names))
        raise ValueError(
            f"runtime model module mapping differs from the plan: missing={missing}, unexpected={unexpected}"
        )
    resolved_paths = tuple(module_paths[name] for name in names)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("runtime model module paths must be unique")

    replacements: list[tuple[nn.Module, str, PreparedLinear]] = []
    for index, (name, path) in enumerate(zip(names, resolved_paths, strict=True)):
        parent_path, separator, attribute = path.rpartition(".")
        if not separator or not parent_path or not attribute:
            raise ValueError(f"runtime model module path must be dotted: {path!r}")
        try:
            parent = model.get_submodule(parent_path)
            existing = getattr(parent, attribute)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"runtime model module is unavailable for {name}: {path}") from error
        if not isinstance(existing, nn.Linear):
            raise ValueError(f"runtime model target is not a linear module for {name}: {path}")
        spec = plans.prefill.dispatches[index].layer.spec
        if (existing.in_features, existing.out_features) != (spec.in_features, spec.out_features):
            raise ValueError(f"runtime model linear dimensions differ for {name}: {path}")
        replacements.append((parent, attribute, PreparedLinear(plans, index, output_dtype=output_dtype)))

    # Resolve and validate the entire inventory before mutating the model.
    for parent, attribute, replacement in replacements:
        setattr(parent, attribute, replacement)
    return len(replacements)


def transformers_decoder_module_paths(layer_names: tuple[str, ...]) -> dict[str, str]:
    """Map canonical ``blocks.N`` names to Hugging Face decoder module paths."""

    result = {}
    for name in layer_names:
        prefix, separator, suffix = name.partition(".")
        if prefix != "blocks" or not separator:
            raise ValueError(f"canonical runtime layer is not block-scoped: {name}")
        index, separator, relative = suffix.partition(".")
        if not separator or not index.isdigit() or not relative:
            raise ValueError(f"canonical runtime layer path is invalid: {name}")
        result[name] = f"model.layers.{index}.{relative}"
    return result
