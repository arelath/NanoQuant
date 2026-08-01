"""Deployment-faithful, foldable MLP row and column multipliers."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.domain.linear_math import functional_factorized_linear, rescale_factorized_terms

MLP_PATHS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _module_parent(block: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    current = block
    for part in parts[:-1]:
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current, parts[-1]


def _child(parent: nn.Module, name: str) -> nn.Module:
    value = parent[name] if isinstance(parent, nn.ModuleDict) else getattr(parent, name, None)
    if not isinstance(value, nn.Module):
        raise KeyError(f"module child not found: {name}")
    return value


def _set_child(parent: nn.Module, name: str, value: nn.Module) -> None:
    if isinstance(parent, nn.ModuleDict):
        parent[name] = value
    else:
        setattr(parent, name, value)


def _decoder(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose a mutable decoder stack")
    return layers


class FoldableMultiplierLinear(nn.Module):
    """Execute the exact factor payload produced by later multiplier folding."""

    def __init__(
        self,
        base: FactorizedReferenceLinear,
        *,
        input_family: str | None = None,
        output_family: str | None = None,
    ) -> None:
        super().__init__()
        if input_family is None and output_family is None:
            raise ValueError("foldable multiplier wrapper needs at least one axis")
        self.base = base
        self.input_family = input_family
        self.output_family = output_family
        device = base.scale_pre.device
        self.log_input_multiplier = (
            None
            if input_family is None
            else nn.Parameter(torch.zeros(base.scale_pre.shape, device=device, dtype=torch.float32))
        )
        self.log_output_multiplier = (
            None
            if output_family is None
            else nn.Parameter(torch.zeros(base.scale_post.shape, device=device, dtype=torch.float32))
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scaled = rescale_factorized_terms(
            self.base.scale_pre,
            self.base.scale_post,
            input_multiplier=(
                None
                if self.log_input_multiplier is None
                else torch.exp(self.log_input_multiplier)
            ),
            output_multiplier=(
                None
                if self.log_output_multiplier is None
                else torch.exp(self.log_output_multiplier)
            ),
            outlier_indices=self.base.outlier_indices,
            outlier_values=self.base.outlier_values,
            patch_left=self.base.patch_left,
            patch_right=self.base.patch_right,
        )
        return functional_factorized_linear(
            value,
            self.base.left_binary,
            self.base.right_binary,
            scaled.scale_pre,
            self.base.scale_mid,
            scaled.scale_post,
            self.base.bias,
            self.base.outlier_indices,
            scaled.outlier_values,
            self.base.outlier_scales,
            scaled.patch_left,
            scaled.patch_right,
            scale_left_before_linear=True,
        )


@dataclass(frozen=True, slots=True)
class InstalledMultipliers:
    wrappers: dict[tuple[int, str], FoldableMultiplierLinear]
    families: dict[str, tuple[nn.Parameter, ...]]

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for values in self.families.values() for parameter in values)

    @property
    def named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (f"families.{family}.{index}", parameter)
            for family, parameters in sorted(self.families.items())
            for index, parameter in enumerate(parameters)
        )


def install_global_mlp_multipliers(model: nn.Module) -> InstalledMultipliers:
    wrappers: dict[tuple[int, str], FoldableMultiplierLinear] = {}
    families: dict[str, list[nn.Parameter]] = defaultdict(list)
    for block_index, block in enumerate(_decoder(model)):
        for path in MLP_PATHS:
            parent, name = _module_parent(block, path)
            base = _child(parent, name)
            if not isinstance(base, FactorizedReferenceLinear):
                raise TypeError(f"global MLP multiplier target is not factorized: {block_index}:{path}")
            if path == "mlp.gate_proj":
                wrapper = FoldableMultiplierLinear(base, output_family="gate_output")
            elif path == "mlp.up_proj":
                wrapper = FoldableMultiplierLinear(base, output_family="up_output")
            else:
                wrapper = FoldableMultiplierLinear(
                    base,
                    input_family="down_input",
                    output_family="down_output",
                )
            _set_child(parent, name, wrapper)
            wrappers[(block_index, path)] = wrapper
            if wrapper.log_input_multiplier is not None:
                assert wrapper.input_family is not None
                families[wrapper.input_family].append(wrapper.log_input_multiplier)
            if wrapper.log_output_multiplier is not None:
                assert wrapper.output_family is not None
                families[wrapper.output_family].append(wrapper.log_output_multiplier)
    return InstalledMultipliers(wrappers, {name: tuple(values) for name, values in families.items()})


def family_identity_penalty(families: dict[str, tuple[nn.Parameter, ...]]) -> torch.Tensor:
    if not families or any(not values for values in families.values()):
        raise ValueError("identity penalty requires non-empty multiplier families")
    means = [
        torch.cat(tuple(value.reshape(-1) for value in values)).square().mean()
        for values in families.values()
    ]
    return torch.stack(means).mean()


def multiplier_summary(installed: InstalledMultipliers, log_limit: float) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, parameters in sorted(installed.families.items()):
        values = torch.cat(tuple(parameter.detach().float().cpu().reshape(-1) for parameter in parameters))
        multipliers = values.exp()
        result[name] = {
            "count": values.numel(),
            "minimum": float(multipliers.min()),
            "q01": float(torch.quantile(multipliers, 0.01)),
            "median": float(torch.quantile(multipliers, 0.5)),
            "q99": float(torch.quantile(multipliers, 0.99)),
            "maximum": float(multipliers.max()),
            "lower_bound_count": int((values <= -log_limit).sum()),
            "upper_bound_count": int((values >= log_limit).sum()),
        }
    return result


def gradient_summary(installed: InstalledMultipliers) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, parameters in sorted(installed.families.items()):
        gradients = [parameter.grad for parameter in parameters]
        present = [gradient for gradient in gradients if isinstance(gradient, torch.Tensor)]
        result[name] = {
            "parameter_tensors": len(parameters),
            "gradient_tensors": len(present),
            "missing_gradient_tensors": len(parameters) - len(present),
            "nonfinite_gradient_tensors": sum(
                int(not bool(torch.isfinite(gradient).all())) for gradient in present
            ),
            "zero_gradient_tensors": sum(
                int(not bool(torch.count_nonzero(gradient))) for gradient in present
            ),
            "gradient_norm": math.sqrt(
                sum(float(gradient.detach().float().square().sum()) for gradient in present)
            ),
        }
    return result


def seed_global_mlp_multipliers(
    installed: InstalledMultipliers,
    tensors: dict[str, torch.Tensor],
    *,
    log_limit: float,
) -> tuple[str, ...]:
    """Load a sparse, semantic log-multiplier seed; omitted axes stay at identity."""

    prefix = "model.layers."
    suffixes = {
        ".input_log_multiplier": "input",
        ".output_log_multiplier": "output",
    }
    consumed: list[str] = []
    with torch.no_grad():
        for tensor_name, value in sorted(tensors.items()):
            if not tensor_name.startswith(prefix):
                raise ValueError(f"foldable MLP initializer tensor has an invalid name: {tensor_name}")
            axis = None
            logical = ""
            for suffix, candidate_axis in suffixes.items():
                if tensor_name.endswith(suffix):
                    axis = candidate_axis
                    logical = tensor_name.removeprefix(prefix).removesuffix(suffix)
                    break
            if axis is None:
                raise ValueError(f"foldable MLP initializer tensor has an invalid axis: {tensor_name}")
            block_text, path = logical.split(".", maxsplit=1)
            try:
                key = (int(block_text), path)
            except ValueError as error:
                raise ValueError(f"foldable MLP initializer block is invalid: {tensor_name}") from error
            wrapper = installed.wrappers.get(key)
            if wrapper is None:
                raise ValueError(f"foldable MLP initializer target is unavailable: {tensor_name}")
            parameter = (
                wrapper.log_input_multiplier if axis == "input" else wrapper.log_output_multiplier
            )
            if parameter is None:
                raise ValueError(f"foldable MLP initializer axis is unavailable: {tensor_name}")
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError(f"foldable MLP initializer tensor shape or dtype differs: {tensor_name}")
            if not torch.isfinite(value).all() or bool((value.abs() > log_limit).any()):
                raise ValueError(f"foldable MLP initializer tensor exceeds the configured bounds: {tensor_name}")
            parameter.copy_(value.to(device=parameter.device))
            consumed.append(tensor_name)
    return tuple(consumed)


def _component_values(prefix: str, module: FactorizedReferenceLinear) -> dict[str, torch.Tensor]:
    values = {
        f"{prefix}.scale_pre": module.scale_pre,
        f"{prefix}.scale_post": module.scale_post,
    }
    for name in ("outlier_values", "patch_left", "patch_right"):
        value = getattr(module, name)
        if isinstance(value, torch.Tensor):
            values[f"{prefix}.{name}"] = value
    return {name: value.detach().cpu().contiguous() for name, value in values.items()}


def fold_global_mlp_multipliers(
    model: nn.Module,
    installed: InstalledMultipliers,
) -> tuple[dict[str, torch.Tensor], int]:
    tensors: dict[str, torch.Tensor] = {}
    replaced_bytes = 0
    for (block_index, path), wrapper in sorted(installed.wrappers.items()):
        input_multiplier = (
            None
            if wrapper.log_input_multiplier is None
            else wrapper.log_input_multiplier.detach().exp()
        )
        output_multiplier = (
            None
            if wrapper.log_output_multiplier is None
            else wrapper.log_output_multiplier.detach().exp()
        )
        base = wrapper.base
        scaled = rescale_factorized_terms(
            base.scale_pre,
            base.scale_post,
            input_multiplier=input_multiplier,
            output_multiplier=output_multiplier,
            outlier_indices=base.outlier_indices,
            outlier_values=base.outlier_values,
            patch_left=base.patch_left,
            patch_right=base.patch_right,
        )
        with torch.no_grad():
            base.scale_pre.copy_(scaled.scale_pre)
            base.scale_post.copy_(scaled.scale_post)
            if base.outlier_values is not None and scaled.outlier_values is not None:
                base.outlier_values.copy_(scaled.outlier_values)
            if base.patch_left is not None and scaled.patch_left is not None:
                base.patch_left.copy_(scaled.patch_left)
            if base.patch_right is not None and scaled.patch_right is not None:
                base.patch_right.copy_(scaled.patch_right)
        parent, name = _module_parent(_decoder(model)[block_index], path)
        _set_child(parent, name, base)
        values = _component_values(f"model.layers.{block_index}.{path}", base)
        tensors.update(values)
        replaced_bytes += sum(value.numel() * value.element_size() for value in values.values())
    return tensors, replaced_bytes


__all__ = [
    "FoldableMultiplierLinear",
    "InstalledMultipliers",
    "family_identity_penalty",
    "fold_global_mlp_multipliers",
    "gradient_summary",
    "install_global_mlp_multipliers",
    "multiplier_summary",
    "seed_global_mlp_multipliers",
]
