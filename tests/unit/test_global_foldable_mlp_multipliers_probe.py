from __future__ import annotations

import torch

from nanoquant.application.foldable_mlp_multipliers import (
    FoldableMultiplierLinear,
    InstalledMultipliers,
    family_identity_penalty,
    fold_global_mlp_multipliers,
    seed_global_mlp_multipliers,
)
from nanoquant.application.layers import FactorizedReferenceLinear


class _Mlp(torch.nn.Module):
    def __init__(self, module: FactorizedReferenceLinear) -> None:
        super().__init__()
        self.gate_proj = module


class _Block(torch.nn.Module):
    def __init__(self, module: FactorizedReferenceLinear) -> None:
        super().__init__()
        self.mlp = _Mlp(module)


class _Stack(torch.nn.Module):
    def __init__(self, module: FactorizedReferenceLinear) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(module)])


class _Model(torch.nn.Module):
    def __init__(self, module: FactorizedReferenceLinear) -> None:
        super().__init__()
        self.model = _Stack(module)


def _linear() -> FactorizedReferenceLinear:
    return FactorizedReferenceLinear(
        torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]),
        torch.tensor([0.5, 0.75, 1.25]),
        torch.tensor([0.8, 1.1]),
        torch.tensor([1.2, 0.9]),
        bias=torch.tensor([0.3, -0.2]),
        outlier_indices=torch.tensor([1]),
        outlier_values=torch.tensor([[0.4], [-0.6]]),
        patch_left=torch.tensor([[0.2], [-0.1]]),
        patch_right=torch.tensor([[0.3, -0.2, 0.5]]),
    )


def test_foldable_multiplier_preserves_bias_and_folds_all_terms() -> None:
    module = _linear()
    wrapper = FoldableMultiplierLinear(
        module,
        input_family="down_input",
        output_family="down_output",
    )
    with torch.no_grad():
        assert wrapper.log_input_multiplier is not None
        assert wrapper.log_output_multiplier is not None
        wrapper.log_input_multiplier.copy_(torch.log(torch.tensor([1.25, 0.8, 1.1])))
        wrapper.log_output_multiplier.copy_(torch.log(torch.tensor([0.7, 1.4])))
    value = torch.tensor([[0.2, -0.4, 0.9], [1.0, 0.3, -0.2]])
    expected = wrapper(value)
    model = _Model(module)
    model.model.layers[0].mlp.gate_proj = wrapper
    installed = InstalledMultipliers(
        {(0, "mlp.gate_proj"): wrapper},
        {
            "down_input": (wrapper.log_input_multiplier,),
            "down_output": (wrapper.log_output_multiplier,),
        },
    )
    tensors, replaced = fold_global_mlp_multipliers(model, installed)
    actual = model.model.layers[0].mlp.gate_proj(value)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert set(tensors) == {
        "model.layers.0.mlp.gate_proj.scale_pre",
        "model.layers.0.mlp.gate_proj.scale_post",
        "model.layers.0.mlp.gate_proj.outlier_values",
        "model.layers.0.mlp.gate_proj.patch_left",
        "model.layers.0.mlp.gate_proj.patch_right",
    }
    assert replaced == sum(value.numel() * value.element_size() for value in tensors.values())


def test_identity_penalty_weights_families_equally() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    penalty = family_identity_penalty({"large": (first,), "small": (second,)})
    assert float(penalty.detach()) == 2.5


def test_sparse_initializer_seeds_named_axis_and_leaves_other_axis_identity() -> None:
    module = _linear()
    wrapper = FoldableMultiplierLinear(
        module,
        input_family="down_input",
        output_family="down_output",
    )
    assert wrapper.log_input_multiplier is not None
    assert wrapper.log_output_multiplier is not None
    installed = InstalledMultipliers(
        {(0, "mlp.gate_proj"): wrapper},
        {
            "down_input": (wrapper.log_input_multiplier,),
            "down_output": (wrapper.log_output_multiplier,),
        },
    )

    consumed = seed_global_mlp_multipliers(
        installed,
        {
            "model.layers.0.mlp.gate_proj.output_log_multiplier": torch.log(
                torch.tensor([0.75, 1.5])
            )
        },
        log_limit=torch.log(torch.tensor(4.0)).item(),
    )

    assert consumed == ("model.layers.0.mlp.gate_proj.output_log_multiplier",)
    torch.testing.assert_close(wrapper.log_input_multiplier, torch.zeros(3))
    torch.testing.assert_close(wrapper.log_output_multiplier.exp(), torch.tensor([0.75, 1.5]))
