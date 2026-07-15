from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention as TransformersGemma3Attention,
)

from nanoquant.runtime.backend import (
    BackendCapabilities,
    PreparedLayer,
    QuantizedLinearSpec,
    SupportResult,
    WorkloadSpec,
)
from nanoquant.runtime.planning import (
    plan_execution_workloads,
    prepare_execution_workloads,
)
from nanoquant.runtime.torch_model import (
    PreparedGemma3Attention,
    PreparedRMSNorm,
    bind_fused_decode_rope,
    bind_prepared_linears,
    bind_prepared_rms_norms,
    execution_workload,
    transformers_decoder_module_paths,
)


@dataclass(frozen=True)
class State:
    spec: QuantizedLinearSpec


class Backend:
    name = "test"
    version = "1"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            ("nanoquant-v1",), ("cpu",), ("float32",), ("float32",), ("float32",), (),
            ("prefill", "decode"), False, False, True,
        )

    def supports(self, op: QuantizedLinearSpec, workload: WorkloadSpec) -> SupportResult:
        return SupportResult.accepted()

    def prepare(self, state: State, device: str) -> PreparedLayer:
        return PreparedLayer(self.name, self.version, state.spec, None)

    def linear(self, value: torch.Tensor, layer: PreparedLayer) -> torch.Tensor:
        return value[..., : layer.spec.out_features] + 1


class Shell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].proj = nn.Linear(3, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0].proj(value)


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.linspace(-0.1, 0.1, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = value.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * (1.0 + self.weight.float())).type_as(value)


class NormShell(nn.Module):
    def __init__(self, count: int = 1) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.norms = nn.ModuleList(Gemma3RMSNorm(8) for _ in range(count))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for norm in self.model.norms:
            value = norm(value)
        return value


class Gemma3Attention(nn.Module):
    def __init__(self, *, valid: bool = True) -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            if valid or name != "k_norm":
                setattr(self, name, nn.Identity())
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.layer_idx = 0
        self.head_dim = 256
        self.num_key_value_groups = 4
        self.scaling = 0.0625
        self.attention_dropout = 0.0
        self.is_causal = True
        self.is_sliding = True
        self.attn_logit_softcapping = None
        self.sliding_window = 512


class AttentionShell(nn.Module):
    def __init__(self, *modules: Gemma3Attention) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.attentions = nn.ModuleList(modules)


def _plans():
    spec = QuantizedLinearSpec("blocks.0.proj", "nanoquant-v1", 3, 2, 2, "float32", "float32")
    backend = Backend()
    plans = plan_execution_workloads(
        (spec,),
        prefill=WorkloadSpec("prefill", "cpu", "float32", 2, 3, True),
        decode=WorkloadSpec("decode", "cpu", "float32", 2, 1, True),
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )
    return prepare_execution_workloads(plans, {spec.name: State(spec)}, (backend,), "cpu")


def test_prepared_model_linear_selects_prefill_and_decode_plans() -> None:
    shell = Shell()
    plans = _plans()
    paths = transformers_decoder_module_paths(("blocks.0.proj",))
    assert bind_prepared_linears(shell, plans, paths) == 1
    assert not tuple(shell.model.layers[0].proj.parameters())

    with execution_workload("prefill"):
        assert torch.equal(shell(torch.zeros(2, 3, 3)), torch.ones(2, 3, 2))
    with execution_workload("decode"):
        assert torch.equal(shell(torch.zeros(2, 1, 3)), torch.ones(2, 1, 2))
    with pytest.raises(RuntimeError, match="execution_workload"):
        shell(torch.zeros(2, 1, 3))


def test_binding_validates_every_target_before_mutation() -> None:
    shell = Shell()
    original = shell.model.layers[0].proj
    with pytest.raises(ValueError, match="unavailable"):
        bind_prepared_linears(shell, _plans(), {"blocks.0.proj": "model.layers.0.missing"})
    assert shell.model.layers[0].proj is original


def test_transformers_path_mapping_is_exact() -> None:
    assert transformers_decoder_module_paths(("blocks.12.mlp.down_proj",)) == {
        "blocks.12.mlp.down_proj": "model.layers.12.mlp.down_proj"
    }
    with pytest.raises(ValueError, match="block-scoped"):
        transformers_decoder_module_paths(("layers.0.proj",))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_prepared_rms_norm_matches_gemma_formula(dtype: torch.dtype) -> None:
    shell = NormShell()
    value = torch.linspace(-2.0, 2.0, 16, dtype=dtype).reshape(2, 1, 8)
    expected = shell(value)

    assert bind_prepared_rms_norms(shell) == 1
    assert isinstance(shell.model.norms[0], PreparedRMSNorm)
    assert not tuple(shell.parameters())
    actual = shell(value)

    assert torch.equal(actual, expected)


def test_rms_norm_binding_validates_every_target_before_mutation() -> None:
    shell = NormShell(count=2)
    first = shell.model.norms[0]
    second = shell.model.norms[1]
    second.eps = 0.0

    with pytest.raises(ValueError, match="epsilon"):
        bind_prepared_rms_norms(shell)

    assert shell.model.norms[0] is first
    assert shell.model.norms[1] is second


def test_rms_norm_binding_is_a_noop_for_other_model_families() -> None:
    model = nn.Sequential(nn.LayerNorm(8))
    assert bind_prepared_rms_norms(model) == 0
    assert isinstance(model[0], nn.LayerNorm)


def test_fused_decode_rope_binding_replaces_pinned_attention() -> None:
    shell = AttentionShell(Gemma3Attention())

    assert bind_fused_decode_rope(shell) == 1
    assert isinstance(shell.model.attentions[0], PreparedGemma3Attention)


def test_fused_decode_rope_binding_validates_before_mutation() -> None:
    shell = AttentionShell(Gemma3Attention(), Gemma3Attention(valid=False))
    original = shell.model.attentions[0]

    with pytest.raises(ValueError, match="missing k_norm"):
        bind_fused_decode_rope(shell)

    assert shell.model.attentions[0] is original


def test_prepared_attention_preserves_cpu_prefill_fallback() -> None:
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=8,
    )
    source = TransformersGemma3Attention(config, layer_idx=0).eval()
    prepared = PreparedGemma3Attention(source).eval()
    generator = torch.Generator().manual_seed(20260715)
    hidden_states = torch.randn(1, 3, 16, generator=generator)
    cosine = torch.randn(1, 3, 4, generator=generator)
    sine = torch.randn(1, 3, 4, generator=generator)

    expected = source(hidden_states, (cosine, sine), None)
    actual = prepared(hidden_states, (cosine, sine), None)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
