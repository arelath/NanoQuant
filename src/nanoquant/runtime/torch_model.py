"""PyTorch model-shell bindings for prepared NanoQuant execution plans."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, cast

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


class PreparedTiedEmbedding(nn.Module):
    """Store the tied BF16 table while preserving the F32 lookup boundary."""

    weight: nn.Parameter
    embed_scale: torch.Tensor

    def __init__(
        self,
        weight: nn.Parameter,
        *,
        padding_idx: int | None,
        embed_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("prepared tied embedding requires a BF16 matrix")
        self.weight = weight
        self.num_embeddings = weight.shape[0]
        self.embedding_dim = weight.shape[1]
        self.padding_idx = padding_idx
        self.register_buffer("embed_scale", embed_scale.detach().float(), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.device.type == "cuda":
            from nanoquant.runtime.cuda_kernels import launch_bfloat16_embedding

            return launch_bfloat16_embedding(input_ids, self.weight, self.embed_scale)
        embedded = F.embedding(input_ids, self.weight, padding_idx=self.padding_idx)
        return embedded.float() * self.embed_scale


class PreparedTiedOutputProjection(nn.Module):
    """Project through the native BF16 tied table with F32 accumulation."""

    weight: nn.Parameter

    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError("prepared tied output projection requires a BF16 matrix")
        self.weight = weight
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.bias = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        from nanoquant.runtime.cuda_kernels import launch_bfloat16_output_projection

        return launch_bfloat16_output_projection(value, self.weight)


def bind_native_bfloat16_tied_projection(model: nn.Module) -> int:
    """Keep Gemma's tied table in its persisted BF16 representation."""

    embedding = model.get_submodule("model.embed_tokens")
    head = getattr(model, "lm_head", None)
    embedding_weight = getattr(embedding, "weight", None)
    head_weight = getattr(head, "weight", None)
    embed_scale = getattr(embedding, "embed_scale", None)
    if not isinstance(head, nn.Module):
        raise ValueError("tied projection model contains no lm_head module")
    if not isinstance(embedding_weight, nn.Parameter):
        raise ValueError("tied projection embedding contains no parameter weight")
    if head_weight is not embedding_weight:
        raise ValueError("tied projection model does not share embedding and output weights")
    if not isinstance(embed_scale, torch.Tensor):
        raise ValueError("tied projection embedding contains no scale buffer")
    source_weight = cast(torch.Tensor, embedding_weight)
    native_value = (
        torch.empty(
            source_weight.shape,
            dtype=torch.bfloat16,
            device=source_weight.device,
        )
        if source_weight.is_meta
        else source_weight.detach().to(torch.bfloat16)
    )
    native_weight = nn.Parameter(native_value, requires_grad=False)
    prepared_embedding = PreparedTiedEmbedding(
        native_weight,
        padding_idx=getattr(embedding, "padding_idx", None),
        embed_scale=embed_scale,
    )
    prepared_head = PreparedTiedOutputProjection(native_weight)
    prepared_embedding.train(embedding.training)
    prepared_head.train(head.training)
    text_model = model.get_submodule("model")
    text_model.add_module("embed_tokens", prepared_embedding)
    model.add_module("lm_head", prepared_head)
    return 1


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


class PreparedGemma3Attention(nn.Module):
    """Pinned eager Gemma3 attention with a decode-only fused RoPE."""

    q_proj: nn.Module
    k_proj: nn.Module
    v_proj: nn.Module
    o_proj: nn.Module
    q_norm: nn.Module
    k_norm: nn.Module
    config: Any
    layer_idx: int
    head_dim: int
    num_key_value_groups: int
    scaling: float
    attention_dropout: float
    is_causal: bool
    is_sliding: bool
    attn_logit_softcapping: float | None
    sliding_window: int | None

    def __init__(self, source: nn.Module, *, fuse_decode_attention: bool = False) -> None:
        super().__init__()
        required_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
        for name in required_modules:
            child = getattr(source, name, None)
            if not isinstance(child, nn.Module):
                raise ValueError(f"Gemma3 attention module is missing {name}")
            setattr(self, name, child)
        for name in (
            "config",
            "layer_idx",
            "head_dim",
            "num_key_value_groups",
            "scaling",
            "attention_dropout",
            "is_causal",
            "is_sliding",
            "attn_logit_softcapping",
            "sliding_window",
        ):
            if not hasattr(source, name):
                raise ValueError(f"Gemma3 attention contract is missing {name}")
            setattr(self, name, getattr(source, name))
        if getattr(self.config, "_attn_implementation", None) != "eager":
            raise ValueError("Gemma3 decode RoPE requires eager attention")
        self.fuse_decode_attention = fuse_decode_attention
        self.train(source.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Any | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cosine, sine = position_embeddings
        if (
            tuple(query_states.shape) == (1, 4, 1, 256)
            and tuple(key_states.shape) == (1, 1, 1, 256)
            and query_states.dtype == torch.float32
            and key_states.dtype == torch.float32
            and query_states.device.type == "cuda"
        ):
            from nanoquant.runtime.cuda_kernels import launch_decode_rope

            query_states, key_states = launch_decode_rope(
                query_states,
                key_states,
                cosine,
                sine,
            )
        else:
            from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(  # type: ignore[no-untyped-call]
                query_states,
                key_states,
                cosine,
                sine,
            )
        if past_key_value is not None:
            cache_kwargs = {
                "sin": sine,
                "cos": cosine,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(query_states)
        if (
            self.fuse_decode_attention
            and not self.training
            and not kwargs.get("output_attentions", False)
            and self.attn_logit_softcapping is None
            and tuple(query_states.shape) == (1, 4, 1, 256)
            and key_states.ndim == 4
            and key_states.shape[:2] == (1, 1)
            and key_states.shape[3] == 256
            and value_states.shape == key_states.shape
            and key_states.shape[2] <= 64
            and query_states.dtype == torch.float32
            and key_states.dtype == torch.float32
            and value_states.dtype == torch.float32
            and query_states.device.type == "cuda"
            and query_states.is_contiguous()
            and key_states.is_contiguous()
            and value_states.is_contiguous()
            and (
                attention_mask is None
                or (
                    tuple(attention_mask.shape) == (1, 1, 1, key_states.shape[2])
                    and attention_mask.is_contiguous()
                )
            )
        ):
            from nanoquant.runtime.cuda_kernels import launch_decode_attention

            attention_output = launch_decode_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
            )
            attention_weights = None
        else:
            from transformers.models.gemma3.modeling_gemma3 import eager_attention_forward

            attention_output, attention_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output), attention_weights


def bind_fused_decode_rope(
    model: nn.Module,
    *,
    fuse_decode_attention: bool = False,
) -> int:
    """Bind the pinned decode-only RoPE without changing eager prefill."""

    replacements: list[tuple[nn.Module, str, PreparedGemma3Attention]] = []
    for path, module in tuple(model.named_modules()):
        if module.__class__.__name__ != "Gemma3Attention":
            continue
        config = getattr(module, "config", None)
        if (
            getattr(module, "head_dim", None) != 256
            or getattr(module, "num_key_value_groups", None) != 4
            or getattr(config, "_attn_implementation", None) != "eager"
        ):
            continue
        parent_path, separator, attribute = path.rpartition(".")
        if not separator or not parent_path or not attribute:
            raise ValueError(f"Gemma3 attention module path must be dotted: {path!r}")
        replacements.append(
            (
                model.get_submodule(parent_path),
                attribute,
                PreparedGemma3Attention(
                    module,
                    fuse_decode_attention=fuse_decode_attention,
                ),
            )
        )
    for parent, attribute, replacement in replacements:
        setattr(parent, attribute, replacement)
    return len(replacements)


class PreparedGemma3DecoderLayer(nn.Module):
    """Pinned Gemma3 layer that elides an identity short-context mask."""

    self_attn: nn.Module
    mlp: nn.Module
    input_layernorm: nn.Module
    post_attention_layernorm: nn.Module
    pre_feedforward_layernorm: nn.Module
    post_feedforward_layernorm: nn.Module
    config: Any
    hidden_size: int
    layer_idx: int
    is_sliding: bool
    sliding_window: int

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        required_modules = (
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        )
        for name in required_modules:
            child = getattr(source, name, None)
            if not isinstance(child, nn.Module):
                raise ValueError(f"Gemma3 decoder layer is missing {name}")
            setattr(self, name, child)
        for name in ("config", "hidden_size", "layer_idx", "is_sliding", "sliding_window"):
            if not hasattr(source, name):
                raise ValueError(f"Gemma3 decoder layer contract is missing {name}")
            setattr(self, name, getattr(source, name))
        if not self.is_sliding:
            raise ValueError("short sliding-mask optimization requires a sliding layer")
        if getattr(self.config, "_attn_implementation", None) != "eager":
            raise ValueError("short sliding-mask optimization requires eager attention")
        if self.sliding_window <= 0:
            raise ValueError("Gemma3 sliding window must be positive")
        self.train(source.training)

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor | None,
        last_cache_position: int,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if cache_position is None:
            raise ValueError("Gemma3 sliding attention requires cache_position")
        effective_seq_len = max(cache_position.shape[0], self.sliding_window)
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask[:, -effective_seq_len:]
        # torch.tril(ones, diagonal=-window) is entirely false while both
        # matrix dimensions fit inside the window. The subsequent where and
        # offset slice therefore return this exact tensor unchanged.
        if (
            attention_mask.ndim == 4
            and attention_mask.shape[-2] <= self.sliding_window
            and attention_mask.shape[-1] <= self.sliding_window
            and last_cache_position <= self.sliding_window
        ):
            return attention_mask
        minimum = torch.finfo(attention_mask.dtype).min
        sliding_window_mask = torch.tril(
            torch.ones_like(attention_mask, dtype=torch.bool),
            diagonal=-self.sliding_window,
        )
        result = torch.where(sliding_window_mask, minimum, attention_mask)
        offset = max(0, last_cache_position - effective_seq_len)
        return result[:, :, :, offset : offset + effective_seq_len]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.Tensor | None = None,
        last_cache_position: int = 0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        attention_mask = self._prepare_attention_mask(
            attention_mask,
            cache_position,
            last_cache_position,
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        position_embeddings = (
            position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global
        )
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        outputs: tuple[Any, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


def bind_short_sliding_masks(model: nn.Module) -> int:
    """Elide identity Gemma3 masks only for contexts inside the window."""

    replacements: list[tuple[nn.Module, str, PreparedGemma3DecoderLayer]] = []
    for path, module in tuple(model.named_modules()):
        if module.__class__.__name__ != "Gemma3DecoderLayer":
            continue
        config = getattr(module, "config", None)
        if (
            not getattr(module, "is_sliding", False)
            or getattr(config, "_attn_implementation", None) != "eager"
        ):
            continue
        parent_path, separator, attribute = path.rpartition(".")
        if not separator or not parent_path or not attribute:
            raise ValueError(f"Gemma3 decoder layer path must be dotted: {path!r}")
        replacements.append(
            (model.get_submodule(parent_path), attribute, PreparedGemma3DecoderLayer(module))
        )
    for parent, attribute, replacement in replacements:
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
