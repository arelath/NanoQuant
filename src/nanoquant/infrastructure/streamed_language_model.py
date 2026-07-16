"""Bounded-device full-model forwards over a host-resident Transformers shell."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.infrastructure.model_adapters import TransformersModelAdapter


def _to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _suffix_modules(model: nn.Module, adapter: TransformersModelAdapter) -> tuple[nn.Module, ...]:
    head = adapter.lm_head(model)
    if adapter.family == "opt":
        decoder = getattr(getattr(model, "model", None), "decoder", None)
        if not isinstance(decoder, nn.Module):
            raise ValueError("streamed OPT evaluation requires a decoder module")
        modules = [getattr(decoder, "final_layer_norm", None), getattr(decoder, "project_out", None), head]
    else:
        base = getattr(model, "model", None)
        norm = getattr(base, "norm", None)
        if not isinstance(norm, nn.Module):
            raise ValueError("streamed evaluation requires a final normalization module")
        modules = [norm, head]
    result: list[nn.Module] = []
    for module in modules:
        if isinstance(module, nn.Module) and all(module is not existing for existing in result):
            result.append(module)
    return tuple(result)


class BlockStreamedCausalLM(nn.Module):
    """Expose a normal causal-LM forward while bounding CUDA weight residency.

    The source model remains on pageable CPU memory. The exact Transformers
    prefix call captures per-request positional/attention metadata, decoder
    blocks visit the compute device one at a time, and the final norm/head move
    only for the suffix. This changes placement, not the model computation.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: TransformersModelAdapter,
        device: str,
    ) -> None:
        super().__init__()
        if next(model.parameters(), torch.empty(0)).device.type != "cpu":
            raise ValueError("block-streamed evaluation requires a host-resident model")
        self.model = model
        self.adapter = adapter
        self.compute_device = torch.device(device)
        self.config = cast(Any, model).config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
        **kwargs: object,
    ) -> SimpleNamespace:
        if use_cache:
            raise ValueError("block-streamed quality evaluation does not support a KV cache")
        if kwargs:
            raise ValueError(f"unsupported block-streamed forward arguments: {sorted(kwargs)}")
        layers = self.adapter.get_decoder_layers(self.model)
        if not layers:
            raise ValueError("block-streamed evaluation requires at least one decoder block")
        host_tokens = input_ids.to("cpu")
        host_mask = None if attention_mask is None else attention_mask.to("cpu")

        def prefix_call() -> object:
            return cast(Any, self.model)(
                input_ids=host_tokens,
                attention_mask=host_mask,
                use_cache=False,
            )

        captured = capture_prefix_invocations(layers[0], (prefix_call,))[0]
        if not captured.positional or not isinstance(captured.positional[0], torch.Tensor):
            raise ValueError("streamed model prefix did not provide decoder hidden states")
        hidden = captured.positional[0].to(self.compute_device)
        metadata = cast(dict[str, object], _to_device(captured.keyword, self.compute_device))

        with torch.no_grad():
            for block in layers:
                block.to(self.compute_device)
                try:
                    hidden = self.adapter.run_block(block, hidden, **metadata)
                finally:
                    block.to("cpu")
            suffix_modules = _suffix_modules(self.model, self.adapter)
            for module in suffix_modules:
                module.to(self.compute_device)
            try:
                logits = self.adapter.run_suffix(self.model, hidden)
            finally:
                for module in reversed(suffix_modules):
                    module.to("cpu")
        return SimpleNamespace(logits=logits)


__all__ = ["BlockStreamedCausalLM"]
