"""Load the causal text model from text-only or multimodal HF checkpoints."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.auto.configuration_auto import AutoConfig


def load_causal_language_model(
    snapshot: str | Path,
    *,
    torch_dtype: torch.dtype,
    attention_implementation: str,
    local_files_only: bool = False,
) -> nn.Module:
    """Load only the decoder language model onto CPU.

    Gemma 3 checkpoints above 1B are multimodal wrappers. Transformers cannot
    load those roots through ``AutoModelForCausalLM``. Loading the wrapper on
    CPU and detaching ``language_model`` keeps the vision tower out of CUDA and
    preserves the checkpoint's tied text weights.
    """

    config = AutoConfig.from_pretrained(snapshot, local_files_only=local_files_only)
    if str(getattr(config, "model_type", "")).lower() != "gemma3":
        return cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(
                snapshot,
                config=config,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
                attn_implementation=attention_implementation,
            ),
        )
    wrapper = cast(
        nn.Module,
        AutoModelForImageTextToText.from_pretrained(
            snapshot,
            config=config,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            attn_implementation=attention_implementation,
        ),
    )
    language_model = getattr(wrapper, "language_model", None)
    if not isinstance(language_model, nn.Module):
        raise TypeError("Gemma 3 multimodal checkpoint exposes no language_model module")
    # Remove the child reference before releasing the wrapper so only the text
    # model survives. The discarded vision/projector parameters never reach CUDA.
    cast(Any, wrapper)._modules.pop("language_model", None)
    del wrapper
    gc.collect()
    return language_model


__all__ = ["load_causal_language_model"]
