from types import SimpleNamespace

import torch
from torch import nn

import nanoquant.infrastructure.hf_language_model as loader


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Linear(2, 2)
        self.vision_tower = nn.Linear(3, 3)


def test_multimodal_loader_detaches_language_model_without_moving_vision(
    monkeypatch,
) -> None:
    wrapper = _Wrapper()
    expected = wrapper.language_model
    config_calls = []
    model_calls = []
    monkeypatch.setattr(
        loader.AutoConfig,
        "from_pretrained",
        lambda *_args, **kwargs: config_calls.append(kwargs) or SimpleNamespace(model_type="gemma3"),
    )
    monkeypatch.setattr(
        loader.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *_args, **kwargs: model_calls.append(kwargs) or wrapper,
    )

    result = loader.load_causal_language_model(
        "fixture",
        torch_dtype=torch.bfloat16,
        attention_implementation="eager",
    )

    assert result is expected
    assert config_calls == [{"local_files_only": False}]
    assert model_calls[0]["local_files_only"] is False
    assert "language_model" not in wrapper._modules
    assert "vision_tower" in wrapper._modules
