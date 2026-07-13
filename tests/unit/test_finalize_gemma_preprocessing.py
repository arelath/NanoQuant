from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nanoquant.application.calibration import MaterializedLayerCalibration
from tools.finalize_gemma_preprocessing import _retained_key, load_retained_fisher


def _materialized() -> tuple[MaterializedLayerCalibration, ...]:
    return (
        MaterializedLayerCalibration(
            "block.2.mlp.gate_proj",
            torch.zeros(3),
            torch.zeros(2),
            256,
            "online_fisher",
        ),
    )


def test_retained_key_maps_checkpoint_paths() -> None:
    assert _retained_key("block.2.mlp.gate_proj", "input") == "i.model.layers.2.mlp.gate_proj"
    assert _retained_key("block.2.mlp.gate_proj", "output") == "o.model.layers.2.mlp.gate_proj"


def test_load_retained_fisher_replaces_exact_vectors(tmp_path: Path) -> None:
    path = tmp_path / "retained.safetensors"
    save_file(
        {
            "i.model.layers.2.mlp.gate_proj": torch.tensor([1.0, 2.0, 3.0]),
            "o.model.layers.2.mlp.gate_proj": torch.tensor([4.0, 5.0]),
        },
        path,
    )

    loaded = load_retained_fisher(_materialized(), path)

    assert torch.equal(loaded[0].input_importance, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(loaded[0].output_importance, torch.tensor([4.0, 5.0]))
    assert loaded[0].sample_count == 256


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"i.model.layers.2.mlp.gate_proj": torch.ones(3)}, "do not exactly match"),
        (
            {
                "i.model.layers.2.mlp.gate_proj": torch.ones(4),
                "o.model.layers.2.mlp.gate_proj": torch.ones(2),
            },
            "input Fisher shape mismatch",
        ),
    ],
)
def test_load_retained_fisher_rejects_invalid_reference(
    tmp_path: Path, values: dict[str, torch.Tensor], message: str
) -> None:
    path = tmp_path / "retained.safetensors"
    save_file(values, path)

    with pytest.raises(ValueError, match=message):
        load_retained_fisher(_materialized(), path)
