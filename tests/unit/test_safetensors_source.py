import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource, SourceIntegrityError


def _snapshot(root: Path) -> Path:
    root.mkdir()
    save_file(
        {"model.layers.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        root / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"model.layers.1.weight": torch.ones(3, 2, dtype=torch.float16)}, root / "model-00002-of-00002.safetensors"
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text('{"model_type":"tiny"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":"fixture"}', encoding="utf-8")
    return root


def test_inventory_reads_metadata_and_direct_tensor_without_full_state_dict(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "snapshot")
    source = SafetensorsModelSource(snapshot, source="fixture/tiny", revision="abc")
    inventory = source.inventory()
    assert inventory.source == "fixture/tiny" and inventory.revision == "abc"
    assert inventory.config["model_type"] == "tiny"
    assert inventory.tokenizer_files == ("tokenizer.json",)
    assert {tensor.key for tensor in inventory.tensors} == {
        "model.layers.0.weight",
        "model.layers.1.weight",
    }
    assert all(tensor.shard_hash and tensor.shard_hash.startswith("sha256:") for tensor in inventory.tensors)
    with source.read_tensor("model.layers.0.weight") as tensor:
        assert torch.equal(tensor, torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_index_path_traversal_and_changed_shard_are_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "snapshot")
    source = SafetensorsModelSource(snapshot, source="fixture/tiny", revision="abc")
    source.inventory()
    shard = snapshot / "model-00001-of-00002.safetensors"
    with shard.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(SourceIntegrityError, match="changed"):
        with source.read_tensor("model.layers.0.weight"):
            pass

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"../escape.safetensors"}}', encoding="utf-8"
    )
    with pytest.raises(SourceIntegrityError, match="unsafe shard path"):
        SafetensorsModelSource(unsafe, source="fixture/unsafe", revision="abc")


def test_single_shard_lookup_and_unknown_key(tmp_path: Path) -> None:
    snapshot = tmp_path / "single"
    snapshot.mkdir()
    save_file({"weight": torch.eye(2)}, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text('{"model_type":"tiny"}', encoding="utf-8")
    source = SafetensorsModelSource(snapshot, source="fixture/single", revision="def", verify_hashes=False)
    assert source.inventory().tensors[0].shard_hash is None
    with pytest.raises(KeyError, match="not in source"):
        with source.read_tensor("missing"):
            pass
