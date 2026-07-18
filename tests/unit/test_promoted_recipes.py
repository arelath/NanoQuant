from __future__ import annotations

from pathlib import Path

import torch
from recipes import BASE_COMPRESSION_TEMPLATE

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.resident_workflow import ResolvedResidentInputs, resident_request_from_config
from tests.support.experiments import load_experiment


def _diff(left: object, right: object, prefix: str = "") -> set[str]:
    left = to_dict(left)
    right = to_dict(right)
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            paths.update(_diff(left.get(key), right.get(key), path))
        return paths
    return set() if left == right else {prefix}


def _inputs(config: RunConfig, tmp_path: Path) -> ResolvedResidentInputs:
    tokens = torch.zeros((config.calibration.sample_count, 8), dtype=torch.long)
    return ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / config.intent.name,
        registry_root=tmp_path,
        token_ids=tokens,
        quality_token_ids=tokens[:1],
        pad_token_id=0,
    )


def test_experiment001_uses_the_current_base_compression_template(tmp_path: Path) -> None:
    definition = load_experiment(1)
    config = definition.config
    experiment = definition.workflow
    request = resident_request_from_config(config, _inputs(config, tmp_path))

    assert _diff(BASE_COMPRESSION_TEMPLATE, config) == {
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.baseline_run",
        "intent.tags",
        "output.run_root",
    }
    assert config.intent.name == "001-compress-gemma-3-1b-it"
    assert config.output.run_root == "evidence/001"
    assert config.outliers.fraction == 0.001
    assert request.nonfactorized_tuning_epochs_by_layer == (8, 4, 3, 2, 2, 2, 2)
    assert request.factorized_tuning_epochs == 8
    assert request.post_block_refit_epochs == 2
    assert request.defer_run_completion
    assert experiment.export.gguf_output.name == "gemma-3-1b-it-nanoquant.gguf"
    assert experiment.wikitext_samples == 64
    assert experiment.task_limit == 200


def test_numbered_launchers_own_their_concrete_definitions() -> None:
    launchers = sorted(Path("experiments").glob("[0-9][0-9][0-9]-*.py"))

    assert [path.name for path in launchers] == [
        "001-compress-gemma-3-1b-it.py",
        "002-benchmark-gemma-3-1b-it.py",
        "003-compress-and-benchmark-gemma-3-4b-it.py",
        "004-gemma-3-4b-it-vproj-plus30.py",
        "005-gemma-3-4b-it-vproj-double-request.py",
        "006-compress-and-benchmark-gemma-3-1b-it.py",
        "007-compress-and-benchmark-gemma-3-270m-it.py",
        "008-compress-and-benchmark-gemma-3-12b-it.py",
        "009-compress-benchmark-and-publish-gemma-3-270m-it.py",
        "010-compress-and-benchmark-gemma-3-270m-it.py",
        "011-compress-and-benchmark-gemma-3-1b-it.py",
        "012-compress-and-benchmark-gemma-3-1b-it.py",
        "013-compress-and-benchmark-gemma-3-270m-it.py",
    ]
    for number, launcher in enumerate(launchers, start=1):
        definition = load_experiment(number)
        assert definition.identity.canonical_name == launcher.stem
        assert definition.config.intent.name == launcher.stem

    assert not tuple(Path("experiments/recipes").glob("experiment[0-9][0-9][0-9].py"))
