from __future__ import annotations

from pathlib import Path

import torch
from recipes import BASE_COMPRESSION_TEMPLATE

from nanoquant.config.schema import RunConfig
from nanoquant.resident_workflow import ResolvedResidentInputs, resident_request_from_config
from tests.support.experiments import config_diff_paths, load_experiment


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

    assert config_diff_paths(BASE_COMPRESSION_TEMPLATE, config) == {
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
    launcher_numbers = [int(path.name[:3]) for path in launchers]
    documented_non_launcher_experiments = {
        20: Path("Docs/ImprovementSuggestions/D2-findings.md"),
    }

    assert launcher_numbers == sorted(set(launcher_numbers))
    assert set(range(1, max(launcher_numbers) + 1)) - set(launcher_numbers) == set(
        documented_non_launcher_experiments
    )
    for number, document in documented_non_launcher_experiments.items():
        assert document.is_file()
        assert f"Experiment {number:03d}" in document.read_text(encoding="utf-8")
    for launcher in launchers:
        number = int(launcher.name[:3])
        definition = load_experiment(number)
        assert definition.identity.canonical_name == launcher.stem
        assert definition.config.intent.name == launcher.stem

    assert not tuple(Path("experiments/recipes").glob("experiment[0-9][0-9][0-9].py"))
