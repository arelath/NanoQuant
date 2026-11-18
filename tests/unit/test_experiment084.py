import sys
from pathlib import Path

import huggingface_hub
from recipes import _experiment

from nanoquant.infrastructure import hf_calibration_dataset
from nanoquant.infrastructure.allocation_profile_dataset import publish_allocation_profile_dataset
from nanoquant.infrastructure.hf_task_evaluation import hash_hf_tokenizer_snapshot
from tests.support.experiments import load_experiment


def test_084_materializes_calibration_from_a_receipt_without_tokenizer_identity(tmp_path, monkeypatch):
    experiment = load_experiment(84)
    model = experiment.config.model
    source = tmp_path / "control"
    source.mkdir()
    (source / "calibration-input.json").write_text("{}", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    dataset_path = Path("allocation")
    dataset = publish_allocation_profile_dataset(
        tmp_path / dataset_path,
        {
            "schema_version": 1,
            "model": {"source": model.source, "revision": model.revision},
            "source_run": str(source),
            "units": [{"evidence": {"unit_id": "test"}}],
            "kl_rank_options": {"test": {"32": {"test": 0.1}}},
        },
        None,
    )

    def resolve_snapshot(*, repo_id, revision, local_files_only, allow_patterns):
        assert (repo_id, revision) == (model.source, model.tokenizer_revision)
        assert local_files_only and "tokenizer.json" in allow_patterns
        return str(snapshot)

    calls = []

    def materialize(source_output, destination_output, **kwargs):
        assert source_output == source
        assert destination_output == tmp_path / experiment.config.output.run_root / experiment.config.intent.name
        assert kwargs["tokenizer_identity"] == hash_hf_tokenizer_snapshot(snapshot)
        calls.append("materialized")

    def dispatch(definition, *, launcher_path):
        assert definition.config.allocation.reconstruction.allocation_dataset_key == dataset.reference.artifact_id
        calls.append("dispatched")
        return 0

    monkeypatch.setattr(sys, "argv", ["084.py"])
    monkeypatch.setattr(huggingface_hub, "snapshot_download", resolve_snapshot)
    monkeypatch.setattr(hf_calibration_dataset, "materialize_pinned_calibration", materialize)
    monkeypatch.setattr(_experiment, "run_experiment", dispatch)

    assert _experiment.run_kl_rank_option_experiment(
        experiment, launcher_path=str(tmp_path / "experiments" / "084.py"), dataset_path=dataset_path,
    ) == 0
    assert calls == ["materialized", "dispatched"]
