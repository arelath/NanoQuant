from pathlib import Path

from nanoquant.config.schema import DType
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment012_uses_ten_times_experiment011_int8_outliers() -> None:
    parent = load_experiment(11)
    definition = load_experiment(12)
    config = definition.config
    experiment = definition.workflow

    assert config_diff_paths(parent.config, config) == {
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.baseline_run",
        "intent.tags",
        "outliers.fraction",
        "output.run_root",
    }
    assert definition.identity.canonical_name == "012-compress-and-benchmark-gemma-3-1b-it"
    assert config.intent.baseline_run == "011-compress-and-benchmark-gemma-3-1b-it"
    assert config.model == parent.config.model
    assert config.outliers.fraction == parent.config.outliers.fraction * 10 == 0.02
    assert config.outliers.storage_dtype is parent.config.outliers.storage_dtype is DType.INT8
    assert config.outliers.selector == parent.config.outliers.selector
    assert config.outliers.charge_to_bit_budget is parent.config.outliers.charge_to_bit_budget is False
    assert experiment.expected_blocks == parent.workflow.expected_blocks == 26
    assert experiment.maximum_wddm_shared_gib == parent.workflow.maximum_wddm_shared_gib == 0.75
    assert experiment.wikitext_batch_size == parent.workflow.wikitext_batch_size == 8
    assert experiment.task_batch_size == parent.workflow.task_batch_size == 4
    assert experiment.export.gguf_output == Path("Results/012/gemma-3-1b-it-nanoquant.gguf")
