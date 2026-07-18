from pathlib import Path

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import DType
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


def test_experiment011_doubles_experiment006_outliers_and_stores_them_as_int8() -> None:
    parent = load_experiment(6)
    definition = load_experiment(11)
    config = definition.config
    experiment = definition.workflow

    assert _diff(parent.config, config) == {
        "intent.experiment_number",
        "intent.name",
        "intent.purpose",
        "intent.hypothesis",
        "intent.baseline_run",
        "intent.tags",
        "outliers.fraction",
        "outliers.storage_dtype",
        "output.run_root",
    }
    assert definition.identity.canonical_name == "011-compress-and-benchmark-gemma-3-1b-it"
    assert config.intent.baseline_run == "006-compress-and-benchmark-gemma-3-1b-it"
    assert config.model == parent.config.model
    assert config.outliers.fraction == parent.config.outliers.fraction * 2 == 0.002
    assert parent.config.outliers.storage_dtype is DType.BFLOAT16
    assert config.outliers.storage_dtype is DType.INT8
    assert config.outliers.selector == parent.config.outliers.selector
    assert config.outliers.charge_to_bit_budget is parent.config.outliers.charge_to_bit_budget is False
    assert experiment.expected_blocks == parent.workflow.expected_blocks == 26
    assert experiment.maximum_wddm_shared_gib == parent.workflow.maximum_wddm_shared_gib == 0.75
    assert experiment.wikitext_batch_size == parent.workflow.wikitext_batch_size == 8
    assert experiment.task_batch_size == parent.workflow.task_batch_size == 4
    assert experiment.export.gguf_output == Path("Results/011/gemma-3-1b-it-nanoquant.gguf")
