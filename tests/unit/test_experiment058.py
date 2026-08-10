import math

from nanoquant.config.schema import DType
from nanoquant.domain.planning import outlier_bit_cost
from tests.support.experiments import config_diff_paths, load_experiment


def test_experiment058_changes_only_057_rate_matched_int8_outlier_policy() -> None:
    baseline = load_experiment(57)
    candidate = load_experiment(58)

    assert candidate.identity.baseline is not None
    assert candidate.identity.baseline.label == baseline.identity.canonical_name
    assert candidate.config.outliers.selector == baseline.config.outliers.selector
    assert candidate.config.outliers.storage_dtype is DType.INT8
    assert candidate.config.outliers.charge_to_bit_budget is False
    assert math.ceil(6912 * candidate.config.outliers.fraction) == 13
    bf16_bits = outlier_bit_cost(1152, 7, value_bits=16, index_bits=13).total
    int8_bits = outlier_bit_cost(
        1152,
        13,
        value_bits=8,
        index_bits=13,
        scale_bits_per_column=16,
    ).total
    assert int8_bits == 120_185 < bf16_bits == 129_115
    assert candidate.config.outliers.residual_probe == baseline.config.outliers.residual_probe
    assert candidate.config.factorization == baseline.config.factorization
    assert candidate.config.block_tuning == baseline.config.block_tuning
    assert candidate.config.distillation == baseline.config.distillation
    assert candidate.workflow.task_limit == baseline.workflow.task_limit == 1000
    assert config_diff_paths(baseline.config, candidate.config) == {
        "intent.baseline_run",
        "intent.experiment_number",
        "intent.hypothesis",
        "intent.name",
        "intent.purpose",
        "intent.tags",
        "outliers.fraction",
        "outliers.storage_dtype",
        "output.run_root",
    }
