import math

from nanoquant.config.schema import (
    AllocationStrategy,
    DType,
    KlAllocationObjective,
    KlSensitivityGranularity,
    RankResponseSource,
)
from nanoquant.domain.planning import factor_bit_cost, outlier_bit_cost
from tests.support.experiments import load_experiment


def test_experiment060_is_a_charged_2bpw_1b_best_methods_run() -> None:
    experiment = load_experiment(60)
    config = experiment.config

    assert config.model.source == "google/gemma-3-1b-it"
    assert config.model.revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert config.intent.baseline_run == (
        "056-physical-cap-rank-redistribution-d2-compress-and-benchmark-gemma-3-1b-it"
    )
    assert config.allocation.target_bpw == 2.0
    assert config.allocation.strategy is AllocationStrategy.KL_CALIBRATED
    assert config.allocation.kl_sensitivity_granularity is KlSensitivityGranularity.EXACT
    assert config.allocation.bounds.ceiling_fraction_of_uniform == 1.5
    assert config.allocation.bounds.overcomplete_rank_ceiling_fraction == 1.0

    reconstruction = config.allocation.reconstruction
    assert reconstruction.kl_objective is KlAllocationObjective.MEASURED_UNIT_KL
    assert reconstruction.response_source is RankResponseSource.MEASURED
    assert reconstruction.objective_mode == "calibration_weighted"
    assert reconstruction.response_curves == ()
    assert reconstruction.rank_trust_reference_run is None

    assert config.outliers.fraction == 0.088
    assert config.outliers.storage_dtype is DType.INT8
    assert config.outliers.charge_to_bit_budget is True
    retry = config.allocation.retry
    assert retry.enabled is True
    assert retry.extra_bit_budget_fraction == 0.01
    assert retry.outlier_count_increment_at_rank_cap == 2

    assert config.factorization.product_codebook.enabled is False
    assert config.factorization.binary_search.enabled is False
    assert config.factorization.bias_correction.enabled is False
    assert config.factorization.low_rank_patch.enabled is False
    shared = config.factorization.shared_input
    assert shared.enabled is True
    assert shared.groups[0].member_multipliers[0].member == "self_attn.v_proj"
    assert shared.groups[0].member_multipliers[0].multiplier == 2.0
    assert config.block_tuning.factorized.learning_rates.binary == 3e-5
    assert config.distillation.enabled is True

    assert experiment.workflow.task_limit == 1000
    assert experiment.workflow.local_files_only is True
    assert experiment.workflow.interrupt_after_block_commits == 1


def test_experiment060_physical_rank_and_retry_upper_bound_fits_2bpw() -> None:
    config = load_experiment(60).config
    shapes = (
        (6912, 1152),  # gate projection
        (6912, 1152),  # up projection
        (1152, 6912),  # down projection
        (1536, 1152),  # stacked QKV
        (1152, 1024),  # output projection
    )
    represented_bits = 0
    weight_elements = 0
    for out_features, in_features in shapes:
        physical_rank = (
            min(out_features, in_features) // config.allocation.bounds.multiple
        ) * config.allocation.bounds.multiple
        outlier_count = math.ceil(in_features * config.outliers.fraction)
        represented_bits += factor_bit_cost(out_features, in_features, physical_rank).total
        represented_bits += outlier_bit_cost(
            out_features,
            outlier_count,
            value_bits=8,
            index_bits=math.ceil(math.log2(in_features)),
            scale_bits_per_column=16,
        ).total
        weight_elements += out_features * in_features

    retry_bits = math.floor(
        weight_elements
        * config.allocation.target_bpw
        * config.allocation.retry.extra_bit_budget_fraction
    )
    upper_bound_bpw = (represented_bits + retry_bits) / weight_elements
    assert 1.98 < upper_bound_bpw < config.allocation.target_bpw
