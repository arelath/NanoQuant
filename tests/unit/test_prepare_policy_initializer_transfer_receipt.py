from dataclasses import replace

import pytest

from tests.support.experiments import load_experiment
from tools.prepare_policy_initializer_transfer_receipt import (
    PolicyInitializerRegime,
    validate_policy_initializer_transfer,
)

MATCHED_256_PROTOCOL = (
    "sha256:0ed7993a02eb980403ebeb97ff2d2cbf738242e64e6a7d07ad9f2900ef611936"
)


def test_transfer_preflight_accepts_experiment048_matched_regime() -> None:
    config = load_experiment(48).config

    receipt = validate_policy_initializer_transfer(
        PolicyInitializerRegime(MATCHED_256_PROTOCOL, 256),
        config,
        selection_slice_id="selection",
        confirmation_slice_id="confirmation",
    )

    assert receipt["development_initializer"] == {
        "protocol_hash": MATCHED_256_PROTOCOL,
        "observed_steps": 256,
    }
    assert receipt["calibration_policy"] == "per-arm-held-out-fit-only"


@pytest.mark.parametrize(
    ("development", "message"),
    (
        (PolicyInitializerRegime("sha256:" + "0" * 64, 256), "development initializer regime"),
        (PolicyInitializerRegime(MATCHED_256_PROTOCOL, 2048), "development initializer regime"),
    ),
)
def test_transfer_preflight_rejects_silent_initializer_regime_changes(
    development: PolicyInitializerRegime,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_policy_initializer_transfer(
            development,
            load_experiment(48).config,
            selection_slice_id="selection",
            confirmation_slice_id="confirmation",
        )


def test_transfer_preflight_rejects_uncapped_primary_and_fixed_fold() -> None:
    config = load_experiment(48).config
    uncapped = replace(
        config,
        distillation=replace(config.distillation, maximum_batches_per_epoch=None),
    )
    with pytest.raises(ValueError, match="explicit primary batches-per-epoch"):
        validate_policy_initializer_transfer(
            PolicyInitializerRegime(MATCHED_256_PROTOCOL, 256),
            uncapped,
            selection_slice_id="selection",
            confirmation_slice_id="confirmation",
        )

    fixed_fold = replace(
        config,
        distillation=replace(
            config.distillation,
            final_norm_calibration=replace(
                config.distillation.final_norm_calibration,
                enabled=True,
                scale=1.015,
            ),
        ),
    )
    with pytest.raises(ValueError, match="fixed final-norm"):
        validate_policy_initializer_transfer(
            PolicyInitializerRegime(MATCHED_256_PROTOCOL, 256),
            fixed_fold,
            selection_slice_id="selection",
            confirmation_slice_id="confirmation",
        )


def test_transfer_preflight_requires_distinct_selection_and_confirmation() -> None:
    with pytest.raises(ValueError, match="distinct slice identities"):
        validate_policy_initializer_transfer(
            PolicyInitializerRegime(MATCHED_256_PROTOCOL, 256),
            load_experiment(48).config,
            selection_slice_id="reused",
            confirmation_slice_id="reused",
        )
