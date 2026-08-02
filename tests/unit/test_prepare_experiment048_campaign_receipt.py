from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.distillation_checkpoint import DistillationCheckpointIdentity
from tests.support.experiments import load_experiment
from tools.prepare_experiment048_campaign_receipt import (
    _checkpoint_receipts,
    _primary_checkpoint_receipt,
    _reserved_c4_slices,
    _retained_reference_argument,
    _validate_experiment048_config,
)
from tools.probe_distillation_checkpoint_tail_mass import CheckpointCandidate


def _slice(identity: str, offset: int, consumer: str) -> dict[str, object]:
    return {
        "id": identity,
        "dataset": "allenai/c4",
        "split": "validation",
        "offset": offset,
        "samples": 48,
        "sequence_length": 512,
        "token_start": offset * 512,
        "token_end": (offset + 48) * 512,
        "token_hash": "sha256:" + str(offset).zfill(64),
        "status": "reserved",
        "consumer": consumer,
    }


def test_retained_reference_parser_preserves_windows_drive_paths() -> None:
    assert _retained_reference_argument(r"accepted040=D:\evidence\040;32") == (
        "accepted040",
        Path(r"D:\evidence\040"),
        32,
    )


def test_campaign_config_accepts_only_the_frozen_experiment048_regime() -> None:
    config = load_experiment(48).config

    assert _validate_experiment048_config(config).startswith("sha256:")

    changed = replace(
        config,
        distillation=replace(
            config.distillation,
            mass_floor_correction=replace(
                config.distillation.mass_floor_correction,
                epochs=3,
            ),
        ),
    )
    with pytest.raises(ValueError, match="frozen campaign"):
        _validate_experiment048_config(changed)


def test_campaign_slices_are_reserved_disjoint_and_role_bound() -> None:
    selection = _slice("selection", 344, "experiment-048-correction-selection")
    confirmation = _slice("confirmation", 392, "experiment-048-final-confirmation")
    payload = {
        "schema_version": 1,
        "policy": "Every used token interval is permanently retired.",
        "slices": [selection, confirmation],
    }

    actual_selection, actual_confirmation, audit = _reserved_c4_slices(
        payload,
        selection_id="selection",
        confirmation_id="confirmation",
    )

    assert actual_selection == selection
    assert actual_confirmation == confirmation
    assert audit["reserved_count"] == 2

    confirmation["status"] = "retired"
    with pytest.raises(ValueError, match="frozen role"):
        _reserved_c4_slices(
            payload,
            selection_id="selection",
            confirmation_id="confirmation",
        )


def test_campaign_checkpoint_inventory_binds_primary_and_all_four_steps() -> None:
    primary = ArtifactRef("global-tuning-result", "sha256-" + "a" * 64, 1)
    source = (ArtifactRef("activation-generation", "sha256-" + "b" * 64, 1),)
    candidates = tuple(
        CheckpointCandidate(
            epoch,
            epoch * 32,
            ArtifactRef("distillation-checkpoint", "sha256-" + str(epoch) * 64, 1),
            DistillationCheckpointIdentity(
                source,
                "sha256:correction",
                "sha256:tokens",
                initializer_global_tuning=primary,
            ),
        )
        for epoch in range(1, 5)
    )

    receipts = _checkpoint_receipts(
        candidates,
        primary=primary,
        source_blocks=source,
        correction_protocol_hash="sha256:correction",
    )

    assert [receipt["steps"] for receipt in receipts] == [32, 64, 96, 128]
    changed = replace(candidates[2], steps=95)
    with pytest.raises(ValueError, match="epoch 3 differs"):
        _checkpoint_receipts(
            (*candidates[:2], changed, candidates[3]),
            primary=primary,
            source_blocks=source,
            correction_protocol_hash="sha256:correction",
        )


def test_campaign_binds_primary_epoch8_as_the_uncorrected_fallback() -> None:
    primary = ArtifactRef("global-tuning-result", "sha256-" + "a" * 64, 1)
    source = (ArtifactRef("activation-generation", "sha256-" + "b" * 64, 1),)
    checkpoint = CheckpointCandidate(
        8,
        256,
        ArtifactRef("distillation-checkpoint", "sha256-" + "c" * 64, 1),
        DistillationCheckpointIdentity(source, "sha256:primary", "sha256:tokens"),
    )

    receipt = _primary_checkpoint_receipt(
        (checkpoint,),
        primary=primary,
        source_blocks=source,
        protocol_hash="sha256:primary",
    )

    assert receipt["epoch"] == 8
    assert receipt["steps"] == 256
    assert receipt["endpoint_reference"]["artifact_id"] == primary.artifact_id
    changed = replace(checkpoint, steps=255)
    with pytest.raises(ValueError, match="fallback checkpoint"):
        _primary_checkpoint_receipt(
            (changed,),
            primary=primary,
            source_blocks=source,
            protocol_hash="sha256:primary",
        )
