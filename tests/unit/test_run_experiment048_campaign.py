from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from tools.probe_non_wikitext_kd_quality import _parser as quality_parser
from tools.run_experiment048_campaign import (
    CORRECTION_NAMESPACE,
    _campaign_paths,
    _confirmation_command,
    _fit_command,
    _selected,
    _selection_evaluation_command,
    _static_plan,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        snapshot=tmp_path / "snapshot",
        slice_registry=tmp_path / "registry.json",
        selection_slice_id="selection",
        confirmation_slice_id="confirmation",
        selection_offset=344,
        confirmation_offset=392,
        c4_file="cached-c4.arrow",
        device="cuda:0",
        local_files_only=True,
    )


def test_campaign_plan_keeps_selection_and_confirmation_in_distinct_stages(
    tmp_path: Path,
) -> None:
    plan = _static_plan(_campaign_paths(tmp_path / "campaign"))

    assert plan.index("open-and-retire-selection-slice") < plan.index("immutable-checkpoint-decision")
    assert plan.index("immutable-checkpoint-decision") < plan.index(
        "if corrected: open final slice and run raw/fitted absolute C4 confirmation"
    )
    assert plan[-1].endswith("1000-example six-task quality")


def test_selection_command_binds_primary_and_all_correction_checkpoints(
    tmp_path: Path,
) -> None:
    command = _selection_evaluation_command(_args(tmp_path), _campaign_paths(tmp_path / "campaign"))
    arms = [command[index + 1] for index, item in enumerate(command) if item == "--arm"]
    expected = [command[index + 1] for index, item in enumerate(command) if item == "--expected-steps"]

    assert arms[0].startswith("uncorrected=tuning;")
    assert [arm.split("=")[0] for arm in arms[1:]] == [
        "correction1",
        "correction2",
        "correction3",
        "correction4",
    ]
    assert all(arm.endswith(";" + CORRECTION_NAMESPACE) for arm in arms[1:])
    assert expected == [
        "uncorrected=256",
        "correction1=32",
        "correction2=64",
        "correction3=96",
        "correction4=128",
    ]
    assert command[command.index("--c4-file") + 1] == "cached-c4.arrow"
    parsed = quality_parser().parse_args(command[2:])
    assert parsed.slice_id == "selection"


def test_selected_decision_maps_fallback_and_correction_namespaces() -> None:
    assert _selected(
        {
            "selected_arm": "uncorrected",
            "selected_steps": 256,
            "correction_applied": False,
        }
    ) == ("uncorrected", 256, 8, "global-distillation", False)
    assert _selected(
        {
            "selected_arm": "correction3",
            "selected_steps": 96,
            "correction_applied": True,
        }
    ) == ("correction3", 96, 3, CORRECTION_NAMESPACE, True)
    with pytest.raises(ValueError, match="invalid selected arm"):
        _selected(
            {
                "selected_arm": "correction3",
                "selected_steps": 95,
                "correction_applied": True,
            }
        )


def test_temperature_fit_command_reuses_exact_selection_slice_protocol(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = _campaign_paths(tmp_path / "campaign")
    arm = (
        f"correction3=checkpoint;{paths.resident};{paths.resident};"
        f"3;{CORRECTION_NAMESPACE}"
    )

    command = _fit_command(
        args,
        paths,
        name="correction3",
        arm=arm,
        role="selected",
        steps=96,
        output=paths.selected_temperature,
    )

    assert command[command.index("--arm") + 1] == arm
    assert command[command.index("--slice-id") + 1] == "selection"
    assert command[command.index("--offset") + 1] == "344"
    assert command[command.index("--c4-file") + 1] == "cached-c4.arrow"


def test_confirmation_command_marks_only_absolute_context_as_references(
    tmp_path: Path,
) -> None:
    command = _confirmation_command(
        _args(tmp_path),
        _campaign_paths(tmp_path / "campaign"),
        selected_name="correction3",
        selected_steps=96,
    )
    references = [command[index + 1] for index, item in enumerate(command) if item == "--reference-arm"]

    assert references == ["prekd", "accepted040", "tailaware044"]
    assert command[command.index("--primary-baseline") + 1] == "uncorrected"
    assert command[command.index("--primary-candidate") + 1] == "correction3"
    arms = [command[index + 1] for index, item in enumerate(command) if item == "--arm"]
    selected = next(arm for arm in arms if arm.startswith("correction3="))
    assert selected.startswith("correction3=checkpoint;")
    assert selected.endswith(";3;" + CORRECTION_NAMESPACE)
    assert command[command.index("--c4-file") + 1] == "cached-c4.arrow"
    parsed = quality_parser().parse_args(command[2:])
    assert parsed.reference_arm == references
