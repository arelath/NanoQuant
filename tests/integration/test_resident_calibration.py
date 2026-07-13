import json
from dataclasses import replace
from pathlib import Path

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.resident_calibration import ResidentCalibrationRequest, run_resident_calibration


def test_resident_calibration_persists_replayable_all_layer_objectives(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    Gemma3ForCausalLM(config).save_pretrained(snapshot, safe_serialization=True)

    request = ResidentCalibrationRequest(
        snapshot,
        tmp_path / "run",
        "fixture/gemma3",
        "pinned-test-revision",
        ((1, 2, 3, 4),),
        device="cpu",
    )
    result = run_resident_calibration(request)
    control = run_resident_calibration(
        replace(
            request,
            output=tmp_path / "control",
            profiling=ProfilingConfig(level=ProfilingLevel.OFF),
        )
    )

    assert len(result.inventory.blocks) == 1
    assert result.layer_count == 7
    assert result.total_tokens == 4
    assert result.maximum_logit_difference < 1e-6
    assert result.logit_mse < 1e-12
    artifacts = LocalArtifactStore(tmp_path / "run" / "artifacts")
    artifacts.validate(result.calibration.artifact_id)
    artifacts.validate(result.objectives.artifact_id)
    artifacts.validate(result.report.artifact_id)
    assert result.inventory == control.inventory
    assert result.calibration == control.calibration
    assert result.objectives == control.objectives
    assert result.layer_count == control.layer_count
    assert result.total_tokens == control.total_tokens
    assert result.maximum_logit_difference == control.maximum_logit_difference
    assert result.logit_mse == control.logit_mse
    assert not (tmp_path / "control" / "profile.json").exists()
    profile = json.loads((tmp_path / "run" / "profile.json").read_text(encoding="utf-8"))
    assert profile["run_id"] == "resident-calibration"
    assert profile["level"] == "macro"
    assert profile["coverage"]["fraction"] >= 0.90
    assert not any(warning["code"] == "PERF001" for warning in profile["warnings"])
    phases = {str(phase["path"]): phase for phase in profile["phases"]}
    assert {
        "run/source",
        "run/model_load",
        "run/reference",
        "run/prefix_capture",
        "run/block/load",
        "run/block/calibrate",
        "run/block/propagate",
        "run/suffix",
        "run/persist_calibration",
        "run/build_objectives",
        "run/report",
    } <= phases.keys()
    assert set(phases["run/block"]["groups"]) == {"block=0"}
    counters = {str(counter["name"]): counter for counter in profile["counters"]}
    assert counters["calibration.blocks"]["total"] == 1
    assert counters["calibration.layers"]["total"] == 7
    assert counters["calibration.tokens"]["total"] == 4
