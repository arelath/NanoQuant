from pathlib import Path

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

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

    result = run_resident_calibration(
        ResidentCalibrationRequest(
            snapshot,
            tmp_path / "run",
            "fixture/gemma3",
            "pinned-test-revision",
            ((1, 2, 3, 4),),
            device="cpu",
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
