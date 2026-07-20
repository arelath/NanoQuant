import json
from pathlib import Path

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.preprocessing_materialization import materialize_resident_preprocessing


def _artifact(store: LocalArtifactStore, artifact_type: str, payload: object) -> ArtifactRef:
    with store.begin_write(artifact_type) as writer:
        (writer.path / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        descriptor = writer.commit()
    return ArtifactRef(descriptor.artifact_type, descriptor.artifact_id, descriptor.schema_version)


def test_materialize_resident_preprocessing_copies_validated_closure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    store = LocalArtifactStore(source / "artifacts")
    tensors = _artifact(store, "calibration-tensors", {"value": 1})
    calibration = _artifact(store, "calibration-stats", {"tensors": to_dict(tensors)})
    objectives = _artifact(store, "objective-specs", {"tensors": to_dict(tensors)})
    plan = _artifact(store, "quantization-plan", {"calibration": to_dict(calibration)})
    state = source / "state"
    state.mkdir()
    (state / "preprocessing.json").write_text(
        json.dumps(
            {
                "calibration": to_dict(calibration),
                "objectives": to_dict(objectives),
                "plan": to_dict(plan),
            }
        ),
        encoding="utf-8",
    )

    result = materialize_resident_preprocessing(source, destination)

    assert result.artifact_count == 4
    assert result.calibration == calibration
    copied = LocalArtifactStore(destination / "artifacts", use_persistent_validation_cache=False)
    for reference in (tensors, calibration, objectives, plan):
        copied.validate(reference.artifact_id)
    receipt = json.loads((destination / "preprocessing-input.json").read_text(encoding="utf-8"))
    assert receipt["artifact_count"] == 4

    repeated = materialize_resident_preprocessing(source, destination)
    assert repeated == result
