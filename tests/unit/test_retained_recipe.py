import json
from pathlib import Path

import pytest

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import ModelConfig, RunConfig
from nanoquant.infrastructure.retained_recipe import load_retained_run_recipe


def _write_manifest(
    root: Path,
    *,
    status: str = "completed",
    source: str = "fixture/model",
    revision: str = "revision",
    maximum_wddm_shared_bytes: int | None = 123,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config = RunConfig(model=ModelConfig(source, revision=revision))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": status,
                "resolved_config": {
                    "canonical_run_config": to_dict(config),
                    "maximum_wddm_shared_bytes": maximum_wddm_shared_bytes,
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_retained_run_recipe(tmp_path: Path) -> None:
    _write_manifest(tmp_path)

    recipe = load_retained_run_recipe(
        tmp_path,
        expected_source="fixture/model",
        expected_revision="revision",
    )

    assert recipe.config.model.source == "fixture/model"
    assert recipe.maximum_wddm_shared_bytes == 123


@pytest.mark.parametrize(
    ("status", "source", "maximum", "match"),
    (
        ("running", "fixture/model", 123, "manifest is invalid"),
        ("completed", "other/model", 123, "model identity differs"),
        ("completed", "fixture/model", -1, "maximum_wddm_shared_bytes is invalid"),
    ),
)
def test_load_retained_run_recipe_rejects_invalid_identity_or_state(
    tmp_path: Path,
    status: str,
    source: str,
    maximum: int,
    match: str,
) -> None:
    _write_manifest(tmp_path, status=status, source=source, maximum_wddm_shared_bytes=maximum)

    with pytest.raises(ValueError, match=match):
        load_retained_run_recipe(
            tmp_path,
            expected_source="fixture/model",
            expected_revision="revision",
        )


def test_load_retained_run_recipe_can_explicitly_allow_interrupted(tmp_path: Path) -> None:
    _write_manifest(tmp_path, status="interrupted")

    recipe = load_retained_run_recipe(
        tmp_path,
        expected_source="fixture/model",
        expected_revision="revision",
        allowed_statuses=("completed", "interrupted"),
    )

    assert recipe.config.model.source == "fixture/model"
