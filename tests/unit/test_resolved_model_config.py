from __future__ import annotations

import json
from pathlib import Path

import pytest

import nanoquant.infrastructure.resolved_model_config as resolved_config
from nanoquant.infrastructure.resolved_model_config import (
    load_snapshot_model_config,
    resolve_model_config,
)


def test_local_model_config_is_loaded_and_given_a_content_identity(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": 28}),
        encoding="utf-8",
    )

    resolved = resolve_model_config(str(snapshot), None)

    assert resolved.values["num_hidden_layers"] == 28
    assert resolved.revision.startswith("local-")
    assert resolved.path == snapshot / "config.json"


def test_pinned_hub_revision_does_not_require_a_model_info_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    snapshot = tmp_path / "hub-snapshot"
    snapshot.mkdir()
    config_path = snapshot / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "llama", "num_hidden_layers": 16}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        resolved_config.HfApi,
        "model_info",
        lambda *_args, **_kwargs: pytest.fail("pinned revisions need no metadata request"),
    )
    monkeypatch.setattr(
        resolved_config,
        "hf_hub_download",
        lambda **_kwargs: str(config_path),
    )

    resolved = resolve_model_config("owner/pinned-model", revision)

    assert resolved.revision == revision
    assert resolved.values["num_hidden_layers"] == 16


def test_snapshot_model_config_rejects_a_non_object_root(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be an object"):
        load_snapshot_model_config(tmp_path)
