from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)


def test_publish_experiment_artifacts_creates_zero_copy_numbered_results(tmp_path: Path) -> None:
    model = tmp_path / "outputs" / "model.gguf"
    statistics = tmp_path / "evidence" / "stats.json"
    model.parent.mkdir()
    statistics.parent.mkdir()
    model.write_bytes(b"model")
    statistics.write_text('{"passed":true}\n', encoding="utf-8")

    result = publish_experiment_artifacts(
        tmp_path,
        3,
        (
            PublishableArtifact(model, PublishableArtifactKind.MODEL),
            PublishableArtifact(statistics, PublishableArtifactKind.STATISTICS),
        ),
    )

    published_model = tmp_path / "Results" / "003" / "model.gguf"
    published_statistics = tmp_path / "Results" / "003" / "stats.json"
    assert os.path.samefile(model, published_model)
    assert os.path.samefile(statistics, published_statistics)
    assert result.results_directory == tmp_path / "Results" / "003"
    payload = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert payload["experiment_number"] == 3
    assert [item["kind"] for item in payload["artifacts"]] == ["model", "statistics"]
    assert all(item["link_type"] in {"hardlink", "symlink"} for item in payload["artifacts"])


def test_republication_refreshes_replaced_sources_and_preserves_other_stage_outputs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    stale = tmp_path / "stale.md"
    first.write_text("first", encoding="utf-8")
    stale.write_text("stale", encoding="utf-8")
    publish_experiment_artifacts(
        tmp_path,
        12,
        (
            PublishableArtifact(first, PublishableArtifactKind.STATISTICS),
            PublishableArtifact(stale, PublishableArtifactKind.REPORT),
        ),
    )
    replacement = tmp_path / "replacement.json"
    replacement.write_text("second", encoding="utf-8")
    replacement.replace(first)

    publish_experiment_artifacts(
        tmp_path,
        12,
        (PublishableArtifact(first, PublishableArtifactKind.STATISTICS),),
    )

    published = tmp_path / "Results" / "012" / "first.json"
    assert published.read_text(encoding="utf-8") == "second"
    assert os.path.samefile(first, published)
    assert (tmp_path / "Results" / "012" / "stale.md").exists()
    manifest = json.loads((tmp_path / "Results" / "012" / "publication.json").read_text(encoding="utf-8"))
    assert [Path(item["published"]).name for item in manifest["artifacts"]] == ["first.json", "stale.md"]


def test_publication_does_not_replace_unmanaged_result_file(tmp_path: Path) -> None:
    source = tmp_path / "stats.json"
    destination = tmp_path / "Results" / "002" / "stats.json"
    source.write_text("source", encoding="utf-8")
    destination.parent.mkdir(parents=True)
    destination.write_text("unmanaged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not managed"):
        publish_experiment_artifacts(
            tmp_path,
            2,
            (PublishableArtifact(source, PublishableArtifactKind.STATISTICS),),
        )


def test_publication_requires_at_least_one_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        publish_experiment_artifacts(tmp_path, 2, ())
