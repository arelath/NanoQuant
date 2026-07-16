"""Zero-copy publication of durable experiment outputs into ``Results/NNN``."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from nanoquant.infrastructure.io_utils import atomic_write_json


class PublishableArtifactKind(str, Enum):
    MODEL = "model"
    STATISTICS = "statistics"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class PublishableArtifact:
    source: Path
    kind: PublishableArtifactKind
    published_name: str | None = None

    def __post_init__(self) -> None:
        name = self.source.name if self.published_name is None else self.published_name
        if not name or Path(name).name != name or name in {".", "..", "publication.json"}:
            raise ValueError(f"published artifact name is invalid: {name!r}")


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    kind: str
    source: str
    published: str
    link_type: str
    bytes: int


@dataclass(frozen=True, slots=True)
class PublicationResult:
    experiment_number: int
    results_directory: Path
    manifest: Path
    artifacts: tuple[PublishedArtifact, ...]


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(path)


def _previous_artifacts(manifest: Path) -> dict[str, PublishedArtifact]:
    if not manifest.is_file():
        return {}

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", ())
    if not isinstance(artifacts, list):
        raise ValueError(f"publication manifest has invalid artifacts: {manifest}")
    result: dict[str, PublishedArtifact] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError(f"publication manifest has invalid artifact entry: {manifest}")
        try:
            published = PublishedArtifact(
                str(item["kind"]),
                str(item["source"]),
                str(item["published"]),
                str(item["link_type"]),
                int(item["bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"publication manifest has invalid artifact entry: {manifest}") from error
        result[Path(published.published).name] = published
    return result


def _hardlink(source: Path, destination: Path, *, replace_owned: bool) -> str:
    if destination.exists() or destination.is_symlink():
        try:
            if os.path.samefile(source, destination):
                return "symlink" if destination.is_symlink() else "hardlink"
        except OSError:
            pass
        if not replace_owned:
            raise FileExistsError(f"publication destination is not managed by NanoQuant: {destination}")
        destination.unlink()
    temporary = destination.with_name(f".{destination.name}.publishing-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
        return "hardlink"
    except OSError as hardlink_error:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        try:
            temporary.symlink_to(source)
            os.replace(temporary, destination)
            return "symlink"
        except OSError as symlink_error:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            raise OSError(
                f"cannot publish {source} as a hard link or symbolic link: {symlink_error}"
            ) from hardlink_error


def publish_experiment_artifacts(
    repository_root: str | Path,
    experiment_number: int,
    artifacts: Iterable[PublishableArtifact],
) -> PublicationResult:
    """Publish files without copying their contents and write link provenance."""

    if experiment_number < 0 or experiment_number > 999:
        raise ValueError("experiment number must be between 0 and 999")
    root = Path(repository_root).resolve()
    results_directory = root / "Results" / f"{experiment_number:03d}"
    results_directory.mkdir(parents=True, exist_ok=True)
    manifest = results_directory / "publication.json"
    previous = _previous_artifacts(manifest)
    previously_owned = set(previous)
    requested = tuple(artifacts)
    if not requested:
        raise ValueError("at least one publishable artifact is required")
    names = tuple(item.source.name if item.published_name is None else item.published_name for item in requested)
    if len(set(names)) != len(names):
        raise ValueError("publishable artifact names must be unique within an experiment")
    published_by_name = {
        name: item
        for name, item in previous.items()
        if (results_directory / name).exists() or (results_directory / name).is_symlink()
    }
    for artifact, name in zip(requested, names, strict=True):
        source = artifact.source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"publishable artifact is not a regular file: {source}")
        destination = results_directory / name
        link_type = _hardlink(source, destination, replace_owned=name in previously_owned)
        published_by_name[name] = PublishedArtifact(
            artifact.kind.value,
            _display_path(source, root),
            _display_path(destination, root),
            link_type,
            source.stat().st_size,
        )
    published = tuple(published_by_name[name] for name in sorted(published_by_name))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_number": experiment_number,
        "artifacts": [asdict(item) for item in published],
    }
    atomic_write_json(manifest, payload)
    return PublicationResult(experiment_number, results_directory, manifest, published)


__all__ = [
    "PublishableArtifact",
    "PublishableArtifactKind",
    "PublicationResult",
    "PublishedArtifact",
    "publish_experiment_artifacts",
]
