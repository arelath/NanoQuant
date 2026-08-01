from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.fork_resident_run import fork_resident_run


def test_fork_resident_run_hardlinks_tree_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "artifacts" / "aa").mkdir(parents=True)
    (source / "manifest.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    payload = source / "artifacts" / "aa" / "payload.bin"
    payload.write_bytes(b"immutable artifact")
    destination = tmp_path / "derived"

    linked = fork_resident_run(source, destination, purpose="fixture replay")

    assert linked == 2
    assert (destination / "manifest.json").read_bytes() == (source / "manifest.json").read_bytes()
    assert os.path.samefile(payload, destination / "artifacts" / "aa" / "payload.bin")
    provenance = json.loads((destination / "derived-run-provenance.json").read_text(encoding="utf-8"))
    assert provenance["purpose"] == "fixture replay"
    assert provenance["linked_source_files"] == 2
    assert provenance["source_run_output"] == str(source.resolve())


def test_fork_resident_run_rejects_nested_or_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain"):
        fork_resident_run(source, source / "derived", purpose="fixture")

    destination = tmp_path / "derived"
    destination.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        fork_resident_run(source, destination, purpose="fixture")
