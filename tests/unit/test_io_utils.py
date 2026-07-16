import json
from pathlib import Path

import pytest

from nanoquant.infrastructure.io_utils import (
    atomic_write_json,
    atomic_write_text,
    hash_file,
    safe_replace,
)


def test_hash_file_and_atomic_json_write_are_deterministic(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    content = tmp_path / "content.bin"
    content.write_bytes(b"abc")

    assert atomic_write_json(destination, {"value": 3, "name": "fixture"}) is True

    assert json.loads(destination.read_text(encoding="utf-8")) == {"name": "fixture", "value": 3}
    assert hash_file(content) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert not tuple(tmp_path.glob(".state.json-*.tmp"))


def test_atomic_text_write_uses_utf8_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    destination = tmp_path / "report.md"

    assert atomic_write_text(destination, "# Résult\n") is True

    assert destination.read_bytes() == "# Résult\n".encode()
    assert not tuple(tmp_path.glob(".report.md-*.tmp"))


def test_safe_replace_retries_permission_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def replace(_source: object, _destination: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("busy")

    monkeypatch.setattr("nanoquant.infrastructure.io_utils.os.replace", replace)
    monkeypatch.setattr("nanoquant.infrastructure.io_utils.time.sleep", lambda _seconds: None)

    assert safe_replace("source", "destination") is True
    assert attempts == 3


def test_safe_replace_can_suppress_advisory_cache_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nanoquant.infrastructure.io_utils.os.replace",
        lambda _source, _destination: (_ for _ in ()).throw(PermissionError("busy")),
    )
    monkeypatch.setattr("nanoquant.infrastructure.io_utils.time.sleep", lambda _seconds: None)

    assert safe_replace("source", "destination", attempts=2, suppress_errors=True) is False
