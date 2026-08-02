"""Irreversibly open one receipt-authorized Experiment 048 C4 slice."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
from validate_evaluation_slice_registry import validate_registry

from nanoquant.config.codec import semantic_hash
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

_ROLES = {
    "selection": "experiment-048-correction-selection",
    "confirmation": "experiment-048-final-confirmation",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-receipt", type=Path, required=True)
    parser.add_argument("--slice-registry", type=Path, required=True)
    parser.add_argument("--role", choices=tuple(_ROLES), required=True)
    return parser


def _sha256(path: Path) -> str:
    return "sha256:" + hash_file(path)


@contextmanager
def _exclusive_registry_lock(registry: Path) -> Iterator[None]:
    """Hold a non-blocking OS lock while performing the read/replace transition."""

    lock_path = registry.with_name(registry.name + ".lifecycle.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("evaluation slice registry is already being changed") from exc
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("evaluation slice registry is already being changed") from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _receipt_protocol(receipt: object) -> tuple[dict[str, Any], str]:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValueError("Experiment 048 campaign receipt schema is invalid")
    protocol = receipt.get("protocol")
    protocol_hash = receipt.get("protocol_hash")
    if (
        receipt.get("status") != "ready_for_selection_evaluation"
        or not isinstance(protocol, dict)
        or protocol.get("experiment") != 48
        or not isinstance(protocol_hash, str)
        or semantic_hash(protocol) != protocol_hash
    ):
        raise ValueError("Experiment 048 campaign receipt identity is invalid")
    return cast(dict[str, Any], protocol), protocol_hash


def _retirement_record(
    *, receipt_hash: str, protocol_hash: str, role: str
) -> dict[str, object]:
    return {
        "action": "permanently-retired-before-evaluation",
        "authorization_receipt_sha256": receipt_hash,
        "authorization_protocol_hash": protocol_hash,
        "role": role,
    }


def _authorized_registry_state(
    current: object,
    snapshot: object,
    *,
    receipt_hash: str,
    protocol_hash: str,
) -> None:
    """Require that only receipt-authorized lifecycle transitions changed."""

    validate_registry(current)
    validate_registry(snapshot)
    if not isinstance(current, dict) or not isinstance(snapshot, dict):
        raise ValueError("Experiment 048 registry snapshot is invalid")
    expected = json.loads(json.dumps(snapshot))
    current_entries = current.get("slices")
    expected_entries = expected.get("slices")
    if not isinstance(current_entries, list) or not isinstance(expected_entries, list):
        raise ValueError("Experiment 048 registry entries are invalid")
    current_by_id = {entry.get("id"): entry for entry in current_entries}
    for entry in expected_entries:
        if entry.get("consumer") not in _ROLES.values():
            continue
        observed = current_by_id.get(entry.get("id"))
        if not isinstance(observed, dict):
            raise ValueError("Experiment 048 authorized slice is missing")
        if observed.get("status") == "retired":
            entry["status"] = "retired"
            role = next(key for key, value in _ROLES.items() if value == entry["consumer"])
            entry["retirement"] = _retirement_record(
                receipt_hash=receipt_hash,
                protocol_hash=protocol_hash,
                role=role,
            )
    if current != expected:
        raise ValueError(
            "evaluation slice registry changed outside receipt-authorized lifecycle transitions"
        )


def open_slice(
    campaign_receipt: Path,
    slice_registry: Path,
    role: str,
) -> dict[str, object]:
    campaign_receipt = campaign_receipt.resolve()
    slice_registry = slice_registry.resolve()
    receipt = json.loads(campaign_receipt.read_text(encoding="utf-8"))
    protocol, protocol_hash = _receipt_protocol(receipt)
    receipt_hash = _sha256(campaign_receipt)
    slices = protocol.get("slices")
    bound_files = protocol.get("bound_files")
    if not isinstance(slices, dict) or not isinstance(bound_files, dict):
        raise ValueError("Experiment 048 campaign slice bindings are invalid")
    snapshot = slices.get("registry_snapshot")
    snapshot_hash = slices.get("registry_snapshot_hash")
    authorized = slices.get(role)
    registry_binding = bound_files.get("slice_registry")
    if (
        role not in _ROLES
        or not isinstance(authorized, dict)
        or authorized.get("consumer") != _ROLES[role]
        or authorized.get("status") != "reserved"
        or not isinstance(snapshot, dict)
        or semantic_hash(snapshot) != snapshot_hash
        or not isinstance(registry_binding, dict)
        or Path(str(registry_binding.get("path"))).resolve() != slice_registry
        or not isinstance(registry_binding.get("sha256"), str)
    ):
        raise ValueError("Experiment 048 registry or slice differs from its receipt")

    with _exclusive_registry_lock(slice_registry):
        current = json.loads(slice_registry.read_text(encoding="utf-8"))
        _authorized_registry_state(
            current,
            snapshot,
            receipt_hash=receipt_hash,
            protocol_hash=protocol_hash,
        )
        if current == snapshot and _sha256(slice_registry) != registry_binding["sha256"]:
            raise ValueError("Experiment 048 base registry bytes differ from its receipt")
        entries = cast(list[dict[str, Any]], current["slices"])
        matches = [entry for entry in entries if entry.get("id") == authorized.get("id")]
        if len(matches) != 1:
            raise ValueError("Experiment 048 authorized slice is missing or ambiguous")
        entry = matches[0]
        retirement = _retirement_record(
            receipt_hash=receipt_hash,
            protocol_hash=protocol_hash,
            role=role,
        )
        if entry.get("status") == "retired":
            if entry.get("retirement") != retirement:
                raise ValueError("Experiment 048 slice has a different retirement authority")
        elif entry == authorized:
            entry["status"] = "retired"
            entry["retirement"] = retirement
            validate_registry(current)
            atomic_write_json(slice_registry, current)
        else:
            raise ValueError("Experiment 048 slice differs from its campaign receipt")
        persisted = json.loads(slice_registry.read_text(encoding="utf-8"))
        if persisted != current:
            raise RuntimeError("evaluation slice retirement did not persist exactly")
    return {
        "schema_version": 1,
        "status": "opened-and-permanently-retired",
        "role": role,
        "slice_id": authorized["id"],
        "campaign_protocol_hash": protocol_hash,
        "campaign_receipt_sha256": receipt_hash,
        "registry_sha256": _sha256(slice_registry),
    }


def run(args: argparse.Namespace) -> int:
    result = open_slice(args.campaign_receipt, args.slice_registry, args.role)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
