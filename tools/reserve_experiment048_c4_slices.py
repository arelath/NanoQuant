"""Hash and atomically reserve the two fresh Experiment 048 C4 slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
from open_experiment048_c4_slice import _exclusive_registry_lock
from probe_non_wikitext_kd_quality import _load_c4_tokens, _token_hash
from validate_evaluation_slice_registry import validate_registry

from nanoquant.config.codec import semantic_hash
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--c4-file", type=Path, required=True)
    parser.add_argument("--slice-registry", type=Path, required=True)
    parser.add_argument("--selection-slice-id", required=True)
    parser.add_argument("--confirmation-slice-id", required=True)
    parser.add_argument("--selection-offset", type=int, default=344)
    parser.add_argument("--confirmation-offset", type=int, default=392)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return "sha256:" + hash_file(path)


def _entry(identity: str, offset: int, token_hash: str, consumer: str) -> dict[str, object]:
    return {
        "id": identity,
        "dataset": "allenai/c4",
        "split": "validation",
        "offset": offset,
        "samples": 48,
        "sequence_length": 512,
        "token_start": offset * 512,
        "token_end": (offset + 48) * 512,
        "token_hash": token_hash,
        "status": "reserved",
        "consumer": consumer,
    }


def _with_reservations(
    snapshot: dict[str, Any], entries: tuple[dict[str, object], dict[str, object]]
) -> dict[str, Any]:
    result = cast(dict[str, Any], json.loads(json.dumps(snapshot)))
    slices = result.get("slices")
    if not isinstance(slices, list):
        raise ValueError("evaluation slice registry has no slice inventory")
    new_ids = {str(entry["id"]) for entry in entries}
    if len(new_ids) != 2 or any(item.get("id") in new_ids for item in slices):
        raise ValueError("Experiment 048 slice identity already exists")
    slices.extend(entries)
    validate_registry(result)
    return result


def _tokens(snapshot: Path, c4_file: Path, offset: int) -> tuple[str, str, int | None]:
    tokens, fingerprint, bos_token_id = _load_c4_tokens(
        snapshot,
        revision="",
        data_file=str(c4_file),
        documents=1_100,
        offset=offset,
        samples=48,
        sequence_length=512,
        local_files_only=True,
    )
    return _token_hash(tokens), fingerprint, bos_token_id


def run(args: argparse.Namespace) -> int:
    snapshot = args.snapshot.resolve()
    c4_file = args.c4_file.resolve()
    registry = args.slice_registry.resolve()
    output = args.output.resolve()
    if (
        args.selection_slice_id == args.confirmation_slice_id
        or args.selection_offset < 0
        or args.confirmation_offset < 0
        or not snapshot.is_dir()
        or not c4_file.is_file()
    ):
        raise ValueError("Experiment 048 slice reservation inputs are invalid")
    selection_hash, selection_fingerprint, selection_bos = _tokens(snapshot, c4_file, args.selection_offset)
    confirmation_hash, confirmation_fingerprint, confirmation_bos = _tokens(snapshot, c4_file, args.confirmation_offset)
    if selection_fingerprint != confirmation_fingerprint or selection_bos != confirmation_bos:
        raise ValueError("Experiment 048 C4 slice tokenization identities differ")
    entries = (
        _entry(
            args.selection_slice_id,
            args.selection_offset,
            selection_hash,
            "experiment-048-correction-selection",
        ),
        _entry(
            args.confirmation_slice_id,
            args.confirmation_offset,
            confirmation_hash,
            "experiment-048-final-confirmation",
        ),
    )
    if output.is_file():
        intent = json.loads(output.read_text(encoding="utf-8"))
        protocol = intent.get("protocol")
        if (
            intent.get("schema_version") != 1
            or intent.get("status") != "authorized-for-reservation"
            or not isinstance(protocol, dict)
            or semantic_hash(protocol) != intent.get("protocol_hash")
            or protocol.get("entries") != list(entries)
            or protocol.get("c4_file_sha256") != _sha256(c4_file)
        ):
            raise ValueError("existing Experiment 048 reservation intent differs")
        base = protocol.get("registry_snapshot")
        if not isinstance(base, dict):
            raise ValueError("Experiment 048 reservation intent has no registry snapshot")
    else:
        encoded = registry.read_bytes()
        base = json.loads(encoded)
        validate_registry(base)
        reserved = _with_reservations(base, entries)
        protocol = {
            "experiment": 48,
            "role": "pre-model-evaluation-slice-reservation",
            "snapshot": str(snapshot),
            "c4_file": str(c4_file),
            "c4_file_sha256": _sha256(c4_file),
            "dataset_fingerprint": selection_fingerprint,
            "bos_token_id": selection_bos,
            "registry": str(registry),
            "registry_sha256_before": "sha256:" + hash_file(registry),
            "registry_snapshot": base,
            "registry_snapshot_hash": semantic_hash(base),
            "entries": list(entries),
            "registry_snapshot_after_hash": semantic_hash(reserved),
        }
        intent = {
            "schema_version": 1,
            "status": "authorized-for-reservation",
            "protocol_hash": semantic_hash(protocol),
            "protocol": protocol,
        }
        atomic_write_json(output, intent)
    reserved = _with_reservations(cast(dict[str, Any], base), entries)
    with _exclusive_registry_lock(registry):
        current = json.loads(registry.read_text(encoding="utf-8"))
        if current == base:
            atomic_write_json(registry, reserved)
        elif current != reserved:
            raise ValueError("slice registry changed outside the reservation intent")
        persisted = json.loads(registry.read_text(encoding="utf-8"))
        if persisted != reserved:
            raise RuntimeError("Experiment 048 slice reservations did not persist exactly")
    result = {
        "schema_version": 1,
        "status": "reserved",
        "intent": str(output),
        "intent_protocol_hash": intent["protocol_hash"],
        "registry_sha256_after": _sha256(registry),
        "selection": entries[0],
        "confirmation": entries[1],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
