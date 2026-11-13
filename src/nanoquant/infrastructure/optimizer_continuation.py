"""Explicit, recoverable import of a fixed checkpoint prefix into a new recipe.

Only small commit envelopes are rewrapped. Original commits, tensor references,
activation references, and a byte-exact source manifest/journal remain retained.
The caller must hold the run/device lease and validate the permitted recipe delta.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity
from nanoquant.infrastructure.io_utils import atomic_write_json, safe_replace


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".optimizer-continuation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        safe_replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def import_optimizer_prefix(
    output: Path, *, source_journal_sha256: str, source_manifest_config_hash: str,
    source_config_hash: str, target_config_hash: str, target_manifest: dict[str, Any], start_block: int,
) -> dict[str, Any]:
    root = output / "state" / "optimizer-continuation"
    receipt_path = root / "receipt.json"
    journal_path = output / "state" / "journal.jsonl"
    manifest_path = output / "manifest.json"
    preprocessing_path = output / "state" / "preprocessing.json"
    expected = {
        "source_journal_sha256": source_journal_sha256,
        "source_config_hash": source_config_hash, "target_config_hash": target_config_hash,
        "target_manifest_config_hash": target_manifest["config_hash"], "start_block": start_block,
    }
    if receipt_path.exists():
        receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != 1 or receipt.get("status") not in {"prepared", "complete"}:
            raise ValueError("unsupported optimizer continuation receipt")
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("optimizer continuation receipt does not match this recipe")
        if receipt["status"] == "complete":
            if json.loads(manifest_path.read_text(encoding="utf-8"))["config_hash"] != target_manifest["config_hash"]:
                raise ValueError("optimizer continuation manifest changed")
            target_prefix = (root / "target-journal.jsonl").read_bytes()
            if (_digest(target_prefix) != receipt["target_journal_sha256"]
                    or _digest((root / "source-journal.jsonl").read_bytes()) != source_journal_sha256
                    or not journal_path.read_bytes().startswith(target_prefix)):
                raise ValueError("optimizer continuation imported prefix changed")
            return receipt
    else:
        journal_bytes, manifest_bytes = journal_path.read_bytes(), manifest_path.read_bytes()
        if _digest(journal_bytes) != source_journal_sha256:
            raise ValueError("optimizer continuation source journal changed; refusing to discard progress")
        source_manifest = json.loads(manifest_bytes)
        if source_manifest["config_hash"] != source_manifest_config_hash:
            raise ValueError("optimizer continuation changes settings beyond the optimizer policy")
        if (output / "state" / "tuning-checkpoint" / "active.json").exists():
            raise ValueError("finish the active tuning checkpoint before importing an optimizer continuation")
        records = [json.loads(line) for line in journal_bytes.splitlines() if line.strip()]
        if not records or start_block <= 0 or target_config_hash == source_config_hash:
            raise ValueError("optimizer continuation requires a nonempty prefix and a distinct target identity")
        source_identity = CommitIdentity(**records[0]["identity"])
        if source_identity.config_hash != source_config_hash:
            raise ValueError("optimizer continuation source algorithm identity differs")
        target_identity = CommitIdentity(target_config_hash, source_identity.model_hash, source_identity.plan_hash)
        preprocessing_bytes = preprocessing_path.read_bytes()
        preprocessing = json.loads(preprocessing_bytes)
        if (preprocessing["resident_config_hash"] != source_config_hash
                or preprocessing["plan"]["artifact_id"] != source_identity.plan_hash):
            raise ValueError("optimizer continuation preprocessing identity differs")
        preprocessing["resident_config_hash"] = target_config_hash
        target_preprocessing = json.dumps(preprocessing, sort_keys=True, indent=2).encode("utf-8")
        blocks = [record["block"] for record in records if record["kind"] == "block"]
        if blocks != list(range(len(blocks))) or len(blocks) not in {start_block - 1, start_block}:
            raise ValueError("optimizer continuation requires the contiguous prefix at its declared boundary")
        for sequence, record in enumerate(records, 1):
            if (record["sequence"] != sequence or record["identity"] != to_dict(source_identity)
                    or record["block"] >= start_block or record["block"] > len(blocks)
                    or record["kind"] not in {"layer", "group", "block"}):
                raise ValueError("optimizer continuation source journal is inconsistent")
        store = LocalArtifactStore(output / "artifacts", use_persistent_validation_cache=False)
        store.validate(source_identity.plan_hash)
        imported = []
        mappings = []
        for record in records:
            descriptor = store.validate(record["artifact_id"])
            filename = {"layer": "layer-result.json", "group": "shared-input-group-result.json",
                        "block": "block-result.json"}[record["kind"]]
            payload = json.loads((store.path_for(record["artifact_id"]) / filename).read_text(encoding="utf-8"))
            if payload["identity"] != record["identity"]:
                raise ValueError("optimizer continuation commit identity differs from its journal")
            result = payload if record["kind"] == "block" else payload["result"]
            position = result["layer"] if record["kind"] == "layer" else result
            member = (position["path"] if record["kind"] == "layer"
                      else result["name"] if record["kind"] == "group" else None)
            if position["block"]["index"] != record["block"] or member != record["layer"]:
                raise ValueError("optimizer continuation commit position differs from its journal")
            payload["identity"] = to_dict(target_identity)
            payload["optimizer_continuation"] = {
                "source_journal_sha256": source_journal_sha256,
                "source_commit": {"artifact_id": descriptor.artifact_id,
                                  "artifact_type": descriptor.artifact_type},
                "numerical_payload": "unchanged; computed with the source optimizer",
            }
            with store.begin_write(descriptor.artifact_type) as writer:
                (writer.path / filename).write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
                imported_id = writer.commit().artifact_id
            imported.append({**record, "artifact_id": imported_id, "identity": to_dict(target_identity)})
            mappings.append({"source": record["artifact_id"], "imported": imported_id})
        target_journal = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                                 for record in imported).encode("utf-8")
        target_manifest_bytes = json.dumps(target_manifest, sort_keys=True, indent=2).encode("utf-8")
        root.mkdir(parents=True, exist_ok=True)
        for filename, data in (
            ("source-journal.jsonl", journal_bytes), ("source-manifest.json", manifest_bytes),
            ("target-journal.jsonl", target_journal), ("target-manifest.json", target_manifest_bytes),
            ("source-preprocessing.json", preprocessing_bytes), ("target-preprocessing.json", target_preprocessing),
        ):
            _write_bytes(root / filename, data)
        receipt = {
            "schema_version": 1, "status": "prepared", **expected,
            "source_manifest_sha256": _digest(manifest_bytes),
            "target_manifest_sha256": _digest(target_manifest_bytes),
            "target_journal_sha256": _digest(target_journal), "commits": mappings,
            "source_preprocessing_sha256": _digest(preprocessing_bytes),
            "target_preprocessing_sha256": _digest(target_preprocessing),
            "complete_blocks_preserved": blocks, "records_preserved": len(records),
        }
        atomic_write_json(receipt_path, receipt)

    # A durable prepared receipt makes interruption between these replaces
    # recoverable. Never start resident work until both replacements complete.
    for path, source_name, target_name, source_hash, target_hash in (
        (journal_path, "source-journal.jsonl", "target-journal.jsonl",
         source_journal_sha256, receipt["target_journal_sha256"]),
        (manifest_path, "source-manifest.json", "target-manifest.json",
         receipt["source_manifest_sha256"], receipt["target_manifest_sha256"]),
        (preprocessing_path, "source-preprocessing.json", "target-preprocessing.json",
         receipt["source_preprocessing_sha256"], receipt["target_preprocessing_sha256"]),
    ):
        original, target = (root / source_name).read_bytes(), (root / target_name).read_bytes()
        if _digest(original) != source_hash or _digest(target) != target_hash:
            raise ValueError("optimizer continuation transaction data changed")
        if _digest(path.read_bytes()) not in {source_hash, target_hash}:
            raise ValueError("optimizer continuation current state changed during import")
        _write_bytes(path, target)
    receipt["status"] = "complete"
    atomic_write_json(receipt_path, receipt)
    return receipt
