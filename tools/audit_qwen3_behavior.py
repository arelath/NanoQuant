"""Audit Qwen3 prompt prefixes and an optional prepared behavior artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.config.schema import ReasoningMode
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.chat_behaviors import chat_behavior_for_snapshot
from nanoquant.infrastructure.hf_calibration_dataset import (
    CALIBRATION_RECEIPT_NAME,
    load_pinned_calibration,
)
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.ports.chat_behavior import ReasoningModeId, TokenRole


def audit(snapshot: Path, run_output: Path | None) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=False)
    behavior = chat_behavior_for_snapshot(snapshot)
    messages = [{"role": "user", "content": "State the result of 2 + 2."}]
    prefixes = {
        mode.value: {
            "token_ids": behavior.render_generation_prompt(tokenizer, messages, mode),
            "rendered": tokenizer.decode(
                behavior.render_generation_prompt(tokenizer, messages, mode)
            ),
        }
        for mode in (ReasoningMode.THINKING, ReasoningMode.NON_THINKING)
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "snapshot": str(snapshot.resolve()),
        "chat_policy_identity": behavior.policy_identity(tokenizer),
        "prefixes": prefixes,
    }
    if run_output is None:
        return payload
    receipt = json.loads((run_output / CALIBRATION_RECEIPT_NAME).read_text(encoding="utf-8"))
    calibration = load_pinned_calibration(
        run_output,
        ArtifactRef("calibration-dataset-manifest", str(receipt["artifact_id"]), 1),
    )
    if calibration.reasoning_mode_ids is None or calibration.token_role_ids is None:
        payload["prepared_artifact"] = {
            "behavior_profile": calibration.behavior_profile,
            "mode_aware": False,
        }
        return payload
    valid = calibration.attention_mask.bool()
    modes = calibration.reasoning_mode_ids
    roles = calibration.token_role_ids
    payload["prepared_artifact"] = {
        "artifact_id": calibration.reference.artifact_id,
        "fingerprint": calibration.fingerprint,
        "behavior_profile": calibration.behavior_profile,
        "mode_aware": True,
        "valid_tokens_by_mode": {
            mode.name.lower(): int(((modes == int(mode)) & valid).sum())
            for mode in ReasoningModeId
            if mode is not ReasoningModeId.PADDING
        },
        "valid_tokens_by_role": {
            role.name.lower(): int(((roles == int(role)) & valid).sum())
            for role in TokenRole
            if role is not TokenRole.PADDING
        },
        "distillation_target_tokens": (
            None
            if calibration.distillation_target_mask is None
            else int(calibration.distillation_target_mask.sum())
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args.snapshot, args.run_output)
    if args.output is not None:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
