"""Run Experiment 054 in clean one-block process slices until completion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_INTERRUPTION_MARKER = "injected interruption after 1 new block commits"
_SLICE_LOG_PREFIX = "campaign-slice-"
_RUN_NAME = "054-functional-binary-lr-d2-compress-and-benchmark-gemma-3-1b-it"


def _completed_blocks(run_root: Path) -> int:
    journal = run_root / "state" / "journal.jsonl"
    if not journal.exists():
        return 0
    completed = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("kind") == "block":
            completed += 1
    return completed


def _next_slice_index(campaign_root: Path) -> int:
    """Continue after retained slice logs instead of overwriting evidence."""

    indexes: list[int] = []
    for path in campaign_root.glob(f"{_SLICE_LOG_PREFIX}*.log"):
        stem = path.name.removeprefix(_SLICE_LOG_PREFIX).split(".", 1)[0]
        if stem.isdigit():
            indexes.append(int(stem))
    return max(indexes, default=0) + 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-slices", type=int, default=64)
    parser.add_argument("--process-cooldown-seconds", type=float, default=5.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.maximum_slices <= 0:
        raise ValueError("maximum slices must be positive")
    if args.process_cooldown_seconds < 0:
        raise ValueError("process cooldown must be non-negative")
    repository = Path(__file__).resolve().parent.parent
    campaign_root = repository / "evidence" / "054"
    run_root = campaign_root / _RUN_NAME
    launcher = repository / "experiments" / f"{_RUN_NAME}.py"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    first_slice = _next_slice_index(campaign_root)
    if first_slice > args.maximum_slices:
        raise RuntimeError(
            f"Experiment 054 already used {first_slice - 1} of {args.maximum_slices} allowed slices"
        )
    for index in range(first_slice, args.maximum_slices + 1):
        before = _completed_blocks(run_root)
        stdout_path = campaign_root / f"campaign-slice-{index:03d}.stdout.log"
        stderr_path = campaign_root / f"campaign-slice-{index:03d}.stderr.log"
        if stdout_path.exists() or stderr_path.exists():
            raise FileExistsError(f"campaign slice log already exists: {index}")
        print(f"starting Experiment 054 slice {index}; completed blocks={before}", flush=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                [sys.executable, str(launcher)],
                cwd=repository,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        after = _completed_blocks(run_root)
        if completed.returncode == 0:
            print(f"Experiment 054 completed in slice {index}; completed blocks={after}", flush=True)
            return 0
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if _INTERRUPTION_MARKER not in stderr_text:
            print(stderr_text[-4000:], file=sys.stderr)
            return completed.returncode
        if after <= before:
            raise RuntimeError("block-bounded slice interrupted without a new durable block")
        print(f"slice {index} committed a block; completed blocks={after}", flush=True)
        if args.process_cooldown_seconds:
            time.sleep(args.process_cooldown_seconds)
    raise RuntimeError("Experiment 054 exceeded its block-bounded slice limit")


if __name__ == "__main__":
    raise SystemExit(main())
