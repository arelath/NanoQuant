"""Stable logical seed derivation independent of execution order."""

import hashlib


def logical_seed(
    run_seed: int, stage: str, block: int | None = None, layer: str | None = None, attempt: int | None = None
) -> int:
    payload = "\0".join(str(item) for item in (run_seed, stage, block, layer, attempt))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
