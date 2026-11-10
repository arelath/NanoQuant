"""CPU reconstruction pilot with equal byte caps and round-tripped diagnostic payloads.

This diagnostic container is not a NanoQuant deployment format. No functional
quality conclusion follows from this single-matrix screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import torch
from safetensors.torch import load_file

from nanoquant.domain.factorization import AdmmParameters, factorize_admm_with_parameters
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.io_utils import atomic_write_json

HEADER_BYTES = 4096
ALIGNMENT = 32


def aligned(size: int) -> int:
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def encode_payload(tensors: dict[str, torch.Tensor]) -> bytes:
    """Pack signs at one bit; charge a fixed header and each array's alignment."""
    entries = []
    chunks = []
    offset = HEADER_BYTES
    for name, tensor in sorted(tensors.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        if name.endswith("binary"):
            if not np.isin(array, [-1, 1]).all():
                raise ValueError("binary tensors must contain only -1 and +1")
            raw = np.packbits(array.reshape(-1) > 0, bitorder="little").tobytes()
            dtype = "sign1"
        else:
            if array.dtype not in (np.dtype("float32"), np.dtype("int32"), np.dtype("int8")):
                raise ValueError(f"unsupported persisted dtype: {array.dtype}")
            raw = array.tobytes()
            dtype = array.dtype.str
        entries.append(dict(name=name, shape=list(array.shape), dtype=dtype, offset=offset, bytes=len(raw)))
        chunks.append(raw + bytes(aligned(len(raw)) - len(raw)))
        offset += aligned(len(raw))
    header = json.dumps(dict(schema="residual-pilot-v1", tensors=entries), sort_keys=True).encode()
    if len(header) + 4 > HEADER_BYTES:
        raise ValueError("diagnostic header exceeds reservation")
    return struct.pack("<I", len(header)) + header + bytes(HEADER_BYTES - 4 - len(header)) + b"".join(chunks)


def decode_payload(payload: bytes) -> dict[str, torch.Tensor]:
    length = struct.unpack("<I", payload[:4])[0]
    header = json.loads(payload[4:4 + length])
    result = {}
    for entry in header["tensors"]:
        raw = payload[entry["offset"]:entry["offset"] + entry["bytes"]]
        if entry["dtype"] == "sign1":
            values = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
            count = int(np.prod(entry["shape"]))
            array = (values[:count].astype(np.int8) * 2 - 1).reshape(entry["shape"])
        else:
            array = np.frombuffer(raw, dtype=np.dtype(entry["dtype"])).reshape(entry["shape"]).copy()
        result[entry["name"]] = torch.from_numpy(array)
    return result


def factor_bytes(m: int, n: int, rank: int) -> int:
    return sum(aligned(size) for size in ((m * rank + 7) // 8, (rank * n + 7) // 8,
                                         4 * m, 4 * n, 4 * rank))


def correction_bytes(kind: str, m: int, count: int) -> int:
    if count == 0:
        return 0
    if kind == "int8":
        return aligned(m * count) + 2 * aligned(4 * count)
    if kind == "sign":
        return aligned((m * count + 7) // 8) + aligned(4 * m) + 2 * aligned(4 * count)
    raise ValueError(kind)


def reconstruct(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    def factor(prefix: str) -> torch.Tensor:
        left = tensors[prefix + "left_binary"].double() * tensors[prefix + "scale_post"].double()[:, None]
        right = (tensors[prefix + "right_binary"].double()
                 * tensors[prefix + "scale_mid"].double()[:, None]
                 * tensors[prefix + "scale_pre"].double()[None, :])
        return left @ right

    weight = factor("first.")
    if "second.left_binary" in tensors:
        weight += factor("second.")
    if "correction.indices" in tensors:
        indices = tensors["correction.indices"].long()
        if "correction.binary" in tensors:
            correction = (tensors["correction.binary"].double()
                          * tensors["correction.row_scale"].double()[:, None]
                          * tensors["correction.column_scale"].double()[None, :])
        else:
            correction = tensors["correction.int8"].double() * tensors["correction.column_scale"].double()[None, :]
        weight[:, indices] += correction
    weight[:, tensors["protected.indices"].long()] = tensors["protected.values"].double()
    return weight


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = load_file(str(args.fixture))
    first = load_file(str(args.first))
    receipt_path = args.fixture.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    fixture_hash = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    if fixture_hash != receipt["fixture_sha256"]:
        raise ValueError("fixture hash mismatch")
    m, n = fixture["target"].shape
    protected = fixture["outlier_indices"].long()
    active = torch.ones(n, dtype=torch.bool)
    active[protected] = False
    target = fixture["target"].double()
    target[:, protected] = fixture["outlier_values"].double()
    base = {"first." + k: v for k, v in first.items()}
    base.update({"protected.indices": protected.int(), "protected.values": fixture["outlier_values"].float()})
    base_payload = encode_payload(base)
    base_prediction = reconstruct(decode_payload(base_payload))
    residual = target - base_prediction
    u, v = fixture["output_importance"].double(), fixture["input_importance"].double()
    denominator = float((target[:, active].square() * u[:, None] * v[None, active]).sum())
    cap = int(args.bpw * m * n / 8)
    available = cap - len(base_payload)
    rank = max((r for r in range(32, min(m, n) + 1, 32) if factor_bytes(m, n, r) <= available), default=0)
    if rank == 0:
        raise ValueError("budget cannot fit a residual stage")
    summary = dict(scope="single-matrix reconstruction pilot; not an Experiment 060 or functional comparison",
                   fixture_sha256=fixture_hash, first_sha256=hashlib.sha256(args.first.read_bytes()).hexdigest(),
                   tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_receipt=receipt, shape=[m, n], seed=args.seed, iterations=args.iterations,
                   cap_bytes=cap, cap_bpw=args.bpw, residual_rank=rank, protected_columns=len(protected),
                   metric="diagonal-weighted NMSE on unprotected columns; protected columns restored exactly",
                   format="diagnostic packed container, 4096-byte header, 32-byte array alignment, FP32 scales",
                   results={}, status="running")

    def save(name: str, state: dict[str, torch.Tensor], seconds: float) -> None:
        payload = encode_payload(state)
        if len(payload) > cap:
            raise ValueError(f"{name} exceeds byte cap")
        path = args.output / (name + ".bin")
        path.write_bytes(payload)
        decoded = decode_payload(path.read_bytes())
        prediction = reconstruct(decoded)
        error = target[:, active] - prediction[:, active]
        nmse = float((error.square() * u[:, None] * v[None, active]).sum()) / denominator
        if not np.isfinite(nmse):
            raise ValueError("nonfinite result")
        assert torch.equal(prediction[:, protected], target[:, protected])
        summary["results"][name] = dict(bytes=len(payload), bpw=len(payload) * 8 / (m * n),
                                        slack_bytes=cap - len(payload), weighted_nmse=nmse,
                                        seconds=seconds, sha256=hashlib.sha256(payload).hexdigest())
        atomic_write_json(args.output / "summary.json", summary)
        print(json.dumps({name: summary["results"][name]}), flush=True)

    save("first_only", base, 0)
    scores = (residual.square() * u[:, None] * v[None, :]).sum(0)
    scores[protected] = -torch.inf
    ordering = torch.argsort(scores, descending=True, stable=True)
    for kind in ("int8", "sign"):
        start = time.perf_counter()
        count = max(k for k in range(n - len(protected) + 1) if correction_bytes(kind, m, k) <= available)
        indices = ordering[:count]
        part = residual[:, indices]
        state = dict(base)
        if count:
            state["correction.indices"] = indices.int()
            if kind == "int8":
                scale = (part.abs().amax(0) / 127).clamp_min(1e-30)
                state["correction.int8"] = torch.round(part / scale).clamp(-127, 127).to(torch.int8)
                state["correction.column_scale"] = scale.float()
            else:
                absolute = part.abs() * u.sqrt()[:, None] * v[indices].sqrt()[None, :]
                column = absolute.mean(0)
                for _ in range(40):
                    row = absolute @ column / column.square().sum().clamp_min(1e-30)
                    column = absolute.T @ row / row.square().sum().clamp_min(1e-30)
                state["correction.binary"] = torch.where(part >= 0, 1, -1).to(torch.int8)
                state["correction.row_scale"] = (row / u.sqrt().clamp_min(1e-30)).float()
                state["correction.column_scale"] = (column / v[indices].sqrt().clamp_min(1e-30)).float()
        summary[kind + "_columns"] = count
        save(kind, state, time.perf_counter() - start)
    start = time.perf_counter()
    fit = factorize_admm_with_parameters(
        residual.float(), v.float(), u.float(), rank, torch.Generator(device="cpu").manual_seed(args.seed),
        AdmmParameters(outer_iterations=args.iterations, transpose_wide=False),
    )
    scales = fit_scales(residual.float(), fit.left_binary, fit.right_binary, fit.scale_pre, fit.scale_mid,
                        fit.scale_post, v.float(), u.float(), alternating_passes=2)
    state = dict(base)
    for name in ("left_binary", "right_binary"):
        state["second." + name] = getattr(fit, name).to(torch.int8)
    for name in ("scale_pre", "scale_mid", "scale_post"):
        state["second." + name] = getattr(scales, name).float()
    save("residual_factor", state, time.perf_counter() - start)
    summary["status"] = "complete_reconstruction_only"
    atomic_write_json(args.output / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path("evidence/066/hour-fit/fixture.safetensors"))
    parser.add_argument("--first", type=Path, default=Path("evidence/067/basin-search/best.safetensors"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bpw", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=12)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
