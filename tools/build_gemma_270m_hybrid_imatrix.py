"""Combine Unsloth activation columns with our Eaddario activation rows.

Copy the raw sums and counts independently, preserving both source axes exactly.
No calibration, gradient collection, normalization change, or GPU work is done.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter

from nanoquant.config.codec import to_dict
from nanoquant.config.schema import SharedInputGroupConfig
from nanoquant.infrastructure.imatrix import imatrix_config, load_imatrix, resolve_imatrix_axes
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.safetensors_source import SafetensorsModelSource

ROOT = Path(__file__).resolve().parents[1]
MODEL = "unsloth/gemma-3-270m-it"
REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
COLUMN_HASH = "2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3"
ROW_HASH = "3deceafea3aadd5fb6b75e22dfa0123bb99f22e0be95816354b374e80ebcc0c9"
GROUPS = (SharedInputGroupConfig("self_attn.attn_qkv",
                               ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),)


def raw_tensors(path):
    return {tensor.name: tensor.data.copy() for tensor in GGUFReader(str(path), "r").tensors}


def validate_axes(path, columns_path, rows_path):
    actual, columns, rows = (load_imatrix(p) for p in (path, columns_path, rows_path))
    if set(actual.entries) != set(columns.entries) or set(actual.row_entries) != set(rows.row_entries):
        raise ValueError("hybrid tensor inventory differs from source axes")
    actual_raw, column_raw, row_raw = (raw_tensors(p) for p in (path, columns_path, rows_path))
    for name in actual.entries:
        if not torch.equal(actual.entries[name], columns.entries[name]):
            raise ValueError(f"hybrid columns changed: {name}")
        if not torch.equal(actual.row_entries[name], rows.row_entries[name]):
            raise ValueError(f"hybrid rows changed: {name}")
        for suffix, source in ((".in_sum2", column_raw), (".counts", column_raw),
                               (".out_sum2", row_raw), (".out_counts", row_raw)):
            key = name + suffix
            if not np.array_equal(actual_raw[key], source[key]):
                raise ValueError(f"hybrid raw sums/counts changed: {key}")


def write_hybrid(path, columns_path, rows_path, *, columns_sha256=None, rows_sha256=None):
    columns = load_imatrix(columns_path, columns_sha256)
    rows = load_imatrix(rows_path, rows_sha256)
    if columns.format != "gguf" or rows.format != "gguf":
        raise ValueError("hybrid builder requires GGUF sources with explicit axis counts")
    if set(columns.entries) != set(rows.row_entries):
        raise ValueError("column and row source inventories differ")
    if path.exists():
        raise FileExistsError(path)
    column_raw, row_raw = raw_tensors(columns_path), raw_tensors(rows_path)
    temporary = path.with_suffix(".partial.gguf")
    writer = GGUFWriter(str(temporary), "imatrix")
    writer.add_string("general.type", "imatrix")
    writer.add_string("nanoquant.imatrix.statistic", "hybrid-activation-columns-and-activation-rows")
    writer.add_string("nanoquant.imatrix.column_source_sha256", columns.sha256)
    writer.add_string("nanoquant.imatrix.row_source_sha256", rows.sha256)
    writer.add_string("nanoquant.imatrix.count_semantics", "independent original source sums and counts per axis")
    for name in sorted(columns.entries):
        for suffix, source in ((".in_sum2", column_raw), (".counts", column_raw),
                               (".out_sum2", row_raw), (".out_counts", row_raw)):
            writer.add_tensor(name + suffix, source[name + suffix])
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    validate_axes(temporary, columns_path, rows_path)
    temporary.replace(path)


def run(args):
    columns = load_imatrix(args.columns, args.columns_sha256)
    rows = load_imatrix(args.rows, args.rows_sha256)
    source = SafetensorsModelSource(args.snapshot, source=MODEL, revision=REVISION, verify_hashes=True)
    inventory = adapter_for_config(source.inventory().config).model_inventory(source)
    identity = dict(schema=1, model=MODEL, revision=REVISION, method="copy-original-axis-sums-and-counts-v1",
                    model_weights_sha256=hash_file(args.snapshot / "model.safetensors"),
                    columns=dict(path=str(args.columns.resolve()), sha256=columns.sha256,
                                 receipt=json.loads(args.columns.with_name("source.json").read_text())),
                    rows=dict(path=str(args.rows.resolve()), sha256=rows.sha256,
                              receipt=json.loads(args.rows.with_name("source.json").read_text())))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "imatrix.gguf"
    receipt_path = args.output / "source.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or hash_file(path) != receipt["sha256"]:
            raise ValueError("existing hybrid has incompatible provenance or changed bytes")
    else:
        write_hybrid(path, args.columns, args.rows, columns_sha256=columns.sha256, rows_sha256=rows.sha256)
    validate_axes(path, args.columns, args.rows)
    config = imatrix_config(path)
    inputs, outputs, validation = resolve_imatrix_axes(config, inventory, GROUPS)
    if len(inputs) != 126 or len(outputs) != 126 or validation["unused_entries"]:
        raise ValueError("hybrid must cover all 126 projections")
    validation.update(column_axis_exact=True, row_axis_exact=True, raw_sums_and_counts_exact=True,
                      input_channels=sum(v.numel() for v in inputs.values()),
                      output_rows=sum(v.numel() for v in outputs.values()))
    if not receipt_path.exists():
        atomic_write_json(args.output / "validation.json", validation)
        atomic_write_json(receipt_path, dict(identity=identity, sha256=config.sha256, bytes=path.stat().st_size,
                                            producer_sha256=hash_file(Path(__file__)),
                                            calibration={"imatrix": to_dict(config)}))
    print(json.dumps(dict(path=str(path), sha256=config.sha256, bytes=path.stat().st_size,
                          projections=len(inputs), use_rows=config.use_rows), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=Path, default=ROOT / "Datasets/gemma-3-270m-it-imatrix-unsloth/"
                        "imatrix_unsloth.gguf_file")
    parser.add_argument("--columns-sha256", default=COLUMN_HASH)
    parser.add_argument("--rows", type=Path,
                        default=ROOT / "Datasets/gemma-3-270m-it-imatrix-eaddario-huge-rows/imatrix.gguf")
    parser.add_argument("--rows-sha256", default=ROW_HASH)
    parser.add_argument("--snapshot", type=Path, default=Path.home() / ".cache/huggingface/hub/"
                        f"models--unsloth--gemma-3-270m-it/snapshots/{REVISION}")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Datasets/gemma-3-270m-it-imatrix-unsloth-columns-eaddario-rows")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
