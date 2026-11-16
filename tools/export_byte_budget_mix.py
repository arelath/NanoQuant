"""Build a diagnostic mixed GGUF by copying a byte allocation's exact source payloads.

This is a post-hoc research export, not a completed resident compression run.
Embeddings and all other unplanned tensors are byte-identical to the staged source.
Grouped QKV changes are atomic; a binary group may retain a prefix of its stages.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import gguf
import numpy as np
from materialize_nanoquant_gguf import _gemma_qkv_sizes

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file


def select_payloads(reader, standard: list, units: dict, choices: dict) -> tuple[dict, dict, set[str]]:
    original = {tensor.name: tensor for tensor in reader.tensors}
    expected_owners = {name.removesuffix(".nq_stage0_v") for name in original if name.endswith(".nq_stage0_v")}
    if set(units) != expected_owners or set(choices) != expected_owners:
        raise ValueError("allocation must cover exactly the staged source's storage units")
    payloads = dict(original)
    stage_counts = {}
    standard_owners = set()
    for owner, label in choices.items():
        entry = units[owner]
        sidecars = {name for name in original if name.startswith(owner + ".nq_")}
        count = int(reader.fields[f"quantization.nanoquant.{owner}.residual_stage_count"].contents()) + 1
        if label.endswith((" stage", " stages")):
            keep = int(label.split()[0])
            if not 1 <= keep <= count:
                raise ValueError(f"invalid stage prefix {label}: {owner}")
            prefixes = (owner + ".nq_stage0_", *(f"{owner}.nq_residual_{i}_" for i in range(keep - 1)))
            for name in sidecars:
                if not name.startswith(prefixes):
                    del payloads[name]
                elif keep == 1:
                    # Match the existing converter/native contract: stage0 denotes
                    # a residual owner, which must have at least one residual.
                    payloads[name.replace(".nq_stage0_", ".nq_")] = payloads.pop(name)
            stage_counts[owner] = keep - 1
            continue
        kinds = label.split()
        members = entry["members"]
        if len(kinds) != len(members):
            raise ValueError(f"standard choice must cover every group member: {owner}")
        columns = int(original[owner + ".nq_stage0_scale_pre"].n_elements)
        rows = ([int(original[owner + ".nq_stage0_scale_post"].n_elements)] if len(members) == 1
                else list(_gemma_qkv_sizes(reader.fields, int(owner.split(".")[1]))))
        for name, kind, height in zip(members, kinds, rows, strict=True):
            matches = [tensors[name] for tensors in standard
                       if name in tensors and tensors[name].tensor_type.name == kind]
            if not matches:
                raise ValueError(f"no source payload for {name} at {kind}")
            tensor = matches[0]
            if list(tensor.shape) != [columns, height]:
                raise ValueError(f"replacement shape differs: {name}")
            if int(tensor.n_bytes) != entry["standard"][name][kind]["bytes"]:
                raise ValueError(f"replacement bytes differ from menu: {name}")
            payloads[name] = tensor
        for name in sidecars:
            del payloads[name]
        standard_owners.add(owner)
    return payloads, stage_counts, standard_owners


def validate_payloads(path: Path, expected: dict, stage_counts: dict, standard_owners: set[str]) -> None:
    reader = gguf.GGUFReader(path)
    actual = {tensor.name: tensor for tensor in reader.tensors}
    if set(actual) != set(expected):
        raise ValueError("mixed export tensor inventory differs from plan")
    for name, tensor in actual.items():
        wanted = expected[name]
        if (tensor.tensor_type != wanted.tensor_type or not np.array_equal(tensor.shape, wanted.shape)
                or not np.array_equal(tensor.data, wanted.data)):
            raise ValueError(f"mixed export changed source payload: {name}")
    for owner, count in stage_counts.items():
        field = reader.fields.get(f"quantization.nanoquant.{owner}.residual_stage_count")
        if (count == 0 and field is not None) or (count > 0 and (field is None or field.contents() != count)):
            raise ValueError(f"incorrect exported stage count: {owner}")
    if int(reader.fields["quantization.nanoquant.tensor_count"].contents()) != len(stage_counts):
        raise ValueError("incorrect exported NanoQuant owner count")
    for owner in standard_owners:
        if any(name.startswith(f"quantization.nanoquant.{owner}.") for name in reader.fields):
            raise ValueError(f"stale NanoQuant metadata: {owner}")


def export_mix(source: Path, standard_paths: list[Path], units: dict, choices: dict, output: Path, budget: int) -> dict:
    if output.exists():
        raise ValueError(f"preserving existing output: {output}")
    reader = gguf.GGUFReader(source)
    readers = [gguf.GGUFReader(path) for path in standard_paths]
    architecture = reader.fields["general.architecture"].contents()
    if any(item.endianess != reader.endianess or item.fields["general.architecture"].contents() != architecture
           for item in readers):
        raise ValueError("GGUF source architecture or endianness differs")
    payloads, counts, removed = select_payloads(
        reader, [{tensor.name: tensor for tensor in item.tensors} for item in readers], units, choices,
    )
    writer = gguf.GGUFWriter(None, architecture, endianess=reader.endianess)
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if any(name.startswith(f"quantization.nanoquant.{owner}.") for owner in removed):
            continue
        if any(count == 0 and name == f"quantization.nanoquant.{owner}.residual_stage_count"
               for owner, count in counts.items()):
            continue
        value = field.contents()
        if name == "general.alignment":
            writer.add_custom_alignment(int(value))
            continue
        if name == "quantization.nanoquant.tensor_count":
            value = len(counts)
        for owner, count in counts.items():
            if name == f"quantization.nanoquant.{owner}.residual_stage_count":
                value = count
        kind = field.types[0]
        writer.add_key_value(name, value, kind, sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    for name, tensor in sorted(payloads.items()):
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mixed-", dir=output.parent) as temporary:
        path = Path(temporary) / output.name
        writer.write_header_to_file(path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        validate_payloads(path, payloads, counts, removed)
        if path.stat().st_size > budget:
            raise ValueError(f"actual GGUF exceeds byte budget: {path.stat().st_size} > {budget}")
        path.replace(output)
    return {
        "source": str(source.resolve()), "source_sha256": hash_file(source),
        "standard_sources": [{"path": str(path.resolve()), "sha256": hash_file(path)} for path in standard_paths],
        "output": str(output.resolve()), "sha256": hash_file(output), "bytes": output.stat().st_size,
        "budget": budget, "choices": choices, "binary_units": len(counts), "standard_units": len(removed),
        "validation": "every stored tensor type, shape and payload equals its selected source; stage metadata checked",
        "embedding_policy": "unchanged source payload", "kind": "post-hoc diagnostic; no resident refitting",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("menu", type=Path)
    parser.add_argument("allocation", type=Path)
    parser.add_argument("--solution", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    menu = json.loads(args.menu.read_text(encoding="utf-8"))
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    if allocation["menu_sha256"] != hash_file(args.menu):
        raise ValueError("allocation is bound to a different menu")
    solution = allocation["solutions"][args.solution]
    receipt = export_mix(Path(menu["staged"]), [Path(path) for path in menu["standard"]], menu["units"],
                         solution["choices"], args.output, int(solution["budget"]))
    receipt.update({"menu_sha256": hash_file(args.menu), "allocation_sha256": hash_file(args.allocation),
                    "predicted_kl": solution["kl"], "predicted_bytes": solution["bytes"]})
    atomic_write_json(args.output.with_suffix(".receipt.json"), receipt)
    print(json.dumps({key: receipt[key] for key in ("output", "bytes", "binary_units", "standard_units")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
