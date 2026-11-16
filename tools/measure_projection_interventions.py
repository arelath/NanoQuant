"""Score resumable, explicit GGUF projection swaps against a pinned sensitivity teacher.

The JSON plan lists named GGUF sources and tasks. A task has a base ("clean" or a
source name) and replacements mapping source names to sweep target lists. All
embeddings and norms stay at the clean teacher's weights. A source may specify
"stages": 1 to measure a truncated staged export, explicitly not a new fitted run.
Completed tasks are atomically saved and reused only with identical inputs and code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

import gguf
import numpy as np
import torch
from materialize_nanoquant_gguf import _gemma_qkv_sizes
from measure_gguf_kl import PROJECTIONS, map_tensors, read_gguf, swapped, sweep_session
from measure_joint_kl_targets import projection_weights
from probe_byte_budget_allocation import MEMBERS, staged_terms

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan: dict, targets: list[str]) -> None:
    sources = set(plan["sources"])
    if "clean" in sources:
        raise ValueError("clean is reserved for the unchanged teacher")
    names = [task["name"] for task in plan["tasks"]]
    if len(names) != len(set(names)):
        raise ValueError("task names must be unique")
    known = set(targets)
    for task in plan["tasks"]:
        if task["base"] not in sources | {"clean"}:
            raise ValueError(f"unknown base: {task['base']}")
        replaced = set()
        for source, selected in task.get("replacements", {}).items():
            if source not in sources or not selected or not set(selected) <= known:
                raise ValueError(f"invalid replacement: {source}, {selected}")
            if len(selected) != len(set(selected)) or replaced.intersection(selected):
                raise ValueError("a projection may be replaced only once per task")
            replaced.update(selected)


def load_source(spec: dict, targets: list[str]) -> dict[str, torch.Tensor]:
    path = Path(spec["path"])
    limit = spec.get("stages")
    if limit is None:
        projections, _, _ = map_tensors(read_gguf(path), targets)
        return {name: item["values"] for name, item in projections.items()}
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("stages must be a positive integer")
    reader = gguf.GGUFReader(path)
    result = {}
    for base, terms in staged_terms(reader).items():
        if limit > len(terms):
            raise ValueError(f"{base} has only {len(terms)} stages")
        block, kind = base.split(".")[1:]
        dense = sum((weight for weight, _ in terms[:limit]), np.zeros_like(terms[0][0]))
        members = MEMBERS.get(kind, (kind,))
        sizes = np.cumsum(_gemma_qkv_sizes(reader.fields, int(block)))[:-1] if kind in MEMBERS else []
        for member, value in zip(members, np.split(dense, sizes, axis=0), strict=True):
            result[f"{block}:{PROJECTIONS[member]}"] = torch.from_numpy(value.copy())
    if set(result) != set(targets):
        raise ValueError("truncated source does not match sweep projection inventory")
    return result


def render(report: dict) -> str:
    clean = report["clean"]
    lines = [
        "# Projection intervention measurements", "",
        f"Completed {len(report['results'])}/{report['expected_tasks']} tasks on "
        f"{clean['token_count']} next-token positions per task.",
        "Full-vocabulary KL(teacher || candidate), in nats/token, using the sensitivity teacher's precision.",
        "Embeddings, tied output and norms remain at clean teacher weights in every task.",
        "These are post-hoc projection swaps through HF dense execution, not native runtime measurements.",
        f"Clean-control KL: {clean['kl_nats_per_token']:.9g}.", "",
        "| Task | Base | KL above clean | NLL above clean | Seconds |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, item in report["results"].items():
        score = item["score"]
        lines.append(
            f"| {name} | {item['task']['base']} | "
            f"{score['kl_nats_per_token'] - clean['kl_nats_per_token']:.7f} | "
            f"{score['nll_nats_per_token'] - clean['nll_nats_per_token']:+.7f} | {item['seconds']:.1f} |"
        )
    lines += ["", "JSON contains task definitions, input hashes and every window's paired KL/NLL scores.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    sweep = json.loads(args.measurements.read_text(encoding="utf-8"))
    validate_plan(plan, sweep["protocol"]["targets"])
    dependencies = [
        Path(__file__), Path(__file__).with_name("measure_gguf_kl.py"),
        Path(__file__).with_name("kl_sensitivity_dataset.py"),
        Path(__file__).with_name("materialize_nanoquant_gguf.py"),
        Path(__file__).with_name("probe_byte_budget_allocation.py"),
    ]
    identity = {
        "measurement_sha256": hash_file(args.measurements),
        "sources": {name: {**spec, "sha256": hash_file(Path(spec["path"]))}
                    for name, spec in plan["sources"].items()},
        "code": {path.name: hash_file(path) for path in dependencies},
        "embedding_policy": "clean-fixed",
    }
    output = args.output.resolve().with_suffix(".json")
    prior = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("output identity differs; choose a new output to preserve existing evidence")
    if prior is not None:
        for task in plan["tasks"]:
            old = prior["results"].get(task["name"])
            if old is not None and old["task"] != task:
                raise ValueError(f"completed task changed: {task['name']}")
    with sweep_session(args.measurements, local_files_only=True) as session:
        report = {
            "schema_version": 1, "identity": identity, "protocol": session.protocol,
            "clean": session.clean, "expected_tasks": len(plan["tasks"]),
            "results": {} if prior is None else prior["results"],
        }
        weights = projection_weights(session.model, session.adapter, session.targets)
        cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

        def source(name: str) -> dict[str, torch.Tensor]:
            if name not in cache:
                while len(cache) >= 2:
                    cache.popitem(last=False)
                print(f"Loading {name}", flush=True)
                cache[name] = load_source(plan["sources"][name], session.targets)
            cache.move_to_end(name)
            return cache[name]

        def save() -> None:
            atomic_write_json(output, report)
            atomic_write_text(output.with_suffix(".md"), render(report))

        save()
        for index, task in enumerate(plan["tasks"], 1):
            if task["name"] in report["results"]:
                print(f"Reusing {task['name']}", flush=True)
                continue
            started = time.perf_counter()
            with ExitStack() as stack:
                if task["base"] != "clean":
                    stack.enter_context(swapped({name: (weights[name], value)
                                                 for name, value in source(task["base"]).items()}))
                for label, targets in task.get("replacements", {}).items():
                    values = source(label)
                    stack.enter_context(swapped({name: (weights[name], values[name]) for name in targets}))
                score = session.score()
            report["results"][task["name"]] = {
                "task": task, "score": score, "seconds": time.perf_counter() - started,
            }
            save()
            print(f"{index}/{len(plan['tasks'])} {task['name']}: "
                  f"KL={score['kl_nats_per_token']:.7f}, {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
