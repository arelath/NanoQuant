"""Compare Gemma staged-rank experiments on one KL teacher and their matched benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from gguf import GGUFReader
from measure_gguf_kl import measure_file, sweep_session

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file

ROOT = Path(__file__).resolve().parents[1]


def load_result(number: str) -> tuple[Path, dict, str]:
    directory = ROOT / "Results" / number
    (quality_path,) = directory.glob(f"{number}-*-compress-and-benchmark-*-quality.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    path = directory / "gemma-3-1b-it-nanoquant.gguf"
    if quality["deployment_storage"]["gguf_bytes"] != path.stat().st_size:
        raise ValueError(f"{number}: quality report does not match the GGUF size")
    reader = GGUFReader(path)
    if any(tensor.name == "output.weight" for tensor in reader.tensors):
        raise ValueError(f"{number}: expected tied embeddings")
    (embedding,) = [tensor for tensor in reader.tensors if tensor.name == "token_embd.weight"]
    if embedding.tensor_type.name != "Q8_0":
        raise ValueError(f"{number}: expected a Q8_0 embedding")
    return path, quality, hashlib.sha256(memoryview(embedding.data).cast("B")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attribution", type=Path, help="attribution JSON identifying the common sensitivity teacher")
    parser.add_argument("experiments", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attribution = json.loads(args.attribution.read_text(encoding="utf-8"))
    sweep = Path(attribution["measurements"])
    if hash_file(sweep) != attribution["measurement_sha256"]:
        raise ValueError("the shared KL teacher sweep differs from the attribution")
    experiments = [f"{number:03}" for number in args.experiments]
    runs = {number: load_result(number) for number in experiments}
    first = runs[experiments[0]][1]
    embeddings = {embedding for _, _, embedding in runs.values()}
    if len(embeddings) != 1:
        raise ValueError("the exported embedding bytes differ across experiments")
    # Dataset cache fingerprints can differ for the exact same token ids.
    def benchmark_protocol(quality: dict) -> dict:
        return {key: value for key, value in quality["protocol"].items() if key != "wikitext_fingerprint"}

    for number, (_, quality, _) in runs.items():
        if (
            (quality["model"]["source"], quality["model"]["revision"])
            != (first["model"]["source"], first["model"]["revision"])
            or quality["candidate"]["backend"] != first["candidate"]["backend"]
            or benchmark_protocol(quality) != benchmark_protocol(first)
        ):
            raise ValueError(f"{number}: benchmark model or protocol differs")
    scores = {}
    with sweep_session(sweep, local_files_only=True) as session:
        for number, (path, quality, _) in runs.items():
            print(f"Measuring Experiment {number}", flush=True)
            kl = measure_file(
                session.model, session.adapter, session.dataset, path, session.targets, session.curves, session.clean,
                device=session.protocol["device"], token_chunk_size=session.protocol["token_chunk_size"],
                log=lambda message: print(message, flush=True),
            )
            tasks = {row["task_name"]: row["frozen"] for row in quality["comparison"]["tasks"]}
            scores[number] = {
                "gguf": kl,
                "wikitext_perplexity": quality["comparison"]["wikitext"]["frozen_perplexity"],
                "six_task_accuracy": statistics.mean(value for task, value in tasks.items() if task != "mmlu"),
                "tasks": tasks,
            }
        report = {
            "teacher": session.protocol["teacher"], "sweep_sha256": attribution["measurement_sha256"],
            "clean": session.clean, "embedding_sha256": embeddings.pop(),
            "benchmark_protocol": benchmark_protocol(first),
            "experiments": scores,
        }
    lines = [
        "# Joint KL and benchmark comparison", "",
        f"Full-vocabulary KL above the clean FP16 reference on {report['clean']['token_count']:,} token positions.",
        "Decoder KL swaps all decoder projections together with clean embeddings. Full model KL also swaps the",
        "byte-identical Q8_0 embeddings shared across these exports.", "",
        "| Experiment | Joint decoder KL | Full model KL | WikiText-2 PPL | Six-task mean | MMLU | GGUF bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for number, row in scores.items():
        gguf = row["gguf"]
        decoder, whole = (gguf["variants"][key]["kl_above_control"] for key in ("projections", "all weights"))
        lines.append(f"| {number} | {decoder:.6f} | {whole:.6f} | {row['wikitext_perplexity']:.4f} | "
                     f"{row['six_task_accuracy']:.2%} | {row['tasks']['mmlu']:.2%} | {gguf['bytes']:,} |")
    lines.extend(["", "| Task | " + " | ".join(experiments) + " |",
                  "| --- | " + " | ".join(["---:"] * len(experiments)) + " |"])
    for task in first["protocol"]["task_names"]:
        lines.append(f"| {task} | " + " | ".join(f"{row['tasks'][task]:.2%}" for row in scores.values()) + " |")
    atomic_write_json(args.output.with_suffix(".json"), report)
    atomic_write_text(args.output, "\n".join([*lines, ""]))
    print(f"Report written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
