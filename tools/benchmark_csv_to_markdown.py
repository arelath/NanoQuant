"""Render per-model Markdown and CSV tables from the consolidated benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "non_nanoquant_model_benchmarks.csv"
DEFAULT_OUTPUT = ROOT / "non_nanoquant_model_reports"
QUANTIZATION = re.compile(r"\b(?P<variant>(?:UD-)?(?:IQ|Q)\d+(?:_[A-Z0-9]+)*|BF16|F16|F32)\b", re.IGNORECASE)
UNQUANTIZED = {"Source reference", "BF16", "F16", "F32"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help="Long-form benchmark CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Markdown and CSV output directory.")
    return parser


def _read_json(path: Path, cache: dict[Path, Any]) -> Any:
    if path not in cache:
        try:
            cache[path] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache[path] = None
    return cache[path]


def _quantization(model: str, model_kind: str, source_report: Any) -> str:
    if model_kind == "source model":
        return "Source reference"
    gguf = source_report.get("gguf") if isinstance(source_report, dict) else None
    filename = PureWindowsPath(gguf.get("path", "")).name if isinstance(gguf, dict) else ""
    match = QUANTIZATION.search(filename) or QUANTIZATION.search(model)
    return match.group("variant").upper() if match else model


def _model_family(model: str, model_kind: str, source_report: Any) -> str:
    if model_kind == "source model":
        return model

    if isinstance(source_report, dict):
        model_info = source_report.get("model")
        source = model_info.get("source") if isinstance(model_info, dict) else None
        if isinstance(source, str):
            return source

    quant_match = QUANTIZATION.search(model)
    family = model[: quant_match.start()] if quant_match else model
    family = re.sub(r"[-_ ]*(?:UD|GGUF)[-_ ]*$", "", family, flags=re.IGNORECASE)
    family = family.removesuffix(".gguf").rstrip("-_ .")

    # GGUF filenames commonly flatten the repository owner/model separator.
    if family.startswith("google_gemma-"):
        family = family.replace("google_gemma-", "google/gemma-", 1)
    return family or model


def _quantized_file_bytes(source_report: Any) -> int | None:
    if isinstance(source_report, dict):
        gguf = source_report.get("gguf")
        size = gguf.get("bytes") if isinstance(gguf, dict) else source_report.get("gguf_bytes")
        if isinstance(size, int):
            return size

    return None


def _benchmark_key(row: dict[str, str]) -> tuple[str, ...]:
    category = row.get("benchmark_category", "")
    return (
        category,
        row.get("task", ""),
        row.get("metric", ""),
        row.get("unit", ""),
        row.get("sample_count", ""),
        row.get("task_limit", ""),
        row.get("task_batch_size", ""),
        row.get("wikitext_samples", ""),
        row.get("wikitext_sequence_length", ""),
        row.get("result_slot", "") if category.startswith("generation_runtime") else "",
    )


def _benchmark_label(key: tuple[str, ...]) -> str:
    category, task, metric, unit, sample_count, limit, batch, wiki_samples, sequence, setting = key
    name = task or category.replace("_", " ").title()
    if category == "language_modeling" and "wikitext" in name.lower():
        name = "WikiText-2"
    details = [metric]
    if sample_count:
        details.append(f"samples={sample_count}")
    if limit:
        details.append(f"limit={limit}")
    if wiki_samples:
        details.append(f"WikiText samples={wiki_samples}")
    if sequence:
        details.append(f"sequence={sequence}")
    if batch:
        details.append(f"batch={batch}")
    if setting:
        details.append(f"setting={setting}")
    return f"{name} ({', '.join(details)}) — {unit}" if details else name


def _format_value(value: float, unit: str, run_count: int) -> str:
    if unit == "score_0_to_1":
        formatted = f"{value:.2%}"
    elif abs(value) >= 1000:
        formatted = f"{value:,.2f}"
    else:
        formatted = f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{formatted} (runs={run_count})"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "model"


def _benchmark_table(
    quantizations: list[str], variants: dict[str, list[dict[str, str]]], sizes: dict[str, int | None]
) -> list[list[str]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for quantization, rows in variants.items():
        for row in rows:
            key = _benchmark_key(row)
            try:
                value = float(row["value"])
            except (KeyError, ValueError):
                continue
            grouped[key][quantization].append(value)

    headers = []
    for quant in quantizations:
        size = sizes[quant]
        headers.append(f"{quant} ({size:,} bytes)" if size is not None else quant)
    table = [["Benchmark", *headers]]
    for key in sorted(grouped):
        cells = [_benchmark_label(key)]
        for quantization in quantizations:
            values = grouped[key].get(quantization)
            cells.append(_format_value(statistics.mean(values), key[3], len(values)) if values else "—")
        table.append(cells)
    return table


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def _render_model(
    model: str,
    table: list[list[str]],
    quantizations: list[str],
    variants: dict[str, list[dict[str, str]]],
    sizes: dict[str, int | None],
    model_size: int | None,
) -> str:
    lines = [
        f"# {model}",
        "",
        "Benchmark values are arithmetic means across matching runs. Each cell shows how many runs were averaged.",
        "The variant header includes the recorded file size when available.",
        "",
        f"Largest recorded quantized file: {model_size:,} bytes."
        if model_size is not None
        else "No quantized file size was recorded for this model.",
        "",
        _markdown_row(table[0]),
        _markdown_row(["---", *["---:"] * len(quantizations)]),
        *(_markdown_row(cells) for cells in table[1:]),
    ]

    lines.extend(("", "## Source reports", ""))
    for quantization in quantizations:
        sources = sorted({row.get("source_file", "") for row in variants[quantization] if row.get("source_file")})
        size_note = f"; file size {sizes[quantization]:,} bytes" if sizes.get(quantization) is not None else ""
        lines.append(f"- **{quantization}**{size_note}: " + ", ".join(f"`{source}`" for source in sources))
    return "\n".join((*lines, ""))


def generate_reports(input_csv: Path, output_dir: Path) -> list[tuple[str, int | None, Path]]:
    input_csv = input_csv.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_cache: dict[Path, Any] = {}
    models: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    file_sizes: dict[tuple[str, str], int | None] = {}

    with input_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"model", "model_kind", "benchmark_category", "metric", "value", "unit", "source_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            model = (row.get("model") or "").strip()
            if not model:
                continue
            model_kind = (row.get("model_kind") or "").strip()
            source_path = Path(row["source_file"])
            if not source_path.is_absolute():
                source_path = input_csv.parent / source_path
            source_report = _read_json(source_path, json_cache)
            quantization = _quantization(model, model_kind, source_report)
            family = row.get("model_family") or _model_family(model, model_kind, source_report)
            models[family][quantization].append(row)
            size_key = (family, quantization)
            size = None
            if model_kind != "source model":
                size = int(row["model_bytes"]) if row.get("model_bytes") else _quantized_file_bytes(source_report)
            if size is not None:
                previous = file_sizes.get(size_key)
                file_sizes[size_key] = max(size, previous or 0)
            else:
                file_sizes.setdefault(size_key, None)

    model_sizes: dict[str, int] = {}
    for (family, quant), size in file_sizes.items():
        if quant not in UNQUANTIZED and size is not None:
            model_sizes[family] = max(size, model_sizes.get(family, 0))
    ordered_models = sorted(models, key=lambda family: (-model_sizes.get(family, -1), family.casefold()))
    reports: list[tuple[str, int | None, Path]] = []
    index_lines = [
        "# Non-NanoQuant benchmark reports",
        "",
        "Models are listed by descending largest recorded quantized file size. "
        "Models without a recorded quantized file size are listed last.",
        "",
    ]
    for family in ordered_models:
        variants = models[family]
        sizes = {quant: file_sizes.get((family, quant)) for quant in variants}
        model_size = model_sizes.get(family)
        report_path = output_dir / f"{_slug(family)}.md"
        csv_path = report_path.with_suffix(".csv")
        quantizations = sorted(variants, key=lambda q: (sizes[q] is None, -(sizes[q] or 0), q))
        table = _benchmark_table(quantizations, variants, sizes)
        markdown = _render_model(family, table, quantizations, variants, sizes, model_size)
        report_path.write_text(markdown, encoding="utf-8")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(table)
        reports.append((family, model_size, report_path))
        size_text = f"{model_size:,} bytes" if model_size is not None else "size unavailable"
        index_lines.append(f"- {family} — {size_text} ([Markdown]({report_path.name}) · [CSV]({csv_path.name}))")

    (output_dir / "README.md").write_text("\n".join((*index_lines, "")), encoding="utf-8")
    return reports


def main() -> int:
    args = _parser().parse_args()
    reports = generate_reports(args.input, args.output_dir)
    print(f"Created {len(reports)} Markdown and CSV report pairs in {args.output_dir.resolve()}")
    for model, size, path in reports:
        size_text = f"{size:,} bytes" if size is not None else "size unavailable"
        print(f"{size_text}\t{model}\t{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
