"""Plot independent X/Y frequency profiles for NanoQuant-quantized weight blocks."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from weight_image_common import (
    DEFAULT_BF16_MODEL,
    DEFAULT_LLAMA_ROOT,
    DEFAULT_NANOQUANT_MODEL,
    axis_frequency_profiles,
    load_dense_weight,
    load_gguf_module,
    nanoquant_inventory,
    reconstruct_nanoquant_weight,
    select_bases,
    tensor_map,
)

_SERIES_COLORS = {
    "BF16": "#2563EB",
    "NanoQuant": "#F97316",
    "Difference": "#A855F7",
}


@dataclass(frozen=True, slots=True)
class FrequencySeries:
    label: str
    x_frequency: np.ndarray
    x_decibels: np.ndarray
    y_frequency: np.ndarray
    y_decibels: np.ndarray


@dataclass(frozen=True, slots=True)
class BlockFrequencyPlot:
    transformer_layer: int
    weight_block: str
    tensor: str
    series: tuple[FrequencySeries, ...]


def _existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _profile(label: str, weight: np.ndarray) -> FrequencySeries:
    x_frequency, x_decibels, y_frequency, y_decibels = axis_frequency_profiles(weight)
    return FrequencySeries(label, x_frequency, x_decibels, y_frequency, y_decibels)


def _positive_peak_samples(
    frequency: np.ndarray,
    decibels: np.ndarray,
    *,
    maximum_points: int = 900,
) -> list[tuple[float, float]]:
    # Per-row/per-column mean removal makes the DC bin intentionally meaningless.
    keep = frequency > 0.0
    x = frequency[keep]
    y = decibels[keep]
    if x.size <= maximum_points:
        return [(float(x_value), float(y_value)) for x_value, y_value in zip(x, y, strict=True)]
    edges = np.linspace(0, x.size, maximum_points + 1, dtype=np.int64)
    result: list[tuple[float, float]] = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        local = start + int(np.argmax(y[start:stop]))
        result.append((float(x[local]), float(y[local])))
    result.sort(key=lambda item: item[0])
    return result


def _reportlab() -> dict[str, Any]:
    try:
        from reportlab.graphics import renderSVG
        from reportlab.graphics.charts.lineplots import LinePlot
        from reportlab.graphics.shapes import Drawing, Line, String
        from reportlab.lib.colors import HexColor
    except ImportError as error:
        raise RuntimeError(
            "frequency plotting requires ReportLab; install with "
            "`.\\.venv\\Scripts\\python.exe -m pip install -e '.[visualization]'`"
        ) from error
    return {
        "Drawing": Drawing,
        "HexColor": HexColor,
        "Line": Line,
        "LinePlot": LinePlot,
        "String": String,
        "renderSVG": renderSVG,
    }


def _add_panel(
    drawing: Any,
    reportlab: dict[str, Any],
    *,
    profiles: tuple[FrequencySeries, ...],
    direction: str,
    x: float,
    y: float,
    width: float,
    height: float,
    y_minimum: float,
    y_maximum: float,
) -> None:
    line_plot = reportlab["LinePlot"]()
    line_plot.x = x
    line_plot.y = y
    line_plot.width = width
    line_plot.height = height
    if direction == "X":
        line_plot.data = [
            _positive_peak_samples(profile.x_frequency, profile.x_decibels) for profile in profiles
        ]
    else:
        line_plot.data = [
            _positive_peak_samples(profile.y_frequency, profile.y_decibels) for profile in profiles
        ]
    for index, profile in enumerate(profiles):
        line_plot.lines[index].strokeColor = reportlab["HexColor"](_SERIES_COLORS[profile.label])
        line_plot.lines[index].strokeWidth = 1.2
    line_plot.xValueAxis.valueMin = 0.0
    line_plot.xValueAxis.valueMax = 0.5
    line_plot.xValueAxis.valueSteps = (0.0, 0.125, 0.25, 0.375, 0.5)
    line_plot.xValueAxis.labelTextFormat = "%.3f"
    line_plot.yValueAxis.valueMin = y_minimum
    line_plot.yValueAxis.valueMax = y_maximum
    line_plot.yValueAxis.valueSteps = tuple(
        np.linspace(y_minimum, y_maximum, 5, dtype=np.float64).tolist()
    )
    line_plot.yValueAxis.labelTextFormat = "%.1f"
    drawing.add(line_plot)
    drawing.add(
        reportlab["String"](
            x + width / 2.0,
            y + height + 18,
            f"{direction}-axis frequency",
            textAnchor="middle",
            fontName="Helvetica-Bold",
            fontSize=11,
        )
    )
    drawing.add(
        reportlab["String"](
            x + width / 2.0,
            y - 30,
            "cycles per weight position",
            textAnchor="middle",
            fontName="Helvetica",
            fontSize=9,
        )
    )


def _render_layer(
    plots: tuple[BlockFrequencyPlot, ...],
    output: Path,
) -> None:
    if not plots:
        raise ValueError("frequency plot requires at least one weight block")
    reportlab = _reportlab()
    width = 1400.0
    row_height = 230.0
    top = 105.0
    bottom = 55.0
    height = top + bottom + row_height * len(plots)
    drawing = reportlab["Drawing"](width, height)
    layer = plots[0].transformer_layer
    drawing.add(
        reportlab["String"](
            width / 2.0,
            height - 34,
            f"Transformer layer {layer}: axis frequency prominence (dB over median power)",
            textAnchor="middle",
            fontName="Helvetica-Bold",
            fontSize=18,
        )
    )
    legend_x = width / 2.0 - 190.0
    for index, profile in enumerate(plots[0].series):
        item_x = legend_x + index * 150.0
        color = reportlab["HexColor"](_SERIES_COLORS[profile.label])
        drawing.add(reportlab["Line"](item_x, height - 66, item_x + 28, height - 66, strokeColor=color, strokeWidth=2))
        drawing.add(
            reportlab["String"](
                item_x + 35,
                height - 70,
                profile.label,
                fontName="Helvetica",
                fontSize=10,
            )
        )

    margin = 85.0
    center_gap = 80.0
    panel_width = (width - 2.0 * margin - center_gap) / 2.0
    panel_height = 145.0
    for row, plot in enumerate(plots):
        panel_y = height - top - (row + 1) * row_height + 55.0
        all_values = np.concatenate(
            [
                values
                for profile in plot.series
                for values in (profile.x_decibels, profile.y_decibels)
            ]
        )
        finite = all_values[np.isfinite(all_values)]
        y_minimum = max(-12.0, float(math.floor(np.percentile(finite, 1.0))))
        y_maximum = max(3.0, float(math.ceil(np.max(finite))))
        drawing.add(
            reportlab["String"](
                margin,
                panel_y + panel_height + 42,
                f"{plot.weight_block}  ({plot.tensor})",
                fontName="Helvetica-Bold",
                fontSize=12,
            )
        )
        _add_panel(
            drawing,
            reportlab,
            profiles=plot.series,
            direction="X",
            x=margin,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            y_minimum=y_minimum,
            y_maximum=y_maximum,
        )
        _add_panel(
            drawing,
            reportlab,
            profiles=plot.series,
            direction="Y",
            x=margin + panel_width + center_gap,
            y=panel_y,
            width=panel_width,
            height=panel_height,
            y_minimum=y_minimum,
            y_maximum=y_maximum,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    reportlab["renderSVG"].drawToFile(drawing, str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="0", help="transformer layer index or 'all' (default: 0)")
    parser.add_argument("--block", default="all", help="weight block ordinal, name, or 'all' (default: all)")
    parser.add_argument("--source", choices=("all", "bf16", "nanoquant", "difference"), default="all")
    parser.add_argument("--bf16-model", type=Path, default=DEFAULT_BF16_MODEL)
    parser.add_argument("--quantized-model", type=Path, default=DEFAULT_NANOQUANT_MODEL)
    parser.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("weight_frequency_plots"))
    args = parser.parse_args()

    gguf = load_gguf_module(args.llama_root)
    quantized_path = _existing_file(args.quantized_model, "NanoQuant GGUF")
    quantized_reader = gguf.GGUFReader(quantized_path, "r")
    quantized_tensors = tensor_map(quantized_reader)
    selected = select_bases(nanoquant_inventory(quantized_tensors), args.layer, args.block)

    need_bf16 = args.source in {"all", "bf16", "difference"}
    bf16_path: Path | None = None
    bf16_tensors: dict[str, Any] | None = None
    if need_bf16:
        bf16_path = _existing_file(args.bf16_model, "BF16 GGUF")
        bf16_reader = gguf.GGUFReader(bf16_path, "r")
        bf16_tensors = tensor_map(bf16_reader)

    grouped: dict[int, list[BlockFrequencyPlot]] = {}
    for layer, weight_block, base in selected:
        source_weight: np.ndarray | None = None
        quantized_weight: np.ndarray | None = None
        if args.source in {"all", "bf16", "difference"}:
            assert bf16_tensors is not None
            source_weight = load_dense_weight(bf16_tensors, base, gguf)
        if args.source in {"all", "nanoquant", "difference"}:
            quantized_weight = reconstruct_nanoquant_weight(quantized_tensors, base, gguf)
        series: list[FrequencySeries] = []
        if args.source in {"all", "bf16"}:
            assert source_weight is not None
            series.append(_profile("BF16", source_weight))
        if args.source in {"all", "nanoquant"}:
            assert quantized_weight is not None
            series.append(_profile("NanoQuant", quantized_weight))
        if args.source in {"all", "difference"}:
            assert source_weight is not None and quantized_weight is not None
            if source_weight.shape != quantized_weight.shape:
                raise ValueError(f"source and NanoQuant shapes differ for {base}")
            series.append(_profile("Difference", quantized_weight - source_weight))
        grouped.setdefault(layer, []).append(
            BlockFrequencyPlot(layer, weight_block, f"{base}.weight", tuple(series))
        )

    output_dir = args.output_dir.expanduser().resolve()
    outputs: list[str] = []
    for layer, plots in sorted(grouped.items()):
        output = output_dir / f"layer-{layer:02d}-weight-frequency-profiles.svg"
        _render_layer(tuple(plots), output)
        outputs.append(str(output))
    print(
        json.dumps(
            {
                "bf16_model": None if bf16_path is None else str(bf16_path),
                "nanoquant_model": str(quantized_path),
                "layer": args.layer,
                "block": args.block,
                "source": args.source,
                "frequency_units": "cycles_per_weight_position",
                "power_units": "dB_over_axis_median",
                "outputs": outputs,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
