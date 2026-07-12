"""Experiment-019-style reconstruction and final-block table rendering."""

from __future__ import annotations

from nanoquant.domain.models import BlockResult


def _number(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def render_reconstruction_tables(blocks: tuple[BlockResult, ...]) -> str:
    lines = [
        "## Per-layer objective-weighted reconstruction",
        "",
        "| Block | Layer | Rank | Export weighted normalized error | Raw normalized error | Bits |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for block in blocks:
        for layer in block.layers:
            metrics = layer.final_reconstruction
            lines.append(
                f"| {block.block.index} | `{layer.layer.path}` | {layer.frozen_state.rank} | "
                f"{_number(metrics.export_weighted_normalized_error, 6)} | "
                f"{_number(metrics.raw_normalized_error, 6)} | {layer.actual_bit_cost.total} |"
            )
    lines.extend(
        [
            "",
            "## Final frozen block error before model-level KD",
            "",
            "| Block | Source reference | Block entry pre-quantization | Final frozen pre-KD | "
            "Final − block entry | Relative vs block entry | Final − source reference | Relative vs source |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for block in blocks:
        losses = block.losses
        lines.append(
            f"| {block.block.index} | {_number(losses.source_reference, 6)} | "
            f"{_number(losses.block_entry_pre_quantization, 6)} | {_number(losses.final_frozen_pre_kd, 6)} | "
            f"{_number(losses.final_vs_block_entry.absolute_delta, 6)} | "
            f"{_number(losses.final_vs_block_entry.relative_delta, 4)} | "
            f"{_number(losses.final_vs_source_reference.absolute_delta, 6)} | "
            f"{_number(losses.final_vs_source_reference.relative_delta, 4)} |"
        )
    return "\n".join(lines) + "\n"
