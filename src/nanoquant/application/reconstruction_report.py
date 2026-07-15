"""Experiment-019-style reconstruction and final-block table rendering."""

from __future__ import annotations

from nanoquant.domain.models import BlockResult, GlobalTuningResult


def _number(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def render_reconstruction_tables(
    blocks: tuple[BlockResult, ...],
    global_tuning: GlobalTuningResult | None = None,
) -> str:
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
    if global_tuning is not None:
        lines.extend(
            [
                "",
                "## Final block error after model-level KD",
                "",
            ]
        )
        if not global_tuning.block_metrics:
            lines.append(
                "This legacy global-tuning artifact predates post-KD block snapshots; "
                "the immutable pre-KD table remains available above."
            )
        else:
            if global_tuning.block_snapshot_protocol_hash is None:
                raise ValueError("global tuning block metrics are missing their snapshot protocol identity")
            if tuple(item.block for item in global_tuning.block_metrics) != tuple(block.block for block in blocks):
                raise ValueError("global tuning block metrics do not align with committed blocks")
            lines.extend(
                [
                    f"Snapshot protocol: `{global_tuning.block_snapshot_protocol_hash}`",
                    "",
                    "| Block | Local final pre-KD | Probe final pre-KD | Probe final post-KD | "
                    "Post-KD − pre-KD | Relative vs pre-KD |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for block, block_metrics in zip(blocks, global_tuning.block_metrics, strict=True):
                comparison = block_metrics.post_kd_vs_pre_kd
                lines.append(
                    f"| {block.block.index} | {_number(block.losses.final_frozen_pre_kd, 6)} | "
                    f"{_number(block_metrics.final_frozen_pre_kd, 6)} | "
                    f"{_number(block_metrics.final_post_kd, 6)} | "
                    f"{_number(comparison.absolute_delta, 6)} | "
                    f"{_number(comparison.relative_delta, 4)} |"
                )
    return "\n".join(lines) + "\n"
