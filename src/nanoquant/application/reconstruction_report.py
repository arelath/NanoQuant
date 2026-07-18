"""Experiment-019-style reconstruction and final-block table rendering."""

from __future__ import annotations

import math

from nanoquant.domain.models import (
    BlockResult,
    GlobalTuningResult,
    LayerResult,
    SharedInputGroupResult,
)


def _number(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2%}"


def render_live_weight_error_report(
    layers: tuple[LayerResult, ...],
    blocks: tuple[BlockResult, ...],
    *,
    groups: tuple[SharedInputGroupResult, ...] = (),
    expected_blocks: int,
    layer_order: tuple[str, ...],
    status: str,
) -> str:
    """Render the incrementally durable reconstruction state for a running compression."""

    if expected_blocks < 0:
        raise ValueError("expected block count must not be negative")
    if not status:
        raise ValueError("live reconstruction status is required")
    block_map = {block.block.index: block for block in blocks}
    layer_map = {(layer.layer.block.index, layer.layer.path): layer for layer in layers}
    group_map = {(group.block.index, group.name): group for group in groups}
    for block in blocks:
        for layer in block.layers:
            layer_map[(block.block.index, layer.layer.path)] = layer
        for group in block.shared_input_groups:
            group_map[(block.block.index, group.name)] = group
    member_metrics = {
        (group.block.index, member.path): metrics
        for group in group_map.values()
        for member, metrics in group.member_reconstruction
    }
    paths = layer_order or tuple(
        sorted({path for _block, path in layer_map} | {path for _block, path in member_metrics})
    )
    durable_unit_count = len(layer_map) + len(group_map)
    group_excess_by_block: dict[int, int] = {}
    for group in group_map.values():
        group_excess_by_block[group.block.index] = group_excess_by_block.get(group.block.index, 0) + (
            len(group.member_reconstruction) - 1
        )
    expected_unit_count = expected_blocks * (len(paths) - max(group_excess_by_block.values(), default=0))
    progress_noun = "physical units" if group_map else "layers"
    lines = [
        "# Live Weight Reconstruction Errors",
        "",
        f"Status: **{status}**",
        "",
        (
            f"Durable progress: **{durable_unit_count}/{expected_unit_count} {progress_noun}**, "
            f"**{len(block_map)}/{expected_blocks} blocks**."
        ),
        "",
        (
            "Cells are final objective-weighted normalized reconstruction error. Rows update after each durable "
            "physical-unit commit; grouped projection cells show their member-specific errors."
        ),
        "",
        "| Block | " + " | ".join(path.rsplit(".", 1)[-1] for path in paths) + " |",
        "| ---: | " + " | ".join("---:" for _ in paths) + " |",
    ]
    for block_index in range(expected_blocks):
        values = []
        for path in paths:
            cell_layer = layer_map.get((block_index, path))
            cell_group_metric = member_metrics.get((block_index, path))
            if cell_layer is not None:
                value = cell_layer.final_reconstruction.export_weighted_normalized_error
            elif cell_group_metric is not None:
                value = cell_group_metric.export_weighted_normalized_error
            else:
                value = None
            values.append("—" if value is None else _number(value))
        lines.append(f"| {block_index + 1} | " + " | ".join(values) + " |")
    actual_bits = sum(layer.actual_bit_cost.total for layer in layer_map.values())
    actual_bits += sum(group.actual_bit_cost.total for group in group_map.values())
    source_parameters = sum(math.prod(layer.plan.source_weight.spec.shape) for layer in layer_map.values())
    source_parameters += sum(group.plan.in_features * group.plan.out_features for group in group_map.values())
    actual_bpw = actual_bits / source_parameters if source_parameters else None
    lines.extend(
        [
            "",
            "## Actual bits per parameter excluding token embeddings",
            "",
            (
                "Actual BPW is the final representation cost of durable quantized linear weights divided by "
                "their original source-weight parameter count. It includes binary factors, scales, outlier "
                "values and indices, and packing padding. Token embeddings, the output head, norms, and "
                "container overhead are excluded."
            ),
            "",
            "| Scope | Durable physical units | Source parameters | Actual bits | Actual BPW |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| Quantized linear weights | {durable_unit_count} | {source_parameters} | "
                f"{actual_bits} | {_number(actual_bpw, 6)} |"
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Durable layer details",
            "",
            "| Block | Layer | State | Rank | Weighted normalized error | Raw normalized error | Bits |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    path_position = {path: index for index, path in enumerate(paths)}
    for (block_index, path), layer in sorted(
        layer_map.items(), key=lambda item: (item[0][0], path_position.get(item[0][1], len(paths)), item[0][1])
    ):
        state = "block final" if block_index in block_map else "layer commit"
        metrics = layer.final_reconstruction
        lines.append(
            f"| {block_index + 1} | `{path}` | {state} | {layer.frozen_state.rank} | "
            f"{_number(metrics.export_weighted_normalized_error, 6)} | "
            f"{_number(metrics.raw_normalized_error, 6)} | {layer.actual_bit_cost.total} |"
        )
    for (block_index, name), group in sorted(group_map.items()):
        state = "block final" if block_index in block_map else "group commit"
        metrics = group.final_reconstruction
        lines.append(
            f"| {block_index + 1} | `{name}` | {state} | {group.frozen_state.rank} | "
            f"{_number(metrics.export_weighted_normalized_error, 6)} | "
            f"{_number(metrics.raw_normalized_error, 6)} | {group.actual_bit_cost.total} |"
        )
    if not layer_map and not group_map:
        lines.append("| — | — | awaiting first durable layer commit | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Completed block error before model-level KD",
            "",
            "| Block | Target weighted power | Entry pre-quantization | Entry normalized | "
            "Final frozen pre-KD | Final normalized | Absolute delta | Relative vs entry |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for block_index, block in sorted(block_map.items()):
        losses = block.losses
        lines.append(
            f"| {block_index + 1} | {_number(losses.target_weighted_mean_square, 6)} | "
            f"{_number(losses.block_entry_pre_quantization, 6)} | "
            f"{_number(losses.block_entry_normalized_error, 6)} | "
            f"{_number(losses.final_frozen_pre_kd, 6)} | "
            f"{_number(losses.final_frozen_normalized_error, 6)} | "
            f"{_number(losses.final_vs_block_entry.absolute_delta, 6)} | "
            f"{_percent(losses.final_vs_block_entry.relative_delta)} |"
        )
    if not block_map:
        lines.append("| — | — | — | — | — | — | — | awaiting first durable block commit |")
    return "\n".join(lines) + "\n"


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
        for group in block.shared_input_groups:
            metrics = group.final_reconstruction
            lines.append(
                f"| {block.block.index} | `{group.name}` | {group.frozen_state.rank} | "
                f"{_number(metrics.export_weighted_normalized_error, 6)} | "
                f"{_number(metrics.raw_normalized_error, 6)} | {group.actual_bit_cost.total} |"
            )
            for member, member_metrics in group.member_reconstruction:
                lines.append(
                    f"| {block.block.index} | `↳ {member.path}` | {group.frozen_state.rank} | "
                    f"{_number(member_metrics.export_weighted_normalized_error, 6)} | "
                    f"{_number(member_metrics.raw_normalized_error, 6)} | — |"
                )
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
            "| Block | Source reference | Target weighted power | Block entry pre-quantization | Entry normalized | "
            "Final frozen pre-KD | Final normalized | "
            "Final − block entry | Relative vs block entry | Final − source reference | Relative vs source |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for block in blocks:
        losses = block.losses
        lines.append(
            f"| {block.block.index} | {_number(losses.source_reference, 6)} | "
            f"{_number(losses.target_weighted_mean_square, 6)} | "
            f"{_number(losses.block_entry_pre_quantization, 6)} | "
            f"{_number(losses.block_entry_normalized_error, 6)} | "
            f"{_number(losses.final_frozen_pre_kd, 6)} | "
            f"{_number(losses.final_frozen_normalized_error, 6)} | "
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
