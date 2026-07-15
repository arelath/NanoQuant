"""Named block-loss snapshots and Experiment-019-compatible comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field

from nanoquant.domain.models import (
    BlockId,
    BlockLossMetrics,
    GlobalTuningBlockMetrics,
    LayerId,
    LossComparison,
)


def compare_losses(
    baseline_name: str, candidate_name: str, baseline: float, candidate: float, denominator_floor: float
) -> LossComparison:
    delta = candidate - baseline
    relative = None if abs(baseline) < denominator_floor else delta / abs(baseline)
    return LossComparison(baseline_name, candidate_name, baseline, candidate, delta, relative, denominator_floor)


def compare_global_tuning_losses(
    block: BlockId,
    final_frozen_pre_kd: float,
    final_post_kd: float,
    denominator_floor: float,
) -> GlobalTuningBlockMetrics:
    return GlobalTuningBlockMetrics(
        block,
        final_frozen_pre_kd,
        final_post_kd,
        compare_losses(
            "final_frozen_pre_kd",
            "final_post_kd",
            final_frozen_pre_kd,
            final_post_kd,
            denominator_floor,
        ),
    )


@dataclass(slots=True)
class BlockLossRecorder:
    denominator_floor: float = 1e-8
    source_reference: float | None = None
    block_entry: float | None = None
    after_layers: list[tuple[LayerId, float]] = field(default_factory=list)
    after_post_block_refit: float | None = None
    final_frozen_pre_kd: float | None = None

    def record_source_reference(self, value: float) -> None:
        self.source_reference = value

    def record_block_entry(self, value: float) -> None:
        self.block_entry = value

    def record_after_layer(self, layer: LayerId, value: float) -> None:
        if any(existing == layer for existing, _ in self.after_layers):
            raise ValueError(f"layer loss already recorded: {layer}")
        self.after_layers.append((layer, value))

    def record_post_block_refit(self, value: float) -> None:
        self.after_post_block_refit = value

    def record_final_frozen_pre_kd(self, value: float) -> None:
        self.final_frozen_pre_kd = value

    def finalize(self) -> BlockLossMetrics:
        if self.source_reference is None or self.block_entry is None or self.final_frozen_pre_kd is None:
            raise ValueError("source-reference, block-entry, and final-frozen-pre-KD losses are required")
        return BlockLossMetrics(
            self.source_reference,
            self.block_entry,
            tuple(self.after_layers),
            self.after_post_block_refit,
            self.final_frozen_pre_kd,
            compare_losses(
                "block_entry_pre_quantization",
                "final_frozen_pre_kd",
                self.block_entry,
                self.final_frozen_pre_kd,
                self.denominator_floor,
            ),
            compare_losses(
                "source_reference",
                "final_frozen_pre_kd",
                self.source_reference,
                self.final_frozen_pre_kd,
                self.denominator_floor,
            ),
        )
