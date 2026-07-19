from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from nanoquant.application.loss_snapshots import BlockLossRecorder
from nanoquant.application.reconstruction_report import render_reconstruction_tables
from nanoquant.domain.models import (
    ActivationStreamRef,
    ArtifactRef,
    BitCost,
    BlockId,
    BlockResult,
    LayerId,
    ReconstructionMetrics,
)
from tests.integration.test_commits_resume import _objects

ROOT = Path(__file__).parents[2]
EXPERIMENT_019 = ROOT / "evidence" / "m0" / "20260712T052926Z"
GOLDEN = ROOT / "tests" / "golden"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_phase_one_block_one() -> BlockResult:
    weight_path = EXPERIMENT_019 / "golden" / "019-phase1-weight-errors.csv"
    rank_path = EXPERIMENT_019 / "golden" / "019-phase1-rank-utility.csv"
    manifest = json.loads((EXPERIMENT_019 / "manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {Path(item["path"]).name: item["sha256"] for item in manifest["golden"]}
    assert _sha256(weight_path) == expected_hashes[weight_path.name]
    assert _sha256(rank_path) == expected_hashes[rank_path.name]

    with weight_path.open(encoding="utf-8", newline="") as stream:
        weights = tuple(row for row in csv.DictReader(stream) if int(row["block"]) == 1)
    with rank_path.open(encoding="utf-8", newline="") as stream:
        ranks = tuple(row for row in csv.DictReader(stream) if int(row["block"]) == 1)
    assert len(weights) == len(ranks) == 7
    rank_by_layer = {row["layer"]: row for row in ranks}

    base_layer, _plan, base_frozen_block, _losses = _objects()
    block = BlockId(0)
    layers = []
    frozen_layers = []
    recorder = BlockLossRecorder(denominator_floor=1e-8)
    recorder.record_source_reference(0.0)
    recorder.record_block_entry(float(ranks[0]["pre_quant_loss"]))
    for row in weights:
        rank_row = rank_by_layer[row["layer"]]
        layer_id = LayerId(block, row["layer"])
        frozen = replace(base_layer.frozen_state, layer=layer_id, rank=int(row["rank"]))
        reconstruction = ReconstructionMetrics(
            row["objective_mode"],
            float(row["target_weighted_norm_sq"]),
            None,
            None,
            float(row["post_unwhiten_weighted_error"]),
            float(row["post_unwhiten_weighted_norm_error"]),
            float(row["export_weighted_error"]),
            float(row["export_weighted_norm_error"]),
            float(row["raw_error"]),
            float(row["norm_error"]),
        )
        bit_cost = BitCost(binary_factor_bits=int(rank_row["binary_bits"]))
        layers.append(
            replace(
                base_layer,
                layer=layer_id,
                frozen_state=frozen,
                final_reconstruction=reconstruction,
                actual_bit_cost=bit_cost,
            )
        )
        frozen_layers.append(frozen)
        recorder.record_after_layer(layer_id, float(rank_row["final_tuned_loss"]))
    final_loss = float(ranks[-1]["final_tuned_loss"])
    recorder.record_final_frozen_pre_kd(final_loss)
    frozen_block = replace(
        base_frozen_block,
        block=block,
        quantized_layers=tuple(frozen_layers),
    )
    artifact = ArtifactRef("fixture", "sha256-" + "0" * 64, 1)
    outputs = ActivationStreamRef(artifact, (1, 1, 1), "float32", 1, 1)
    return BlockResult(
        1,
        block,
        tuple(layers),
        frozen_block,
        recorder.finalize(),
        outputs,
        outputs,
        0,
        0.0,
        0,
        0,
        (),
    )


def test_legacy_phase_one_reconstruction_report_matches_golden() -> None:
    observed = render_reconstruction_tables((_legacy_phase_one_block_one(),))
    expected = (GOLDEN / "legacy-phase1-block1-reconstruction.md").read_text(encoding="utf-8")

    assert observed == expected


def test_near_zero_loss_denominators_match_golden_na_rendering() -> None:
    block = _legacy_phase_one_block_one()
    recorder = BlockLossRecorder(denominator_floor=1e-6)
    recorder.record_source_reference(0.0000005)
    recorder.record_block_entry(0.0000005)
    recorder.record_after_layer(block.layers[0].layer, 1.0)
    recorder.record_final_frozen_pre_kd(1.0)
    observed = render_reconstruction_tables((replace(block, layers=block.layers[:1], losses=recorder.finalize()),))
    expected = (GOLDEN / "near-zero-reconstruction.md").read_text(encoding="utf-8")

    assert observed == expected
