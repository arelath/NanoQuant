import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from build_gemma_270m_hybrid_imatrix import raw_tensors, validate_axes, write_hybrid
from gguf import GGUFWriter

from nanoquant.infrastructure.imatrix import load_imatrix


def fixture(path, *, column_sum, column_count, row_sum, row_count):
    writer = GGUFWriter(str(path), "imatrix")
    writer.add_string("general.type", "imatrix")
    for suffix, values in (("in_sum2", column_sum), ("counts", [column_count]),
                           ("out_sum2", row_sum), ("out_counts", [row_count])):
        writer.add_tensor("blk.0.attn_q.weight." + suffix, np.asarray(values, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_hybrid_preserves_independent_source_axes_and_counts(tmp_path):
    columns, rows, hybrid = (tmp_path / name for name in ("columns.gguf", "rows.gguf", "hybrid.gguf"))
    fixture(columns, column_sum=[2, 8, 18], column_count=2, row_sum=[99, 99], row_count=3)
    fixture(rows, column_sum=[100, 100, 100], column_count=10, row_sum=[25, 100], row_count=5)
    write_hybrid(hybrid, columns, rows)
    result = load_imatrix(hybrid)
    key = "blk.0.attn_q.weight"
    assert torch.equal(result.entries[key], torch.tensor([[1., 4., 9.]]))
    assert torch.equal(result.row_entries[key], torch.tensor([[5., 20.]]))
    assert raw_tensors(hybrid)[key + ".counts"].item() == 2
    assert raw_tensors(hybrid)[key + ".out_counts"].item() == 5
    validate_axes(hybrid, columns, rows)
    with pytest.raises(FileExistsError):
        write_hybrid(hybrid, columns, rows)


def test_hybrid_rejects_changed_source_hash(tmp_path):
    source = tmp_path / "source.gguf"
    fixture(source, column_sum=[1, 2, 3], column_count=1, row_sum=[1, 2], row_count=1)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        write_hybrid(tmp_path / "out.gguf", source, source, columns_sha256="0" * 64)


def test_experiment076_changes_only_imatrix_and_numbered_outputs():
    root = Path(__file__).resolve().parents[2] / "experiments"
    parent = runpy.run_path(str(next(root.glob("075-*.py"))))
    candidate = runpy.run_path(str(next(root.glob("076-*.py"))))
    old, new = parent["CONFIG"], candidate["CONFIG"]
    assert new == replace(old, calibration=replace(old.calibration, imatrix=new.calibration.imatrix))
    assert new.calibration.imatrix.use_rows is True
    assert "unsloth-columns-eaddario-rows" in new.calibration.imatrix.path
    assert parent["PROFILE_OPTIONS"] == candidate["PROFILE_OPTIONS"]
    old_workflow, new_workflow = parent["EXPERIMENT"].workflow, candidate["EXPERIMENT"].workflow
    assert new_workflow.export == replace(
        old_workflow.export,
        logical_output=new_workflow.export.logical_output, packed_output=new_workflow.export.packed_output,
        checkpoint_output=new_workflow.export.checkpoint_output, gguf_output=new_workflow.export.gguf_output,
    )
    assert new_workflow == replace(
        old_workflow, export=new_workflow.export, summary_output=new_workflow.summary_output,
        quality_output=new_workflow.quality_output, quality_markdown_output=new_workflow.quality_markdown_output,
    )
    assert "076" in str(new_workflow.export.gguf_output)
