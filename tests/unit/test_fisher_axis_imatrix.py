import pytest
import torch
from build_fisher_axis_imatrix import derive_axes, marginal_axes, write_axis_gguf

from nanoquant.infrastructure.imatrix import load_imatrix


def test_marginal_product_preserves_rows_and_stacked_columns():
    first = torch.tensor([[1., 2., 9.], [4., 7., 3.]])
    second = torch.tensor([[2., 8., 6.]])
    rows, column = marginal_axes([first, second])
    reconstructed = [r[:, None] * column[None] for r in rows]
    for expected, actual in zip((first, second), reconstructed, strict=True):
        assert torch.allclose(actual.mean(1), expected.mean(1))
    assert torch.allclose(torch.cat(reconstructed).mean(0), torch.cat((first, second)).mean(0))
    assert column.mean() == pytest.approx(1)


def test_separable_fisher_recovered_without_squaring_scale():
    row = torch.tensor([2., 5., 11.])
    column = torch.tensor([3., 7.])
    fisher = row[:, None] * column[None]
    rows, shared = marginal_axes([fisher[:1], fisher[1:]])
    assert torch.allclose(torch.cat(rows)[:, None] * shared[None], fisher)


@pytest.mark.parametrize("value", [torch.zeros(2, 3), torch.full((2, 3), float("nan")), -torch.ones(2, 3)])
def test_reject_invalid_fisher(value):
    with pytest.raises(ValueError, match="finite, nonnegative and nonzero"):
        marginal_axes([value])


def test_qkv_shared_columns_and_exact_gguf_roundtrip(tmp_path):
    fisher = {f"model.layers.0.self_attn.{p}_proj": torch.arange(1, 7).reshape(2, 3).float() * factor
              for p, factor in (("q", 1), ("k", 2), ("v", 7))}
    columns, rows = derive_axes(fisher)
    vectors = list(columns.values())
    assert all(torch.equal(vectors[0], v) for v in vectors[1:])
    path = tmp_path / "imatrix.gguf"
    write_axis_gguf(path, columns, rows, {"source_targets": 512})
    loaded = load_imatrix(path)
    for projection in ("q", "k", "v"):
        key = f"blk.0.attn_{projection}.weight"
        name = f"model.layers.0.self_attn.{projection}_proj"
        assert torch.equal(loaded.entries[key][0], columns[name])
        assert torch.equal(loaded.row_entries[key][0], rows[name])
    with pytest.raises(FileExistsError):
        write_axis_gguf(path, columns, rows, {})
