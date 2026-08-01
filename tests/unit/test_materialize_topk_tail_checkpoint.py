import os

from tools.materialize_topk_tail_checkpoint import _hardlink_tree


def test_hardlink_tree_reproduces_nested_inventory_without_copying(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (source / "root.json").write_text("root", encoding="utf-8")
    (source / "nested" / "tensor.bin").write_bytes(b"tensor")

    linked = _hardlink_tree(source, destination)

    assert linked == 2
    assert (destination / "root.json").read_text(encoding="utf-8") == "root"
    assert os.stat(source / "root.json").st_ino == os.stat(destination / "root.json").st_ino
    assert os.stat(source / "nested" / "tensor.bin").st_ino == os.stat(
        destination / "nested" / "tensor.bin"
    ).st_ino
