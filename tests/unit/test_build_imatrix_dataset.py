import json

import build_imatrix_dataset as builder
import numpy as np
import pytest
from build_imatrix_dataset import extract_calibration, validate_completion, verify_files
from gguf import GGUFWriter

from nanoquant.infrastructure.io_utils import hash_file


@pytest.mark.parametrize("data,special", [(b" ", False), (b"\xef\xbb\xbftext", False),
                                         (b"\xff", False), (b"<bos>hello", False)])
def test_local_text_rejects_invalid_encoding_empty_data_or_unparsed_control_tokens(tmp_path, data, special):
    path = tmp_path / "local.txt"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        builder.read_calibration_text(path, special)


def test_parquet_extraction_preserves_unicode_and_embedded_newlines(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"content": ["hello\r\n世界", None, "", "  ", " last\n"]}), source)
    target = tmp_path / "calibration.txt"
    result = extract_calibration(source, target, "content")
    assert target.read_bytes() == "hello\r\n世界\n\n last\n".encode()
    assert result["sha256"] == hash_file(target)
    assert result["rows_written"] == 2
    assert result["rows_skipped"] == 3


@pytest.mark.parametrize("values,column", [([None, " "], "content"), ([42], "content"), (["text"], "missing")])
def test_bad_calibration_does_not_replace_existing_text(tmp_path, values, column):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"content": values}), source)
    target = tmp_path / "calibration.txt"
    target.write_bytes(b"previous data")
    with pytest.raises(ValueError):
        extract_calibration(source, target, column)
    assert target.read_bytes() == b"previous data"
    assert not target.with_suffix(".txt.partial").exists()


def write_imatrix(path, chunks=3, context=512):
    writer = GGUFWriter(str(path), "imatrix")
    writer.add_string("general.type", "imatrix")
    writer.add_uint32("imatrix.chunk_count", chunks)
    writer.add_uint32("imatrix.chunk_size", context)
    writer.add_tensor("blk.0.attn_q.weight.in_sum2", np.array([3, 6, 9], dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight.counts", np.array([3], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize("completed,context,requested,logged,valid", [
    (3, 512, -1, 3, True), (2, 512, -1, 3, False), (3, 256, -1, 3, False),
    (3, 512, 4, 3, False), (3, 512, 3, 3, True), (3, 512, -1, None, False),
])
def test_completion_rejects_partial_or_short_runs(tmp_path, completed, context, requested, logged, valid):
    path = tmp_path / "imatrix.gguf"
    write_imatrix(path, completed, context)
    log = tmp_path / "imatrix.log"
    log.write_text("" if logged is None else
                   f"compute_imatrix: computing over {logged} chunks, n_ctx=512, batch_size=512")
    if valid:
        result = validate_completion(path, log, 512, requested)
        assert result["chunks"] == 3
        assert result["tensor_count"] == 1
    else:
        with pytest.raises(ValueError):
            validate_completion(path, log, 512, requested)


def test_completed_dataset_hash_audit_detects_tampering(tmp_path):
    path = tmp_path / "imatrix.gguf"
    write_imatrix(path)
    receipt = {"filename": path.name, "sha256": hash_file(path),
               "files": [{"filename": path.name, "sha256": hash_file(path)}]}
    verify_files(tmp_path, json.loads(json.dumps(receipt)))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_files(tmp_path, receipt)


@pytest.mark.parametrize("fail_inference", [False, True])
@pytest.mark.parametrize("local_text", [False, True])
def test_workflow_publishes_only_completed_datasets_and_reuses_them(tmp_path, monkeypatch, fail_inference, local_text):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import huggingface_hub

    parquet = tmp_path / "calibration.parquet"
    pq.write_table(pa.table({"content": ["hello world"]}), parquet)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("# converter fixture")
    executable = llama / "fake-imatrix"
    executable.write_bytes(b"test executable")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(parquet))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(snapshot))
    calls = []

    def fake_run(command, log):
        calls.append(command)
        if "--outfile" in command:
            from pathlib import Path
            Path(command[command.index("--outfile") + 1]).write_bytes(b"converted model")
            log.write_text("conversion completed")
        else:
            from pathlib import Path
            write_imatrix(Path(command[command.index("-o") + 1]))
            log.write_text("compute_imatrix: computing over 3 chunks, n_ctx=512, batch_size=512")
            if fail_inference:
                raise RuntimeError("inference failed")

    monkeypatch.setattr(builder, "run_command", fake_run)
    root = tmp_path / "dataset"
    args = builder.parser().parse_args([
        "--llama-cpp-root", str(llama), "--imatrix-exe", str(executable),
        "--dataset-dir", str(root), "--gpu-layers", "0",
    ])
    if local_text:
        source = tmp_path / "local.txt"
        source.write_bytes(b"<bos><start_of_turn>user\nHello<end_of_turn>\n")
        args.calibration_text = source
        args.parse_special = True

        def no_download(*args, **kwargs):
            raise AssertionError("local text must not download Parquet")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    if fail_inference:
        with pytest.raises(RuntimeError, match="inference failed"):
            builder.build(args)
        assert not (root / "source.json").exists()
        assert not (root / "imatrix.gguf").exists()
        assert (root / "imatrix.partial.gguf").exists()
    else:
        receipt_path = builder.build(args)
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "complete"
        assert receipt["chunks"] == 3
        assert receipt["model"]["revision"] == "snapshot"
        assert not (root / "imatrix.partial.gguf").exists()
        assert builder.build(args) == receipt_path
        assert len(calls) == 2
        if local_text:
            assert "--parse-special" in calls[-1]
            assert (root / "calibration.txt").read_bytes() == source.read_bytes()
            assert receipt["calibration"]["sha256"] == hash_file(source)
            source.write_bytes(b"modified local text")
            with pytest.raises(ValueError, match="settings differ"):
                builder.build(args)
        args.context_size = 1024
        with pytest.raises(ValueError, match="settings differ"):
            builder.build(args)
