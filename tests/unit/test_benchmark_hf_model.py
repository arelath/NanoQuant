from pathlib import Path

import benchmark_hf_model
from huggingface_hub.hf_api import ModelInfo


def test_gguf_download_url_pins_revision_and_selects_only_the_requested_file(tmp_path, monkeypatch):
    repo = "unsloth/gemma-3-1b-it-GGUF"
    filename = "gemma-3-1b-it-Q2_K.gguf"
    commit = "a" * 40
    info = ModelInfo(
        id=repo,
        sha=commit,
        siblings=[{"rfilename": filename}, {"rfilename": "gemma-3-1b-it-Q4_K_M.gguf"}],
    )

    def model_info(hub):
        assert hub == benchmark_hf_model.HubModel(repo, "main", filename)
        return info

    def download(*, repo_id, revision, allow_patterns):
        assert (repo_id, revision, allow_patterns) == (repo, commit, [filename])
        (tmp_path / filename).write_bytes(b"GGUF")
        return str(tmp_path)

    monkeypatch.setattr(benchmark_hf_model, "_model_info", model_info)
    monkeypatch.setattr(benchmark_hf_model, "snapshot_download", download)

    source = f"https://huggingface.co/{repo}/resolve/main/{filename}?download=true"
    path, model_repo, model_revision, _ = benchmark_hf_model._source(source, None)

    assert path == Path(tmp_path, filename)
    assert model_repo == repo
    assert model_revision == commit
