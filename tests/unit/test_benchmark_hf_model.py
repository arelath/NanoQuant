from pathlib import Path

import benchmark_hf_model
import pytest
import torch
from huggingface_hub.hf_api import ModelInfo

from nanoquant.quality_evaluation import PreparedQualityInputs, describe_quality_protocol


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


@pytest.mark.parametrize("gguf", [False, True])
def test_benchmark_writes_markdown_companion_for_each_backend(tmp_path, monkeypatch, gguf):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model_path = tmp_path / "model.gguf" if gguf else snapshot
    if gguf:
        model_path.touch()
    prepared = PreparedQualityInputs(torch.tensor([[1, 2]]), "fixture", 1, 0, "sha256:tokenizer", ())
    monkeypatch.setattr(benchmark_hf_model, "_source", lambda *_: (model_path, "fixture/model", "rev", None))
    monkeypatch.setattr(benchmark_hf_model, "_tokenizer", lambda *_: (snapshot, "fixture/tokenizer", "token-rev"))
    monkeypatch.setattr(benchmark_hf_model, "prepare_quality_inputs", lambda *_, **__: prepared)

    def report(request):
        payload = {
            "passed": True,
            "wall_seconds": 1.5,
            "model": {"source": request.source, "revision": request.revision},
            "protocol": describe_quality_protocol(request, prepared),
            "results": {"gguf" if gguf else "base": {
                "wikitext": {"perplexity": 12.34, "sample_count": 64},
                "tasks": [{"task": {"task_name": "piqa"}, "result": {
                    "primary_metric": "acc_norm", "primary_value": 0.72, "sample_count": 200,
                }}],
                "elapsed_seconds": 1.25,
                "peak_device_bytes": None,
                "peak_host_bytes": 1024,
            }},
            "comparison": None,
        }
        if gguf:
            payload["gguf"] = {"bytes": 1000, "sha256": "model-hash"}
            payload["identity"] = {
                "llama_cpp_commit": "runner-rev", "parallel": 4,
                "gguf_shards": [{"name": "part2.gguf", "sha256": "shard-hash", "bytes": 250}],
            }
        return payload

    monkeypatch.setattr(benchmark_hf_model, "execute_quality_evaluation", lambda request, **_: report(request))
    monkeypatch.setattr(
        benchmark_hf_model, "execute_llamacpp_quality_evaluation", lambda _, request, *a: report(request),
    )
    output = tmp_path / "reports" / "custom.json"

    assert benchmark_hf_model.main(["fixture/model", "--device", "cpu", "--output", str(output)]) == 0
    assert output.is_file()
    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert markdown.startswith("# fixture/model quality benchmark")
    assert "| WikiText-2 | perplexity ↓ | 12.340000 | 64 |" in markdown
    assert "| piqa | acc_norm ↑ | 0.7200 | 200 |" in markdown
    if gguf:
        assert "| GGUF (2 files) | 1,250 |" in markdown
        assert "| llama.cpp | 1.25 | unavailable | 1,024 |" in markdown
        assert "4 sequences in parallel" in markdown
        assert "`runner-rev`" in markdown
    else:
        assert "| transformers | 1.25 | unavailable | 1,024 |" in markdown
        assert "batch 8" in markdown
