import os
from pathlib import Path

import nanoquant
from nanoquant.infrastructure.environment import load_repository_dotenv


def test_expandable_cuda_segments_is_the_default_without_overriding_explicit_policy(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    nanoquant._configure_cuda_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    nanoquant._configure_cuda_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128,expandable_segments:True"

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    nanoquant._configure_cuda_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_repository_dotenv_overrides_inherited_values(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("HF_TOKEN", "inherited-read-only")
    (tmp_path / ".env").write_text("HF_TOKEN=repository-write-token\n", encoding="utf-8")

    assert load_repository_dotenv(tmp_path)
    assert os.environ["HF_TOKEN"] == "repository-write-token"
