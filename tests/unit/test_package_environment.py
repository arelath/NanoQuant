import os

import nanoquant


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
