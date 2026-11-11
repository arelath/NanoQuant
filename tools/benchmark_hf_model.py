"""Run NanoQuant's 200-example quality benchmark on one Hugging Face model."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

import _paths  # noqa: F401
import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import ModelInfo
from transformers.models.auto.configuration_auto import AutoConfig

from nanoquant.infrastructure.hf_task_evaluation import TOKENIZER_BEHAVIOR_FILES
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.llamacpp_quality import LlamaCppQualityRequest, execute_llamacpp_quality_evaluation
from nanoquant.quality_evaluation import (
    DEFAULT_QUALITY_TASKS,
    QualityEvaluationRequest,
    describe_quality_protocol,
    execute_quality_evaluation,
    prepare_quality_inputs,
)
from nanoquant.quality_evaluation_workflow import render_quality_evaluation_markdown

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_FILES = (*TOKENIZER_BEHAVIOR_FILES, "config.json", "vocab.*", "merges.txt", "*.model", "*.tiktoken")
SPLIT_GGUF = re.compile(r"(.+)-(\d{5})-of-(\d{5})(\.gguf)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class HubModel:
    repo: str
    revision: str | None
    filename: str | None = None


def _hub_model(value: str, revision: str | None) -> HubModel:
    url = urlsplit(value)
    if url.scheme not in {"https", "http"}:
        return HubModel(value, revision)
    if url.hostname not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError("Use a huggingface.co model URL, model ID, or local path.")
    parts = [unquote(part) for part in url.path.strip("/").split("/")]
    if len(parts) == 2 and all(parts):
        return HubModel("/".join(parts), revision)
    if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"}:
        file = "/".join(parts[4:]) or None
        if (parts[2] == "tree" and file is None) or (file and file.lower().endswith(".gguf")):
            if revision is not None and revision != parts[3]:
                raise ValueError("--revision conflicts with the revision in the Hugging Face URL.")
            return HubModel("/".join(parts[:2]), parts[3], file)
    raise ValueError("Use a Hugging Face repository URL or a GGUF file URL.")


def _model_info(hub: HubModel) -> ModelInfo:
    info = HfApi().model_info(hub.repo, revision=hub.revision)
    if not info.sha:
        raise ValueError(f"Hugging Face returned no commit for {hub.repo}.")
    return info


def _snapshot(hub: HubModel, revision: str, patterns: tuple[str, ...]) -> Path:
    return Path(snapshot_download(repo_id=hub.repo, revision=revision, allow_patterns=list(patterns))).resolve()


def _choose_gguf(files: list[str]) -> str:
    choices = [
        name for name in files
        if not name.rsplit("/", 1)[-1].lower().startswith("mmproj")
        and ((split := SPLIT_GGUF.fullmatch(name)) is None or split[2] == "00001")
    ]
    if not choices:
        raise ValueError("No compatible Transformers checkpoint or GGUF model files were found.")
    if len(choices) == 1:
        return choices[0]
    for index, name in enumerate(choices, start=1):
        print(f"  {index}. {name}")
    try:
        selected = int(input("GGUF file number: ").strip())
    except ValueError as error:
        raise ValueError("Enter a GGUF file number from the list.") from error
    if not 1 <= selected <= len(choices):
        raise ValueError("Enter a GGUF file number from the list.")
    return choices[selected - 1]


def _gguf_parts(filename: str) -> tuple[str, ...]:
    split = SPLIT_GGUF.fullmatch(filename)
    if split is None:
        return (filename,)
    count = int(split[3])
    if not 1 <= int(split[2]) <= count:
        raise ValueError(f"Invalid split GGUF name: {filename}")
    return tuple(f"{split[1]}-{index:05d}-of-{split[3]}{split[4]}" for index in range(1, count + 1))


def _source(value: str, revision: str | None) -> tuple[Path, str, str, ModelInfo | None]:
    local = Path(value).expanduser()
    if not value.startswith(("https://", "http://")) and local.exists():
        path = local.absolute()
        if path.is_file() and path.suffix.lower() == ".gguf":
            local_parts = tuple(path.parent / name for name in _gguf_parts(path.name))
            if not all(part.is_file() for part in local_parts):
                raise FileNotFoundError(f"Missing split GGUF files beside {path}.")
            path = local_parts[0]
        elif not path.is_dir():
            raise ValueError("Local input must be a GGUF file or Transformers model directory.")
        return path, str(path), "local", None

    hub = _hub_model(value, revision)
    info = _model_info(hub)
    commit = info.sha
    assert commit is not None
    files = sorted(file.rfilename for file in info.siblings or ())
    weight_patterns: tuple[str, ...] = ()
    if {"model.safetensors", "model.safetensors.index.json"}.intersection(files):
        weight_patterns = ("model*.safetensors", "model.safetensors.index.json")
    elif {"pytorch_model.bin", "pytorch_model.bin.index.json"}.intersection(files):
        weight_patterns = ("pytorch_model*.bin", "pytorch_model.bin.index.json")
    if hub.filename is None and "config.json" in files and weight_patterns:
        print(f"Resolving Transformers weights for {hub.repo} at {commit}.", flush=True)
        return _snapshot(hub, commit, (*TOKENIZER_FILES, *weight_patterns)), hub.repo, commit, info

    filename = hub.filename or _choose_gguf([file for file in files if file.lower().endswith(".gguf")])
    parts = _gguf_parts(filename)
    if any(part not in files for part in parts):
        raise FileNotFoundError(f"GGUF or a required shard is missing in {hub.repo} at {commit}: {filename}")
    print(f"Resolving {parts[0]} from {hub.repo} at {commit}.", flush=True)
    return _snapshot(hub, commit, parts) / parts[0], hub.repo, commit, info


def _tokenizer(value: str | None, revision: str | None, info: ModelInfo | None) -> tuple[Path, str, str]:
    if value is None and info is not None:
        files = {file.rfilename for file in info.siblings or ()}
        if files.intersection({"tokenizer.json", "tokenizer.model"}) and "config.json" in files:
            print(f"Using the tokenizer published with {info.id}. Override with --tokenizer.")
            hub = HubModel(info.id, revision or info.sha)
            commit = _model_info(hub).sha if revision is not None else info.sha
            assert commit is not None
            return _snapshot(hub, commit, TOKENIZER_FILES), hub.repo, commit
    if value is None and info is not None and info.card_data is not None:
        base = info.card_data.get("base_model")
        if isinstance(base, list) and len(base) == 1:
            base = base[0]
        if isinstance(base, str) and base:
            value = base
            print(f"Using the model card's base-model tokenizer ({value}). Override with --tokenizer.")
    if value is None:
        try:
            value = input("Matching Hugging Face tokenizer model ID or local directory: ").strip().strip("\"'")
        except EOFError as error:
            raise ValueError("GGUF scoring requires a matching Hugging Face tokenizer. Specify --tokenizer.") from error
    if not value:
        raise ValueError("GGUF scoring requires a matching Hugging Face tokenizer. Specify --tokenizer.")
    if not value.startswith(("https://", "http://")) and Path(value).expanduser().is_dir():
        path = Path(value).expanduser().resolve()
        return path, str(path), "local"
    hub = _hub_model(value, revision)
    if hub.filename is not None:
        raise ValueError("--tokenizer must identify a Transformers tokenizer repository or directory.")
    commit = _model_info(hub).sha
    assert commit is not None
    return _snapshot(hub, commit, TOKENIZER_FILES), hub.repo, commit


def _progress(event: str, fields: Mapping[str, object]) -> None:
    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"{event.replace('_', ' ')} {detail}".rstrip(), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="Hugging Face model ID, URL, local Transformers folder or GGUF")
    parser.add_argument("--revision", help="Hugging Face model branch, tag, or commit")
    parser.add_argument("--tokenizer", help="Matching Hugging Face tokenizer repo or folder for a GGUF")
    parser.add_argument("--tokenizer-revision", help="Tokenizer branch, tag, or commit")
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), help="Scoring device (default: auto)")
    parser.add_argument(
        "--output", type=Path,
        help="JSON path with a Markdown companion (default: outputs/hf-quality/<time>/quality.json)",
    )
    parser.add_argument(
        "--llama-cpp-root", type=Path,
        default=Path(os.environ.get("NANOQUANT_LLAMA_CPP_ROOT", ROOT.parent / "llama.cpp")),
    )
    parser.add_argument("--llama-cpp-runner", type=Path, help="Override the built NanoQuant GGUF quality runner")
    parser.add_argument("--gpu-layers", type=int, help="GGUF layers to offload (default: all on CUDA, zero on CPU)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (default: backend default)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        model = args.model or input("Hugging Face model ID, URL, or local path: ").strip().strip("\"'")
        if not model:
            raise ValueError("Enter a model ID, URL, or local path.")
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = args.device or default_device
        if args.model is None and args.device is None and sys.stdin.isatty():
            device = input(f"Device [cpu/cuda, default {default_device}]: ").strip().lower() or default_device
        if device not in {"cpu", "cuda", "cuda:0"} or (device != "cpu" and not torch.cuda.is_available()):
            raise ValueError("Choose cpu or a CUDA device available to the benchmark.")
        if args.threads < 0:
            raise ValueError("--threads must be zero or positive.")

        path, source, revision, info = _source(model, args.revision)
        gguf = path.is_file() and path.suffix.lower() == ".gguf"
        snapshot, tokenizer, tokenizer_revision = (
            _tokenizer(args.tokenizer, args.tokenizer_revision, info) if gguf else (path, source, revision)
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (args.output or ROOT / "outputs" / "hf-quality" / timestamp / "quality.json").resolve()
        if output.suffix.lower() != ".json":
            raise ValueError("--output must be a JSON report path.")
        request = QualityEvaluationRequest(
            snapshot=snapshot,
            source=tokenizer,
            revision=tokenizer_revision,
            run_output=output.parent,
            device=device,
            wikitext_samples=64,
            wikitext_sequence_length=128,
            task_names=DEFAULT_QUALITY_TASKS,
            task_limit=200,
        )
        print(f"Scoring 200 examples per task and 64 WikiText-2 windows on {device}.", flush=True)
        prepared = prepare_quality_inputs(request, progress=_progress)
        maximum_context = None
        if (snapshot / "config.json").is_file():
            config = AutoConfig.from_pretrained(snapshot, local_files_only=True).get_text_config()
            maximum_context = getattr(config, "max_position_embeddings", None)
        if isinstance(maximum_context, int) and maximum_context > 0:
            if maximum_context < request.wikitext_sequence_length:
                raise ValueError("The model context is too short for the 128-token WikiText-2 protocol.")
            prepared = replace(prepared, tasks=tuple(
                replace(task, task=replace(task.task, maximum_length=min(task.task.maximum_length, maximum_context)))
                for task in prepared.tasks
            ))
        if gguf:
            gpu_layers = args.gpu_layers if args.gpu_layers is not None else (0 if device == "cpu" else -1)
            if gpu_layers < -1 or (device == "cpu" and gpu_layers != 0):
                raise ValueError("GPU layers must be -1 (all) or non-negative, and zero on CPU.")
            report = execute_llamacpp_quality_evaluation(
                LlamaCppQualityRequest(
                    gguf=path,
                    output=output,
                    llama_cpp_root=args.llama_cpp_root,
                    runner=args.llama_cpp_runner,
                    device=device,
                    gpu_layers=gpu_layers,
                    threads=args.threads,
                    batch_threads=args.threads,
                    gguf_shards=tuple(path.parent / name for name in _gguf_parts(path.name)[1:]),
                ),
                request,
                prepared,
                None,
                describe_quality_protocol(request, prepared),
            )
        else:
            if args.threads:
                torch.set_num_threads(args.threads)
            report = execute_quality_evaluation(
                request, prepared=prepared, evaluate_candidate=False, progress=_progress,
                attention_implementation="eager",
            )
        report["benchmark_model"] = {
            "source": source,
            "revision": revision,
            "path": str(path),
            "backend": "llama.cpp" if gguf else "transformers",
        }
        atomic_write_json(output, report)
        markdown = output.with_suffix(".md")
        atomic_write_text(markdown, render_quality_evaluation_markdown(report))
        scores = report["results"]["gguf" if gguf else "base"]
        print(f"WikiText-2 perplexity: {scores['wikitext']['perplexity']:.6f}")
        for task in scores["tasks"]:
            print(f"{task['task']['task_name']}: {task['result']['primary_value']:.2%}")
        print(f"JSON report: {output}")
        print(f"Markdown report: {markdown}")
        return 0 if report["passed"] else 1
    except KeyboardInterrupt:
        print("\nBenchmark cancelled.", file=sys.stderr)
        return 130
    except EOFError:
        print("Benchmark failed: supply a model or choose from the available options.", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
