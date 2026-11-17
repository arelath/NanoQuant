"""Shared Gemma decoder weight-sensitivity dataset workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from fit_kl_sensitivity import fit_dataset, plot_measurements, plot_targets, require_plotting
from huggingface_hub import hf_hub_download, snapshot_download
from sensitivity_teacher_cache import cache_teacher_targets, teacher_batches
from torch import nn
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.application.kl_budget import causal_kl_nll_from_logits
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import BlockId
from nanoquant.infrastructure.device_lease import acquire_device_lease, canonical_device_name
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_task_evaluation import hash_hf_tokenizer_snapshot
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file
from nanoquant.infrastructure.kl_splice import DenseKlSpliceEvaluator, SpliceReconstructionSet
from nanoquant.infrastructure.kl_teacher_cache import commit_active_kl_teacher_cache, load_active_kl_teacher_cache
from nanoquant.infrastructure.kl_teacher_dataset import (
    KlTeacherDataset,
    KlTeacherProtocol,
    load_kl_teacher_dataset,
    publish_kl_teacher_dataset,
    require_compatible_teacher_dataset,
    token_hash,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config, vocabulary_size_from_config
from nanoquant.infrastructure.runs import RunLease
from nanoquant.kl_budget_workflow import _model_config_hash

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-3-1b-it"
MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
MODEL_WEIGHTS_SHA256 = "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6"
CALIBRATION_REPO = "eaddario/imatrix-calibration"
CALIBRATION_REVISION = "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4"
CALIBRATION_FILE = "combined_all_small.parquet"
CALIBRATION_SHA256 = "5c99d3d04bf3ec239b3a2a774cb61b9d1dc868680a50697d0418808d0b8eb091"
DTYPE = torch.float16
_PRECISION_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
DEFAULT_ERROR_LEVELS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.35)


@dataclass(frozen=True)
class ModelSpec:
    source: str
    revision: str
    model_type: str
    blocks: int
    weight_files: tuple[tuple[str, str], ...]
    dtype: torch.dtype = DTYPE

    @property
    def precision(self) -> str:
        return _PRECISION_NAMES[self.dtype]

    @property
    def slug(self) -> str:
        return self.source.rsplit("/", 1)[1]

    @property
    def weights_identity(self) -> str | dict[str, str]:
        # Preserve the existing 1B sweep ID and its resumable checkpoints.
        return self.weight_files[0][1] if len(self.weight_files) == 1 else dict(self.weight_files)


GEMMA_1B = ModelSpec(MODEL, MODEL_REVISION, "gemma3_text", 26, (("model.safetensors", MODEL_WEIGHTS_SHA256),))
GEMMA_4B = ModelSpec(
    "google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767", "gemma3", 34,
    (
        ("model-00001-of-00002.safetensors", "eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f"),
        ("model-00002-of-00002.safetensors", "fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13"),
    ),
)


def verify_local_weights(snapshot: Path, spec: ModelSpec) -> None:
    if len(spec.weight_files) > 1:
        index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
        if set(index["weight_map"].values()) != set(dict(spec.weight_files)):
            raise ValueError(f"local shard index differs from pinned {spec.source}")
    for filename, expected in spec.weight_files:
        if hash_file(snapshot / filename) != expected:
            raise ValueError(f"local model weights do not match pinned {spec.source}: {filename}")


def calibration_windows(
    corpus: list[int], *, bos: int, samples: int, length: int, seed: int,
) -> torch.Tensor:
    """Sample disjoint windows from the tokenized corpus in reproducible order."""
    width = length - 1
    available = len(corpus) // width
    if available < samples:
        raise ValueError(f"calibration corpus has {available} complete windows, {samples} requested")
    selected = sorted(random.Random(seed).sample(range(available), samples))
    return torch.tensor([[bos, *corpus[i * width : (i + 1) * width]] for i in selected], dtype=torch.int64)


def _load_tokens(args: argparse.Namespace, snapshot: Path, data_sha256: str) -> tuple[torch.Tensor, int]:
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("the calibration tokenizer must define BOS and EOS")
    text_path = getattr(args, "calibration_text", None)
    if text_path is not None:
        if hash_file(text_path) != data_sha256:
            raise ValueError("calibration text SHA-256 differs from the requested input")
        text = text_path.read_text(encoding="utf-8")
        corpus = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        return calibration_windows(
            corpus, bos=bos, samples=args.samples, length=args.sequence_length, seed=args.calibration_seed,
        ), bos
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("install the NanoQuant evaluation extra to read calibration Parquet files") from exc

    path = args.calibration_parquet or Path(hf_hub_download(
        repo_id=CALIBRATION_REPO, filename=CALIBRATION_FILE, repo_type="dataset",
        revision=CALIBRATION_REVISION, local_files_only=args.local_files_only,
    ))
    if hash_file(path) != data_sha256:
        raise ValueError("calibration Parquet SHA-256 differs from the requested input")
    corpus: list[int] = []
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(columns=[args.text_column]):
            for text in batch.column(0).to_pylist():
                if isinstance(text, str) and text.strip():
                    if corpus:
                        corpus.append(eos)
                    corpus.extend(tokenizer.encode(text, add_special_tokens=False, verbose=False))
    return calibration_windows(
        corpus, bos=bos, samples=args.samples, length=args.sequence_length, seed=args.calibration_seed,
    ), bos


def _teacher_dataset(
    args: argparse.Namespace, snapshot: Path, model: nn.Module, config: dict[str, object],
    root: Path, data_sha256: str, spec: ModelSpec = GEMMA_1B,
) -> KlTeacherDataset:
    fingerprint = (
        f"text-windows-v1:{data_sha256}:{args.calibration_seed}"
        if getattr(args, "calibration_text", None) is not None
        else f"parquet-windows-v1:{data_sha256}:{args.text_column}:{args.calibration_seed}"
    )
    compatible = dict(
        source=spec.source, revision=spec.revision, model_hash=_model_config_hash(config),
        tokenizer_hash=hash_hf_tokenizer_snapshot(snapshot), model_dtype=spec.dtype,
        attention_implementation=adapter_for_config(config).attention_implementation,
        device=args.device, batch_size=1, samples=args.samples, sequence_length=args.sequence_length,
        vocabulary_size=vocabulary_size_from_config(config),
    )
    with RunLease(root / ".active-lease.json"):
        if (root / "dataset.json").exists():
            dataset = load_kl_teacher_dataset(root)
            require_compatible_teacher_dataset(dataset, **compatible)
            if dataset.protocol.dataset_fingerprint != fingerprint:
                raise ValueError("teacher dataset uses a different calibration input or window protocol")
            print(f"Reusing teacher targets in {root}", flush=True)
            return dataset

        tokens, bos = _load_tokens(args, snapshot, data_sha256)
        protocol = KlTeacherProtocol(
            **{**compatible, "model_dtype": str(spec.dtype)},
            dataset_fingerprint=fingerprint, bos_token_id=bos, token_hash=token_hash(tokens),
        )
        cache = load_active_kl_teacher_cache(root, protocol.cache_key)
        if cache is None:
            expected_gib = tokens.numel() * protocol.vocabulary_size * 2 / 2**30
            print(f"Capturing {expected_gib:.2f} GiB of teacher log probabilities in {root}", flush=True)
            evaluator = DenseKlSpliceEvaluator(
                model, SpliceReconstructionSet((), (), ()), tokens, device=args.device,
            )
            baseline_nll, batches = evaluator.teacher_cache_state()
            cache = commit_active_kl_teacher_cache(root, protocol.cache_key, baseline_nll, batches)
        return publish_kl_teacher_dataset(root, protocol, tokens, cache)


@contextmanager
def perturb_weights(
    weights: dict[str, torch.Tensor], *, level: float, seed: int,
) -> Iterator[dict[str, dict[str, float | int]]]:
    """Install Gaussian noise scaled per projection and restore the exact original weights."""
    with torch.no_grad():
        originals = {name: weight.detach().clone() for name, weight in weights.items()}
        metrics: dict[str, dict[str, float | int]] = {}
        try:
            for name, weight in weights.items():
                original = originals[name].float()
                original_norm = torch.linalg.vector_norm(original)
                # Stable across target order and noise levels so each level scales the same draw.
                digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
                generator = torch.Generator(device=weight.device).manual_seed(int.from_bytes(digest[:8], "little"))
                noise = torch.randn(weight.shape, device=weight.device, dtype=torch.float32, generator=generator)
                noise.mul_(level * original_norm / torch.linalg.vector_norm(noise))
                weight.copy_(noise.add_(original))
                error_norm = float(torch.linalg.vector_norm(weight.float() - original))
                if not math.isfinite(error_norm):
                    raise ValueError(f"noise level {level} produced nonfinite weights in {name}")
                reference_norm = float(original_norm)
                metrics[name] = {
                    "parameter_count": weight.numel(),
                    "weight_squared_norm": reference_norm**2,
                    "error_squared_norm": error_norm**2,
                    "relative_frobenius_error": error_norm / reference_norm if reference_norm else 0.0,
                }
            yield metrics
        finally:
            for name, weight in weights.items():
                weight.copy_(originals[name])


@torch.inference_mode()
def score_model(
    model: nn.Module, dataset: KlTeacherDataset, *, device: str, token_chunk_size: int,
) -> dict:
    sequences = []
    for index, teacher in enumerate(teacher_batches(dataset, device)):
        tokens = dataset.tokens[index : index + 1].to(device)
        logits = model(input_ids=tokens, use_cache=False).logits
        # Treat rounded log probabilities as logits to renormalize them in FP32.
        nll, kl, count = causal_kl_nll_from_logits(
            teacher.to(device), logits, tokens, token_chunk_size=token_chunk_size,
        )
        if not math.isfinite(kl) or not math.isfinite(nll):
            raise ValueError(f"model returned nonfinite KL or NLL for calibration window {index}")
        sequences.append({"kl_nats_per_token": kl, "nll_nats_per_token": nll, "token_count": count})
        del logits
    return combine_window_scores(sequences)


def combine_window_scores(sequences: list[dict]) -> dict:
    """Reduce ordered window measurements identically for one or several passes."""
    count = sum(row["token_count"] for row in sequences)
    return {
        "kl_nats_per_token": math.fsum(row["kl_nats_per_token"] * row["token_count"] for row in sequences) / count,
        "nll_nats_per_token": math.fsum(row["nll_nats_per_token"] * row["token_count"] for row in sequences) / count,
        "token_count": count,
        "sequences": sequences,
    }


def write_reports(output: Path, report: dict) -> None:
    identity = report["protocol"]
    teacher = identity["teacher"]
    precision = {str(dtype): name.upper() for dtype, name in _PRECISION_NAMES.items()}[teacher["model_dtype"]]
    baseline = report["clean_baseline"]["kl_nats_per_token"]
    lines = [
        f"# {teacher['source']} weight sensitivity", "",
        f"Measured {len(report['trials'])} of {report['expected_trials']} trials.",
        f"Scored {report['clean_baseline']['token_count']:,} next-token positions per trial.",
        f"Teacher dataset at `{identity['teacher_directory']}`.",
        f"Unchanged {precision} model KL against the stored teacher is {baseline:.6g} nats/token.", "",
        "Gaussian weight noise is scaled to the requested relative Frobenius error per projection.",
        f"Measured weight error includes {precision} rounding. KL is full-vocabulary KL(teacher ‖ perturbed).",
        "KL above the control is the signed difference from the unchanged model.", "",
    ]
    grouped = defaultdict(list)
    for trial in report["trials"]:
        grouped[trial["target"], trial["requested_error"]].append(trial)
    summary = []
    for level in identity["levels"]:
        rows = []
        for target in identity["targets"]:
            trials = grouped[target, level]
            if not trials:
                continue
            kl = [row["kl_nats_per_token"] for row in trials]
            rows.append({
                "target": target,
                "requested_error": level,
                "mean_actual_error": statistics.mean(row["actual_error"] for row in trials),
                "mean_kl_nats_per_token": statistics.mean(kl),
                "kl_above_control": statistics.mean(kl) - baseline,
                "seed_stddev_kl": statistics.stdev(kl) if len(kl) > 1 else None,
                "seeds_completed": len(trials),
            })
        rows.sort(key=lambda row: row["mean_kl_nats_per_token"], reverse=True)
        summary.extend(rows)
        lines.extend([
            f"## Requested weight error {level:.3%}", "",
            "| Target | Actual weight error | Mean KL | KL above control | Seed SD | Seeds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in rows:
            sd = "—" if row["seed_stddev_kl"] is None else f"{row['seed_stddev_kl']:.6g}"
            lines.append(
                f"| {row['target']} | {row['mean_actual_error']:.3%} | {row['mean_kl_nats_per_token']:.6g} "
                f"| {row['kl_above_control']:.6g} | {sd} | {row['seeds_completed']} |"
            )
        lines.append("")
    atomic_write_json(output / "results.json", {**report, "summary": summary})
    atomic_write_text(output / "results.md", "\n".join(lines))


def _without(protocol: dict, *keys: str) -> dict:
    return {key: value for key, value in protocol.items() if key not in keys}


def _differs_only_in_levels(protocol: dict, identity: dict) -> bool:
    return _without(protocol, "teacher_directory", "levels") == _without(identity, "levels")


def _load_sweep_report(path: Path, identity: dict, *, superseded_levels_ok: bool = False) -> dict | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = _without(report["protocol"], "teacher_directory")
    if report["schema_version"] == 1 and protocol != identity and superseded_levels_ok and _differs_only_in_levels(
        protocol, identity,
    ):
        # An exported report of the same sweep at other error levels; its trials stay in its own
        # measurements directory and are imported from there.
        return None
    if report["schema_version"] != 1 or protocol != identity:
        raise ValueError(f"{path} contains a different sweep protocol; choose a different --output")
    report.pop("summary", None)
    report["protocol"] = protocol
    return report


def _import_prior_trials(report: dict, identity: dict, checkpoint: Path) -> None:
    """Copy trials at shared error levels from sweeps that differ from this one only in levels.

    Perturbations depend only on target, level and seed, so a trial measured under an otherwise
    identical protocol is the same measurement. Its clean control must also match exactly.
    """
    levels = set(identity["levels"])
    have = {(row["target"], row["requested_error"], row["seed"]) for row in report["trials"]}
    imported = report.setdefault("imported_trials", [])
    for path in sorted(checkpoint.parent.parent.glob("*/results.json")):
        if path.resolve() == checkpoint.resolve():
            continue
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("schema_version") != 1 or not _differs_only_in_levels(prior["protocol"], identity):
            continue
        baseline = report.get("clean_baseline")
        if baseline is not None and baseline != prior["clean_baseline"]:
            print(f"Skipping {path}: its clean control differs from this sweep's", flush=True)
            continue
        rows = [
            row for row in prior["trials"]
            if row["requested_error"] in levels and (row["target"], row["requested_error"], row["seed"]) not in have
        ]
        if not rows:
            continue
        report["clean_baseline"] = prior["clean_baseline"]
        report["trials"].extend(rows)
        have.update((row["target"], row["requested_error"], row["seed"]) for row in rows)
        imported.append({"sweep_id": path.parent.name, "sha256": hash_file(path), "trials": len(rows)})
        print(f"Imported {len(rows)} measured trials from sweep {path.parent.name}", flush=True)
    if not imported:
        report.pop("imported_trials")


def _sweep(
    args: argparse.Namespace, config: dict[str, object], model: nn.Module, dataset: KlTeacherDataset,
    spec: ModelSpec = GEMMA_1B,
) -> Path:
    adapter = adapter_for_config(config)
    layers = adapter.get_decoder_layers(model)
    indices = args.blocks if args.blocks is not None else range(len(layers))
    targets: dict[str, dict[str, torch.Tensor]] = {}
    for index in indices:
        if not 0 <= index < len(layers):
            raise ValueError(f"block {index} is out of range for the {len(layers)}-block model")
        block = layers[index]
        weights = {
            f"{index}:{layer.path}": block.get_submodule(layer.path).weight
            for layer in adapter.quantizable_layers(block, BlockId(index))
        }
        if args.granularity == "block":
            targets[f"block {index}"] = weights
        else:
            targets.update({name: {name: weight} for name, weight in weights.items()})
    identity = {
        "teacher_artifact": dataset.reference.artifact_id,
        "teacher": to_dict(dataset.protocol),
        "model_weights_sha256": spec.weights_identity,
        "noise": "per-projection isotropic Gaussian with fixed relative Frobenius norm",
        "torch_version": str(torch.__version__),
        "token_chunk_size": args.token_chunk_size,
        "device": args.device,
        "targets": list(targets),
        "levels": args.error_levels,
        "seeds": args.seeds,
    }
    sweep_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint = args.teacher_dataset / "measurements" / sweep_id / "results.json"
    output = args.output or checkpoint.parent / "reports"
    with RunLease(checkpoint.parent / ".measurement-lease.json"):
        exported = _load_sweep_report(output / "results.json", identity, superseded_levels_ok=True)
        report = _load_sweep_report(checkpoint, identity) or exported
        if report is None:
            report = {
                "schema_version": 1,
                "protocol": identity,
                "expected_trials": len(targets) * len(args.error_levels) * len(args.seeds),
                "trials": [],
            }
            _import_prior_trials(report, identity, checkpoint)
            if "clean_baseline" not in report:
                report["clean_baseline"] = score_model(
                    model, dataset, device=args.device, token_chunk_size=args.token_chunk_size,
                )
        if not checkpoint.exists():
            atomic_write_json(checkpoint, report, indent=None, sort_keys=False)
        print(f"Saving sweep measurements to {checkpoint}", flush=True)
        try:
            if getattr(args, "teacher_passes", 1) > 1:
                _run_trial_passes(args, model, dataset, targets, report, checkpoint)
            else:
                _run_trials(args, model, dataset, targets, report, checkpoint)
        finally:
            # Read the committed version so the exported reports reflect durable trials.
            committed = json.loads(checkpoint.read_text(encoding="utf-8"))
            committed["protocol"] = {**identity, "teacher_directory": str(args.teacher_dataset.resolve())}
            write_reports(output, committed)
        if getattr(args, "plots", False):
            _write_plots(checkpoint, committed, output, args.granularity)
    return output


def _write_plots(checkpoint: Path, report: dict, output: Path, granularity: str) -> None:
    if len(report["trials"]) != report["expected_trials"]:
        raise ValueError("cannot plot an incomplete sensitivity sweep")
    plot_measurements(report, output)
    positive_levels = sum(level > 0 for level in report["protocol"]["levels"])
    if granularity == "projection" and positive_levels >= 4:
        fit = fit_dataset(checkpoint)
        atomic_write_json(checkpoint.with_name("kl-response-fits.json"), fit)
        plot_targets(fit, output)
    else:
        print("Equal-KL fit requires projection mode and four positive error levels", flush=True)


def _run_trials(
    args: argparse.Namespace, model: nn.Module, dataset: KlTeacherDataset,
    targets: dict[str, dict[str, torch.Tensor]], report: dict, checkpoint: Path,
    *,
    skip_trials: set[tuple[str, float, int]] | None = None,
) -> None:
    done = {(row["target"], row["requested_error"], row["seed"]) for row in report["trials"]}
    done.update(skip_trials or ())
    for target, weights in targets.items():
        for level in args.error_levels:
            for seed in args.seeds:
                if (target, level, seed) in done:
                    continue
                print(
                    f"Trial {len(report['trials']) + 1}/{report['expected_trials']} {target}, {level:.2%}, seed {seed}",
                    flush=True,
                )
                with perturb_weights(weights, level=level, seed=seed) as metrics:
                    score = score_model(model, dataset, device=args.device, token_chunk_size=args.token_chunk_size)
                norm_sq = math.fsum(row["weight_squared_norm"] for row in metrics.values())
                error_sq = math.fsum(row["error_squared_norm"] for row in metrics.values())
                baseline_kl = report["clean_baseline"]["kl_nats_per_token"]
                report["trials"].append({
                    "target": target, "requested_error": level, "seed": seed,
                    "actual_error": math.sqrt(error_sq / norm_sq) if norm_sq else 0.0,
                    "kl_above_control": score["kl_nats_per_token"] - baseline_kl,
                    "projections": metrics, **score,
                })
                atomic_write_json(checkpoint, report, indent=None, sort_keys=False)


def _run_trial_passes(args, model, dataset, targets, report, checkpoint):
    """Keep each teacher partition resident while measuring every outstanding perturbation."""
    count = len(dataset.cache.batches)
    passes = args.teacher_passes
    if not 1 < passes <= count:
        raise ValueError("teacher pass count must be between two and the window count")
    completed = {(row["target"], row["requested_error"], row["seed"]) for row in report["trials"]}
    if len(completed) == report["expected_trials"]:
        return
    parts = []
    for index in range(passes):
        start, stop = index * count // passes, (index + 1) * count // passes
        pass_identity = {"protocol": report["protocol"], "windows": [start, stop], "passes": passes}
        path = checkpoint.parent / f"passes-{passes}" / f"{index:03d}.json"
        if path.exists():
            partial = json.loads(path.read_text(encoding="utf-8"))
            if partial["pass_identity"] != pass_identity:
                raise ValueError("teacher-pass checkpoint has a different scoring identity")
        else:
            partial = {
                "schema_version": 1, "pass_identity": pass_identity, "protocol": report["protocol"],
                "expected_trials": report["expected_trials"],
                "clean_baseline": combine_window_scores(report["clean_baseline"]["sequences"][start:stop]),
                "trials": [],
            }
        done = {(row["target"], row["requested_error"], row["seed"]) for row in partial["trials"]} | completed
        if len(done) != report["expected_trials"]:
            batches = dataset.cache.batches[start:stop]
            cache_bytes = sum(batch.numel() * batch.element_size() for batch in batches)
            budget = int(args.teacher_gpu_cache_gib * 2**30)
            if args.device.startswith("cuda") and cache_bytes > budget:
                raise ValueError("teacher partition exceeds GPU cache budget; increase the pass count")
            subset = replace(dataset, tokens=dataset.tokens[start:stop],
                             cache=replace(dataset.cache, batches=batches))
            subset = cache_teacher_targets(subset, args.device, budget)
            print(f"Pass {index + 1}/{passes}: windows {start}:{stop}, {cache_bytes / 2**30:.2f} GiB; "
                  f"{report['expected_trials'] - len(done)} remaining trials", flush=True)
            _run_trials(args, model, subset, targets, partial, path, skip_trials=completed)
            del subset
        parts.append(partial)
    per_pass = [
        {(row["target"], row["requested_error"], row["seed"]): row for row in part["trials"]}
        for part in parts
    ]
    for target in targets:
        for level in args.error_levels:
            for seed in args.seeds:
                key = (target, level, seed)
                if key in completed:
                    continue
                segments = [part[key] for part in per_pass]
                metadata = ("target", "requested_error", "seed", "actual_error", "projections")
                if any(any(segment[field] != segments[0][field] for field in metadata) for segment in segments):
                    raise ValueError("teacher partitions measured different weight perturbations")
                sequences = [row for segment in segments for row in segment["sequences"]]
                if len(sequences) != count:
                    raise ValueError("teacher partitions do not cover all windows")
                score = combine_window_scores(sequences)
                report["trials"].append({
                    **{field: segments[0][field] for field in metadata}, **score,
                    "kl_above_control": score["kl_nats_per_token"] - report["clean_baseline"]["kl_nats_per_token"],
                })
    atomic_write_json(checkpoint, report, indent=None, sort_keys=False)


def make_parser(spec: ModelSpec, *, dataset_reports: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Build a weight-sensitivity dataset for pinned {spec.source}.")
    parser.add_argument("--snapshot", type=Path, help=f"local snapshot of {spec.source} at {spec.revision}")
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument("--calibration-parquet", type=Path, help="override the pinned imatrix calibration file")
    corpus.add_argument("--calibration-text", type=Path, help="UTF-8 corpus, retaining native tokenizer special tokens")
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--error-levels", type=float, nargs="+", default=list(DEFAULT_ERROR_LEVELS),
        help=(
            f"relative Frobenius errors (default {' '.join(map(str, DEFAULT_ERROR_LEVELS))}); trials at "
            "shared levels are imported from earlier sweeps of the same dataset"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--blocks", type=int, nargs="+", help="zero-based block indices; defaults to all")
    parser.add_argument(
        "--granularity", choices=("projection", "block"), default="projection",
        help="perturb one projection in one block (default), or all projections in one block",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-chunk-size", type=int, default=32)
    parser.add_argument("--teacher-dataset", type=Path, help="reusable teacher dataset directory")
    parser.add_argument("--teacher-only", action="store_true", help="capture or verify targets and exit")
    parser.add_argument("--teacher-gpu-cache-gib", type=float, default=0.0,
                        help="cache this many GiB of teacher targets on CUDA and prefetch pinned host batches")
    parser.add_argument("--teacher-passes", type=int, default=1,
                        help="measure all perturbations per teacher partition, then combine exact window scores")
    parser.set_defaults(output=None)
    if not dataset_reports:
        parser.add_argument("--output", type=Path, default=ROOT / "Results" / f"{spec.slug}-weight-sensitivity")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=dataset_reports,
                        help="render charts after completing the sweep")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(
    arguments: list[str] | None = None, *, spec: ModelSpec = GEMMA_1B, dataset_reports: bool = False,
) -> int:
    parser = make_parser(spec, dataset_reports=dataset_reports)
    args = parser.parse_args(arguments)
    if args.samples < 1 or args.sequence_length < 2 or args.token_chunk_size < 1:
        parser.error("samples and token chunk size must be positive and sequence length must be at least two")
    if not math.isfinite(args.teacher_gpu_cache_gib) or args.teacher_gpu_cache_gib < 0:
        parser.error("teacher GPU cache size must be finite and nonnegative")
    if args.teacher_passes < 1 or args.teacher_passes > args.samples:
        parser.error("teacher passes must be between one and the number of samples")
    if args.teacher_passes > 1 and args.teacher_gpu_cache_gib <= 0:
        parser.error("multiple teacher passes require a positive GPU cache budget")
    if any(not math.isfinite(level) or level < 0 for level in args.error_levels):
        parser.error("error levels must be finite and nonnegative")
    if args.blocks is not None and any(not 0 <= index < spec.blocks for index in args.blocks):
        parser.error(f"{spec.source} block indices must be between 0 and {spec.blocks - 1}")
    for name in ("error_levels", "seeds", "blocks"):
        values = getattr(args, name)
        if values is not None and len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    args.device = canonical_device_name(args.device)
    args.teacher_dataset = args.teacher_dataset or (
        ROOT / "Datasets" / f"{spec.slug}-{spec.precision}-sensitivity-{args.samples}x{args.sequence_length}"
    )
    calibration_path = args.calibration_text or args.calibration_parquet
    data_sha256 = hash_file(calibration_path) if calibration_path else CALIBRATION_SHA256
    snapshot = args.snapshot or Path(snapshot_download(
        repo_id=spec.source, revision=spec.revision, local_files_only=args.local_files_only,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    ))
    if args.snapshot is not None:
        verify_local_weights(snapshot, spec)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    if config.get("model_type") != spec.model_type or adapter.decoder_block_count_from_config(config) != spec.blocks:
        raise ValueError(f"expected a snapshot of pinned {spec.source}")
    if args.plots and not args.teacher_only:
        # Fail before model loading and the lengthy sweep if the optional dependency is missing.
        require_plotting()
    if spec.dtype == torch.float32:
        # TF32 matmuls would reintroduce a reduced-precision rounding floor into FP32 measurements.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with acquire_device_lease(args.device):
        model = load_causal_language_model(
            snapshot, torch_dtype=spec.dtype, attention_implementation=adapter.attention_implementation,
            local_files_only=True,
        ).to(args.device).eval()
        dataset = _teacher_dataset(args, snapshot, model, config, args.teacher_dataset, data_sha256, spec)
        if not args.teacher_only:
            if args.teacher_passes == 1:
                dataset = cache_teacher_targets(dataset, args.device, int(args.teacher_gpu_cache_gib * 2**30))
            if args.teacher_gpu_cache_gib and args.teacher_passes == 1:
                resident = sum(batch.is_cuda for batch in dataset.cache.batches)
                pinned = sum(batch.is_pinned() for batch in dataset.cache.batches)
                print(f"Teacher targets: {resident} CUDA batches, {pinned} pinned-host batches with prefetch",
                      flush=True)
            if dataset_reports:
                output = _sweep(args, config, model, dataset, spec)
            else:
                with RunLease(args.output / ".active-lease.json"):
                    output = _sweep(args, config, model, dataset, spec)
            print(f"Reports written to {output}", flush=True)
        else:
            print(f"Teacher dataset ready in {args.teacher_dataset}", flush=True)
    return 0
