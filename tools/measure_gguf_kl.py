"""Measure the KL of llama.cpp GGUF quantizations against a sensitivity sweep's teacher.

Each GGUF's quantized matrices are dequantized and loaded into the sweep's pinned model at the
sweep's precision, then scored with full-vocabulary KL against the same stored teacher and tokens.
This isolates weight quantization error: activations and arithmetic stay in the sweep's precision,
unlike the llama.cpp runtime. Three variants separate the sources of error:

- projections: only the decoder projections the sweep measured;
- embedding: only the token embedding, which Gemma ties to the output projection;
- all weights: every quantized matrix in the file.

NanoQuant GGUFs are read by rebuilding each projection from its binary-factor sidecars.
Unquantized GGUF norms are checked against the model (Gemma stores ``1 + weight``) and left as is.
The projections variant also reports the sum of each projection's isolated fitted KL at its
measured weight error, so real quantization error can be compared with the additive prediction.

Usage:
    .venv\\Scripts\\python.exe tools\\measure_gguf_kl.py ^
        Datasets\\<dataset>\\measurements\\<sweep-id>\\results.json --gguf A.gguf --gguf B.gguf
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import gguf
import numpy as np
import torch
from fit_kl_sensitivity import fit_dataset
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import snapshot_download
from kl_sensitivity_dataset import score_model, verify_local_weights
from materialize_nanoquant_gguf import dense_nanoquant_projections
from measure_joint_kl_targets import projection_weights, spec_for_protocol
from torch import nn

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text, hash_file
from nanoquant.infrastructure.kl_teacher_dataset import load_kl_teacher_dataset
from nanoquant.infrastructure.model_adapters import adapter_for_config

PROJECTIONS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
NORMS = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
}
_BLOCK = re.compile(r"blk\.(\d+)\.(\w+)\.weight")
VARIANTS = ("projections", "embedding", "all weights")


def read_gguf(path: Path) -> dict[str, dict]:
    """Return every GGUF tensor dequantized to FP32 with its storage type and size.

    NanoQuant files store each projection as binary-factor sidecars; those are rebuilt into the
    dense ``<base>.weight`` matrices they encode, with a stacked QKV group split into q, k and v.
    """
    reader = GGUFReader(path)
    stored = {tensor.name: tensor for tensor in reader.tensors}
    tensors: dict[str, dict] = {}
    consumed: set[str] = set()
    stage_format = reader.fields.get("quantization.nanoquant.stage_format")
    if stage_format is not None:
        if stage_format.contents() != "additive_binary_v1":
            raise ValueError(f"unsupported NanoQuant stage format {stage_format.contents()!r}")
        for name, projection in dense_nanoquant_projections(reader.fields, stored, gguf).items():
            stages = sum(source.endswith("_v") for source in projection.sources)
            tensors[name] = {
                "type": f"NanoQuant {stages}-stage binary",
                "elements": int(projection.weight.size),
                "bytes": projection.storage_bytes,
                "values": torch.from_numpy(np.array(projection.weight, dtype=np.float32, copy=True)),
            }
            consumed.update(projection.sources)
    for tensor in reader.tensors:
        if tensor.name in consumed:
            continue
        if ".nq_" in tensor.name:
            raise ValueError(f"unconsumed NanoQuant sidecar {tensor.name}")
        values = dequantize(tensor.data, tensor.tensor_type)
        tensors[tensor.name] = {
            "type": tensor.tensor_type.name,
            "elements": int(tensor.n_elements),
            "bytes": int(tensor.n_bytes),
            "values": torch.from_numpy(np.array(values, dtype=np.float32, copy=True)),
        }
    return tensors


def map_tensors(tensors: Mapping[str, dict], targets: list[str]) -> tuple[dict[str, dict], dict | None, dict]:
    """Split GGUF tensors into sweep projections, the token embedding and norms keyed by model path."""
    projections, norms = {}, {}
    for name, tensor in tensors.items():
        match = _BLOCK.fullmatch(name)
        if match is None:
            continue
        block, kind = int(match.group(1)), match.group(2)
        if kind in PROJECTIONS:
            projections[f"{block}:{PROJECTIONS[kind]}"] = tensor
        elif kind in NORMS:
            norms[f"{block}:{NORMS[kind]}"] = tensor
    if set(projections) != set(targets):
        missing = sorted(set(targets) - set(projections))[:5]
        raise ValueError(f"GGUF projections do not match the sweep targets; missing {missing}")
    if "output.weight" in tensors:
        raise ValueError("this measurement assumes the output projection is tied to the token embedding")
    return projections, tensors.get("token_embd.weight"), norms


def relative_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    reference = original.detach().float().cpu()
    if tuple(reference.shape) != tuple(quantized.shape):
        raise ValueError(f"GGUF tensor shape {tuple(quantized.shape)} differs from model {tuple(reference.shape)}")
    return float(torch.linalg.vector_norm(quantized - reference) / torch.linalg.vector_norm(reference))


def check_norms(model: nn.Module, adapter, norms: Mapping[str, dict]) -> float:
    """Return the largest |GGUF norm − (1 + model norm)|; Gemma GGUF norms store the added one."""
    layers = adapter.get_decoder_layers(model)
    worst = 0.0
    for key, tensor in norms.items():
        block, path = key.split(":", 1)
        weight = layers[int(block)].get_submodule(path).weight.detach().float().cpu()
        worst = max(worst, float((tensor["values"] - (weight + 1)).abs().max()))
    return worst


def _bits(tensors) -> float:
    tensors = list(tensors)
    return 8 * sum(t["bytes"] for t in tensors) / sum(t["elements"] for t in tensors)


def additive_prediction(curves: Mapping, errors: Mapping[str, float]) -> tuple[float, int]:
    total, outside = 0.0, 0
    for name, error in errors.items():
        curve = curves[name]
        low, high = curve["measured_error_range"]
        outside += not low <= error <= high
        total += curve["coefficient"] * error ** curve["exponent"]
    return total, outside


def measure_file(
    model: nn.Module, adapter, dataset, path: Path, sweep_targets: list[str], curves: Mapping, clean: Mapping,
    *, device: str, token_chunk_size: int, log=print,
) -> dict:
    started = time.perf_counter()
    tensors = read_gguf(path)
    projections, embedding, norms = map_tensors(tensors, sweep_targets)
    weights = projection_weights(model, adapter, sweep_targets)
    embed = model.get_input_embeddings().weight
    errors = {name: relative_error(weights[name], tensor["values"]) for name, tensor in projections.items()}
    embedding_error = None if embedding is None else relative_error(embed, embedding["values"])
    swaps = {
        "projections": {name: (weights[name], projections[name]["values"]) for name in projections},
        "embedding": {} if embedding is None else {"token_embd": (embed, embedding["values"])},
    }
    swaps["all weights"] = {**swaps["projections"], **swaps["embedding"]}
    variants = {}
    for variant in VARIANTS:
        pairs = swaps[variant]
        originals = {name: weight.detach().cpu().clone() for name, (weight, _) in pairs.items()}
        try:
            with torch.no_grad():
                for weight, values in pairs.values():
                    weight.copy_(values.to(device=weight.device, dtype=weight.dtype))
            score = score_model(model, dataset, device=device, token_chunk_size=token_chunk_size)
        finally:
            with torch.no_grad():
                for name, (weight, _) in pairs.items():
                    weight.copy_(originals[name].to(weight.device))
        nll_change = score["nll_nats_per_token"] - clean["nll_nats_per_token"]
        variants[variant] = {
            "kl_nats_per_token": score["kl_nats_per_token"],
            "kl_above_control": score["kl_nats_per_token"] - clean["kl_nats_per_token"],
            "nll_nats_per_token": score["nll_nats_per_token"],
            "nll_above_control": nll_change,
            "perplexity_ratio": math.exp(nll_change),
            "sequences": score["sequences"],
        }
        log(f"{path.name} {variant}: KL above control {variants[variant]['kl_above_control']:.6g} nats/token, "
            f"NLL change {nll_change:+.4f}")
    predicted, outside = additive_prediction(curves, errors)
    values = sorted(errors.values())
    return {
        "file": path.name,
        "sha256": hash_file(path),
        "bytes": path.stat().st_size,
        "projection_types": dict(Counter(tensor["type"] for tensor in projections.values())),
        "projection_bits_per_weight": _bits(projections.values()),
        "embedding_type": None if embedding is None else embedding["type"],
        "embedding_bits_per_weight": None if embedding is None else _bits([embedding]),
        "all_quantized_bits_per_weight": _bits(
            [*projections.values(), *([] if embedding is None else [embedding])],
        ),
        "projection_errors": errors,
        "median_projection_error": statistics.median(values),
        "projection_error_range": [values[0], values[-1]],
        "embedding_error": embedding_error,
        "norm_max_abs_difference": check_norms(model, adapter, norms),
        "additive_prediction": predicted,
        "projections_outside_measured_range": outside,
        "variants": variants,
        "seconds": time.perf_counter() - started,
    }


def _types(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} ×{count}" for name, count in sorted(counts.items(), key=lambda item: -item[1]))


def render_markdown(report: Mapping) -> str:
    protocol = report["protocol"]
    clean = report["clean"]
    files = report["files"]
    lines = [
        f"# GGUF quantization KL: {protocol['teacher']['source']}",
        "",
        f"Measured on {report['measured_at']} against the teacher of sweep `{report['measurement_sweep']}`.",
        f"Each GGUF's dequantized weights were loaded into the pinned model at "
        f"`{protocol['teacher']['model_dtype']}` and scored on the same {clean['token_count']:,} next-token "
        "positions as the sensitivity sweep. KL is full-vocabulary KL(teacher ‖ quantized) in nats/token,",
        "averaged over next-token positions. This measures weight quantization only; the llama.cpp runtime",
        "also quantizes activations and uses its own kernels, so its outputs can differ further.",
        "",
        f"Clean model: KL {clean['kl_nats_per_token']:.6g} nats/token against the stored teacher, "
        f"NLL {clean['nll_nats_per_token']:.6f} nats/token.",
        "",
        "## KL by file",
        "",
        "All KL and NLL values are in nats/token, measured above the clean model.",
        "",
        "| GGUF | File MB | Projection bits/weight | Embedding | All weights KL (nats/token) | "
        "All weights NLL change (nats/token) | Perplexity change | Projections-only KL (nats/token) | "
        "Embedding-only KL (nats/token) |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in files:
        variants = item["variants"]
        lines.append(
            f"| {item['file']} | {item['bytes'] / 1e6:.1f} | {item['projection_bits_per_weight']:.2f} | "
            f"{item['embedding_type']} ({item['embedding_bits_per_weight']:.2f} bits) | "
            f"{variants['all weights']['kl_above_control']:.4f} | "
            f"{variants['all weights']['nll_above_control']:+.4f} | "
            f"{variants['all weights']['perplexity_ratio'] - 1:+.1%} | "
            f"{variants['projections']['kl_above_control']:.4f} | "
            f"{variants['embedding']['kl_above_control']:.4f} |"
        )
    lines += [
        "",
        "Perplexity change is exp(NLL change in nats/token) − 1 on the calibration windows for the",
        "all-weights variant.",
        "",
    ]
    if any(name.startswith("NanoQuant") for item in files for name in item["projection_types"]):
        lines += [
            "NanoQuant files are scored on the dense weights rebuilt from their stored binary factors",
            "and scales, with every stage summed. Their bits/weight count all stored factor and scale",
            "bytes; a stacked QKV group's bytes are split across q, k and v by row count.",
            "",
        ]
    lines += [
        "## Projection errors against the isolated-sensitivity prediction",
        "",
        "The prediction sums each projection's fitted isolated KL in nats/token at its measured relative",
        "weight error. Fits come from Gaussian-noise trials between "
        f"{report['measured_error_range'][0]:.0%} and {report['measured_error_range'][1]:.0%} error; "
        "projections outside that range are extrapolated.",
        "",
        "| GGUF | Projection types | Median projection error | Error range | "
        "Additive prediction (nats/token) | Measured projections-only KL (nats/token) | Measured / predicted | "
        "Extrapolated projections |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in files:
        measured = item["variants"]["projections"]["kl_above_control"]
        low, high = item["projection_error_range"]
        lines.append(
            f"| {item['file']} | {_types(item['projection_types'])} | {item['median_projection_error']:.2%} | "
            f"{low:.2%}–{high:.2%} | {item['additive_prediction']:.4f} | {measured:.4f} | "
            f"{measured / item['additive_prediction']:.3f} | {item['projections_outside_measured_range']} |"
        )
    joint = report.get("joint_reference")
    if joint:
        lines += [
            "",
            "## Gaussian equal-KL reference",
            "",
            f"From `{joint['source']}`: every projection perturbed at once to its equal-KL target error "
            f"({joint['runs']} noise draws each).",
            "",
            "| KL target per projection (nats/token) | Median target error | Mean joint KL (nats/token) | "
            "95% CI (nats/token) |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for row in joint["rows"]:
            interval = row["interval"]
            lines.append(
                f"| {row['kl_target']:g} | {row['median_error']:.2%} | {row['mean']:.4f} | "
                + ("n/a" if interval is None else f"{interval[0]:.4f} – {interval[1]:.4f}") + " |"
            )
    lines += [
        "",
        "## Checks",
        "",
        *(
            f"- {item['file']}: SHA-256 `{item['sha256']}`; token embedding relative error "
            f"{item['embedding_error']:.2%}; largest GGUF norm difference from 1 + model norm "
            f"{item['norm_max_abs_difference']:.2e}."
            for item in files
        ),
        f"- Full per-projection errors and per-window scores: `{report['json']}`.",
        "",
    ]
    return "\n".join(lines)


def joint_reference(path: Path | None) -> dict | None:
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "kl_target": item["kl_target"],
            "median_error": statistics.median(item["target_errors"].values()),
            "mean": item["mean_kl_above_control"],
            "interval": item.get("kl_confidence_interval_95"),
        }
        for item in report["results"] if item["kl_target"] > 0
    ]
    return {"source": path.name, "runs": len(report["seeds"]), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path, help="a completed projection sweep's measurements/<id>/results.json")
    parser.add_argument("--gguf", type=Path, action="append", required=True, help="GGUF file; repeatable")
    parser.add_argument("--joint-report", type=Path, help="optional joint-kl-targets JSON to show as a reference")
    parser.add_argument("--output", type=Path, help="Markdown report (default gguf-kl.md beside the sweep)")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    measurements = args.measurements.resolve()
    sweep = json.loads(measurements.read_text(encoding="utf-8"))
    protocol = sweep["protocol"]
    if len(sweep["trials"]) != sweep["expected_trials"]:
        raise ValueError("the sensitivity sweep is incomplete")
    spec = spec_for_protocol(protocol)
    fit = fit_dataset(measurements)
    curves = fit["curves"]
    low = min(curve["measured_error_range"][0] for curve in curves.values())
    high = max(curve["measured_error_range"][1] for curve in curves.values())
    output = (args.output or measurements.with_name("gguf-kl.md")).resolve()
    snapshot = Path(snapshot_download(
        repo_id=spec.source, revision=spec.revision, local_files_only=args.local_files_only,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    ))
    verify_local_weights(snapshot, spec)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    if spec.dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = protocol["device"]
    with acquire_device_lease(device):
        dataset = load_kl_teacher_dataset(measurements.parent.parent.parent)
        if dataset.reference.artifact_id != protocol["teacher_artifact"]:
            raise ValueError("the teacher dataset does not match the sweep's teacher artifact")
        model = load_causal_language_model(
            snapshot, torch_dtype=spec.dtype, attention_implementation=adapter.attention_implementation,
            local_files_only=True,
        ).to(device).eval()
        clean = score_model(model, dataset, device=device, token_chunk_size=protocol["token_chunk_size"])
        files = [
            measure_file(
                model, adapter, dataset, path.resolve(), protocol["targets"], curves, clean,
                device=device, token_chunk_size=protocol["token_chunk_size"],
            )
            for path in args.gguf
        ]
    report = {
        "schema_version": 1,
        "measured_at": time.strftime("%Y-%m-%d %H:%M"),
        "measurement_sweep": measurements.parent.name,
        "measurement_sha256": hash_file(measurements),
        "protocol": protocol,
        "measured_error_range": [low, high],
        "clean": clean,
        "files": files,
        "joint_reference": joint_reference(args.joint_report),
        "json": str(output.with_suffix(".json")),
    }
    atomic_write_json(output.with_suffix(".json"), report)
    atomic_write_text(output, render_markdown(report))
    print(f"Report written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
