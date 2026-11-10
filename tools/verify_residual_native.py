"""Verify residual GGUF loading and summation against a deterministic tiny Gemma model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import torch
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.runtime.artifact import RuntimeModelMetadata, write_logical_artifact
from nanoquant.runtime.backend import QuantizedLinearSpec
from nanoquant.runtime.llamacpp import export_llamacpp_checkpoint
from nanoquant.runtime.logical import LogicalLayerState
from nanoquant.runtime.packed_artifact import convert_logical_to_packed
from nanoquant.runtime.residual import stage_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--llama-build", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    torch.manual_seed(17)
    config = Gemma3TextConfig(
        vocab_size=262144,
        hidden_size=64,
        intermediate_size=384,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=512,
        sliding_window=256,
        query_pre_attn_scalar=32,
    )
    config._attn_implementation = "eager"
    model = Gemma3ForCausalLM(config).to(torch.bfloat16).eval()
    model.save_pretrained(source, safe_serialization=True)
    for filename in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        original = args.tokenizer_snapshot / filename
        if original.is_file():
            shutil.copyfile(original, source / filename)
    spec = QuantizedLinearSpec(
        "blocks.0.mlp.down_proj", "nanoquant-residual-v2", 384, 64, 64, "float32", "bfloat16", residual_ranks=(32,)
    )

    def term(term_spec: QuantizedLinearSpec) -> LogicalLayerState:
        return LogicalLayerState(
            term_spec,
            torch.randint(0, 2, (64, term_spec.rank)).float() * 2 - 1,
            torch.randint(0, 2, (term_spec.rank, 384)).float() * 2 - 1,
            (torch.rand(384) + 0.5).to(torch.bfloat16),
            torch.rand(term_spec.rank).mul(0.005).to(torch.bfloat16),
            (torch.rand(64) + 0.5).to(torch.bfloat16),
        )

    child = term(stage_spec(spec, 0))
    primary = replace(
        term(replace(spec, logical_format="nanoquant-v1", residual_ranks=())),
        spec=spec,
        residual_stages=(child,),
    )
    logical = write_logical_artifact(
        output / "logical", RuntimeModelMetadata("fixture", "pinned", "gemma3", "c", "t"), {0: (primary,)}
    )
    convert_logical_to_packed(logical.root, output / "packed")
    export_llamacpp_checkpoint(output / "packed", output / "checkpoint")

    def module(state: LogicalLayerState) -> FactorizedReferenceLinear:
        return FactorizedReferenceLinear(
            state.left_binary.to(torch.bfloat16),
            state.right_binary.to(torch.bfloat16),
            state.scale_pre,
            state.scale_mid,
            state.scale_post,
            residual_stages=tuple(module(stage) for stage in state.residual_stages),
        )

    model.model.layers[0].mlp.down_proj = module(primary)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids=tokens, use_cache=False).logits[0, -1].float().numpy()
        model.model.layers[0].mlp.down_proj.residual_stages = torch.nn.ModuleList()
        omitted = model(input_ids=tokens, use_cache=False).logits[0, -1].float().numpy()
    expected.tofile(output / "expected-logits.f32")
    omitted.tofile(output / "omitted-residual-logits.f32")
    environment = os.environ.copy()
    environment["PATH"] = str(args.llama_build / "bin" / "Release") + os.pathsep + environment["PATH"]

    def run(name: str, command: list[str]) -> None:
        with (output / f"{name}.stdout.log").open("w") as stdout, (output / f"{name}.stderr.log").open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, env=environment, check=True)

    gguf = output / "tiny-residual.gguf"
    run(
        "convert",
        [
            sys.executable,
            str(args.llama_root / "convert_nanoquant_to_gguf.py"),
            str(source),
            "--nanoquant-checkpoint",
            str(output / "checkpoint"),
            "--outfile",
            str(gguf),
            "--outtype",
            "bf16",
        ],
    )
    probe_source = Path(__file__).parent / "llamacpp" / "residual_probe"
    run(
        "configure-probe",
        [
            "cmake",
            "-S",
            str(probe_source),
            "-B",
            str(output / "probe-build"),
            f"-DLLAMA_ROOT={args.llama_root}",
            f"-DLLAMA_BUILD={args.llama_build}",
        ],
    )
    run("build-probe", ["cmake", "--build", str(output / "probe-build"), "--config", "Release"])
    run(
        "probe",
        [
            str(output / "probe-build" / "Release" / "nanoquant-residual-probe.exe"),
            str(gguf),
            str(output / "logits.f32"),
        ],
    )
    actual = np.fromfile(output / "logits.f32", dtype=np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("native logits are missing or non-finite")
    rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
    omitted_rmse = float(np.sqrt(np.mean((actual - omitted) ** 2)))
    result = {
        "schema_version": 1,
        "fixture": "synthetic-one-block-gemma3",
        "gguf_sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
        "native_library_sha256": hashlib.sha256(
            (args.llama_build / "bin" / "Release" / "llama.dll").read_bytes()
        ).hexdigest(),
        "residual_ranks": [64, 32],
        "native_vs_torch_rmse": rmse,
        "native_vs_omitted_residual_rmse": omitted_rmse,
        "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
        "passed": rmse < 0.02 and rmse < omitted_rmse * 0.25,
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise ValueError("native residual sum does not match the complete Python representation")


if __name__ == "__main__":
    main()
