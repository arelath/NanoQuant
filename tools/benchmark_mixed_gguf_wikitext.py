"""Score mixed GGUFs on a saved quality report's exact WikiText windows.

All candidates must retain the first GGUF's embedding. Dense BF16 replay uses
dequantized GGUF weights; optional native scoring uses the existing llama.cpp
quality runner on identical token IDs. No multiple-choice tasks are run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from benchmark_imatrix_q8_hybrid import request_for
from measure_gguf_kl import PROJECTIONS, map_tensors, read_gguf, swapped
from measure_joint_kl_targets import projection_weights

from nanoquant.application.evaluation import CausalEvaluationRequest, evaluate_causal_nll, model_logits
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_task_evaluation import hash_hf_tokenizer_snapshot
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.llamacpp_quality import LlamaCppQualityRequest, execute_llamacpp_quality_evaluation
from nanoquant.quality_evaluation import PreparedQualityInputs, _prepare_wikitext_windows, describe_quality_protocol


def verify_tokens(tokens: torch.Tensor, mask: torch.Tensor | None, expected: str) -> str:
    digest = hashlib.sha256(tokens.contiguous().numpy().tobytes())
    if mask is not None:
        digest.update(mask.contiguous().numpy().tobytes())
    actual = "sha256:" + digest.hexdigest()
    if actual != expected:
        raise ValueError("WikiText token/mask digest differs from retained protocol")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-cpp-root", type=Path, help="also score each file through native llama.cpp")
    args = parser.parse_args()
    baseline = json.loads(args.quality.read_text(encoding="utf-8"))
    request = request_for(baseline)
    tokens, fingerprint, bos, mask = _prepare_wikitext_windows(
        request.snapshot, samples=request.wikitext_samples, sequence_length=request.wikitext_sequence_length,
        local_files_only=True,
    )
    token_hash = verify_tokens(tokens, mask, baseline["protocol"]["wikitext_token_hash"])
    tokenizer_hash = hash_hf_tokenizer_snapshot(request.snapshot)
    if tokenizer_hash != baseline["protocol"]["tokenizer_hash"]:
        raise ValueError("tokenizer differs from retained protocol")
    prepared = PreparedQualityInputs(tokens, fingerprint, bos, baseline["protocol"]["pad_token_id"], tokenizer_hash,
                                     (), wikitext_attention_mask=mask)
    protocol = describe_quality_protocol(request, prepared)
    protocol["task_names"] = ()
    protocol.pop("mmlu", None)
    identity = {"quality_sha256": hash_file(args.quality), "token_hash": token_hash,
                "files": {str(path.resolve()): hash_file(path) for path in args.gguf},
                "code_sha256": hash_file(Path(__file__))}
    report = {"identity": identity, "scope": "WikiText only; fixed GGUF embedding; dense BF16", "protocol": protocol,
              "files": {}}
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report["identity"] != identity:
            raise ValueError("preserving results with different inputs; select a new output")
    config = json.loads((request.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    targets = [f"{block}:{kind}" for block in range(config["num_hidden_layers"]) for kind in PROJECTIONS.values()]
    with acquire_device_lease(request.device):
        model = load_causal_language_model(request.snapshot, torch_dtype=torch.bfloat16,
                                           attention_implementation=adapter.attention_implementation,
                                           local_files_only=True).to(request.device).eval()
        weights = projection_weights(model, adapter, targets)
        evaluation = CausalEvaluationRequest(tokens.to(request.device),
                                            attention_mask=None if mask is None else mask.to(request.device),
                                            max_length=request.wikitext_sequence_length,
                                            stride=request.wikitext_sequence_length,
                                            batch_size=request.wikitext_batch_size)
        if "bf16" not in report:
            report["bf16"] = to_dict(evaluate_causal_nll(evaluation, model_logits(model)))
            atomic_write_json(args.output, report)
        fixed_embedding = None
        pairs = {}
        for path in args.gguf:
            projections, embedding, _ = map_tensors(read_gguf(path), targets)
            if embedding is None:
                raise ValueError("GGUF is missing the fixed embedding")
            if fixed_embedding is None:
                fixed_embedding = embedding["values"]
            elif not torch.equal(fixed_embedding, embedding["values"]):
                raise ValueError("candidate embedding differs from the control")
            if str(path.resolve()) in report["files"]:
                continue
            pairs = {name: (weights[name], item["values"]) for name, item in projections.items()}
            pairs["embedding"] = (model.get_input_embeddings().weight, fixed_embedding)
            with swapped(pairs):
                result = to_dict(evaluate_causal_nll(evaluation, model_logits(model)))
            report["files"][str(path.resolve())] = result
            atomic_write_json(args.output, report)
            print(f"{path.name}: dense perplexity={result['perplexity']:.6f}", flush=True)
        del weights, model, evaluation, pairs, projections, embedding
        gc.collect()
        torch.cuda.empty_cache()
    if args.llama_cpp_root is not None:
        for path in args.gguf:
            native_path = args.output.parent / (path.stem + ".native-wikitext.json")
            result = execute_llamacpp_quality_evaluation(
                LlamaCppQualityRequest(path, native_path, args.llama_cpp_root, device=request.device),
                request, prepared, None, protocol,
            )
            report.setdefault("native", {})[str(path.resolve())] = result
            atomic_write_json(args.output, report)
            print(f"{path.name}: native evaluation completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
