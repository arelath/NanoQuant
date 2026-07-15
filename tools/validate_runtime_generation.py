"""Validate packed NanoQuant generation in the pinned Transformers model shell."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.runtime_export import load_frozen_run_auxiliary
from nanoquant.runtime import (
    CudaPackedBackend,
    GenerationRequest,
    SamplingConfig,
    TransformersGenerationModel,
    WorkloadSpec,
    batch_prompts,
    bind_prepared_linears,
    generate,
    hybrid_cache_factory,
    open_packed_artifact,
    plan_execution_workloads,
    prepare_execution_workloads,
    transformers_decoder_module_paths,
)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _eos_token_ids(value: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and value:
        return tuple(int(item) for item in value)
    raise ValueError("pinned model configuration contains no EOS token ID")


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("packed generation validation requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = _DTYPES[args.input_dtype]
    artifact = open_packed_artifact(args.packed_artifact, verify_hashes=True)
    entries = tuple(entry for block in artifact.manifest.blocks for entry in block.layers)
    states = {entry.spec.name: artifact.load_layer(entry.spec.name) for entry in entries}
    specs = tuple(entry.spec for entry in entries)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_ids = (
        tuple(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for prompt in args.prompt
        )
        if args.chat_template
        else tuple(
            tokenizer.encode(prompt, add_special_tokens=not args.no_special_tokens)
            for prompt in args.prompt
        )
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("pinned tokenizer contains no pad token ID")
    input_ids, attention_mask = batch_prompts(prompt_ids, pad_token_id=pad_token_id)
    batch_size, prompt_width = input_ids.shape
    backend = CudaPackedBackend()
    plans = plan_execution_workloads(
        specs,
        prefill=WorkloadSpec(
            "prefill",
            "cuda",
            args.input_dtype,
            batch_size,
            prompt_width,
            deterministic=True,
        ),
        decode=WorkloadSpec(
            "decode",
            "cuda",
            args.input_dtype,
            batch_size,
            1,
            deterministic=True,
        ),
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )

    started = time.perf_counter()
    auxiliary = (
        None
        if args.run_output is None
        else load_frozen_run_auxiliary(
            args.run_output,
            args.expected_blocks,
            use_global_tuning=True,
            fresh_validation=True,
        )
    )
    with wait_for_device_lease(str(device), args.wait_for_device_seconds):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).eval()
        model.to(device)
        if auxiliary is not None:
            model_parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in auxiliary.parameters:
                    if name not in model_parameters:
                        raise ValueError(f"frozen auxiliary parameter is absent from shell: {name}")
                    target = model_parameters[name]
                    if target.shape != value.shape:
                        raise ValueError(f"frozen auxiliary parameter shape differs: {name}")
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
            del model_parameters
        prepared = prepare_execution_workloads(plans, states, (backend,), device)
        layer_names = tuple(entry.spec.name for entry in entries)
        replaced = bind_prepared_linears(
            model,
            prepared,
            transformers_decoder_module_paths(layer_names),
        )
        del states
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        request = GenerationRequest(
            input_ids.to(device),
            attention_mask.to(device),
            args.max_new_tokens,
            (
                (int(model.config.vocab_size),)
                if args.ignore_eos
                else _eos_token_ids(model.config.eos_token_id)
            ),
            pad_token_id,
            sampling=SamplingConfig(
                mode=args.sampling,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            ),
            stopping_check_interval=args.stopping_check_interval,
        )
        shell = TransformersGenerationModel(
            model,
            hybrid_cache_factory(model.config),
        )
        torch.cuda.synchronize(device)
        generation_started = time.perf_counter()
        first = generate(request, shell)
        torch.cuda.synchronize(device)
        first_seconds = time.perf_counter() - generation_started
        first_allocated = torch.cuda.memory_allocated(device)

        second = generate(request, shell)
        torch.cuda.synchronize(device)
        second_allocated = torch.cuda.memory_allocated(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)

    if not torch.equal(first.token_ids, second.token_ids):
        raise ValueError("deterministic generation token replay differs")
    if first.lengths != second.lengths or first.stop_reasons != second.stop_reasons:
        raise ValueError("deterministic generation stopping replay differs")
    if replaced != len(entries):
        raise ValueError(f"model shell replaced {replaced} linears, expected {len(entries)}")

    generated_ids = [
        first.token_ids[row, :length].detach().cpu().tolist()
        for row, length in enumerate(first.lengths)
    ]
    generated_text = [tokenizer.decode(item) for item in generated_ids]
    reference_prefix: dict[str, Any] | None = None
    if args.reference_output is not None:
        reference = json.loads(args.reference_output.read_text(encoding="utf-8"))
        if not isinstance(reference, dict):
            raise ValueError("generation reference output must contain one JSON object")
        expected_prompt = reference.get("prompt")
        expected_text = reference.get("generated")
        expected_count = reference.get("generated_token_count")
        if expected_prompt != args.prompt[0] or not isinstance(expected_text, str):
            raise ValueError("generation reference prompt or text differs from the request")
        if not isinstance(expected_count, int) or expected_count <= 0:
            raise ValueError("generation reference token count is invalid")
        if len(generated_ids) != 1 or len(generated_ids[0]) < expected_count:
            raise ValueError("generation output is shorter than the reference prefix")
        actual_prefix = tokenizer.decode(generated_ids[0][:expected_count])
        if actual_prefix != expected_text:
            raise ValueError(
                f"generation reference prefix differs: {actual_prefix!r} != {expected_text!r}"
            )
        reference_prefix = {
            "path": str(args.reference_output.resolve()),
            "token_count": expected_count,
            "text": expected_text,
            "exact": True,
        }
    return {
        "packed_artifact": str(artifact.root),
        "packed_descriptor_sha256": hash_file(
            artifact.root / "nanoquant-packed-model.json"
        ),
        "model": str(Path(args.model).resolve()),
        "run_output": None if args.run_output is None else str(Path(args.run_output).resolve()),
        "auxiliary_parameter_count": 0 if auxiliary is None else len(auxiliary.parameters),
        "global_tuning_artifact": (
            None
            if auxiliary is None or auxiliary.global_tuning is None
            else auxiliary.global_tuning.artifact_id
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "input_dtype": args.input_dtype,
        "batch_size": batch_size,
        "prompt_width": prompt_width,
        "prompt_lengths": [len(item) for item in prompt_ids],
        "add_special_tokens": not args.no_special_tokens,
        "chat_template": args.chat_template,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "sampling": {
            "mode": args.sampling,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "layer_count": len(entries),
        "replaced_linear_count": replaced,
        "prefill_backend": plans.prefill.layers[0].backend_name,
        "decode_backend": plans.decode.layers[0].backend_name,
        "prefill_fallback_count": plans.prefill.fallback_count,
        "decode_fallback_count": plans.decode.fallback_count,
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "reference_prefix": reference_prefix,
        "stop_reasons": list(first.stop_reasons),
        "prefill_forward_count": first.prefill_forward_count,
        "decode_forward_count": first.decode_forward_count,
        "maximum_cache_length": first.maximum_cache_length,
        "stopping_sync_count": first.stopping_sync_count,
        "terminal_sync_count": first.terminal_sync_count,
        "first_generation_seconds": first_seconds,
        "first_allocated_bytes": first_allocated,
        "second_allocated_bytes": second_allocated,
        "peak_allocated_bytes": peak_allocated,
        "deterministic_replay": True,
        "wall_seconds": time.perf_counter() - started,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Use an out-of-vocabulary EOS sentinel to force the configured length.",
    )
    parser.add_argument("--sampling", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stopping-check-interval", type=int, default=1)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt text; repeat for a variable-length batch.",
    )
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    if not args.prompt:
        args.prompt = ["Hello", "Write one short sentence about quantization."]
    if args.max_new_tokens <= 0:
        raise ValueError("generation validation max_new_tokens must be positive")
    if args.chat_template and args.no_special_tokens:
        parser.error("--chat-template and --no-special-tokens are mutually exclusive")
    if args.sampling == "sample" and args.seed is None:
        parser.error("--sampling sample requires --seed")
    if args.sampling == "greedy" and any(
        value is not None for value in (args.top_k, args.top_p, args.seed)
    ):
        parser.error("greedy validation does not accept sampling-only settings")

    result = _validate(args)
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
