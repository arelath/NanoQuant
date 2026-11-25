"""Experiment 089 targets a whole-model 1 BPW GGUF for original Gemma 7B IT."""

from recipes import (
    LARGE_MODEL_COMPRESSION_TEMPLATE,
    BaselineRef,
    CompressionExportPolicy,
    ExperimentIdentity,
    define_compression_quality_experiment,
    experiment_main,
)
from recipes._delta import config_delta

MODEL_SOURCE = "google/gemma-7b-it"
MODEL_REVISION = "9c5798d27f588501ce1e108079d2a19e4c3a2353"
TEMPLATE = LARGE_MODEL_COMPRESSION_TEMPLATE

CONFIG = config_delta(
    TEMPLATE,
    model=config_delta(
        TEMPLATE.model,
        source=MODEL_SOURCE,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
    ),
    allocation=config_delta(
        TEMPLATE.allocation,
        # 7,751,073,792 projection weights, 786,432,000 tied embedding weights
        # at 4.5 bits each (Q4_K), and 175,104 norms at 32 bits. Leave about
        # 9.4 MiB above these modeled costs for GGUF metadata and alignment.
        target_bpw=0.634,
        maximum_rank_layer_patterns=(),
        layer_budget_multipliers=(),
        retry=config_delta(TEMPLATE.allocation.retry, enabled=False),
    ),
    outliers=config_delta(TEMPLATE.outliers, charge_to_bit_budget=True),
    factorization=config_delta(
        TEMPLATE.factorization,
        admm=config_delta(TEMPLATE.factorization.admm, transpose_wide=True),
    ),
    block_tuning=config_delta(TEMPLATE.block_tuning, microbatch_size=1),
    runtime=config_delta(TEMPLATE.runtime, block_forward_batch_size=1),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=89,
        name="one-bpw-compress-and-benchmark-gemma-7b-it",
        purpose="Compress original Gemma 7B IT to a GGUF sized near one bit per source parameter.",
        hypothesis="CPU-offloaded compression can fit one RTX 3090 at the target storage rate.",
        baseline=BaselineRef.external(f"{MODEL_SOURCE}@{MODEL_REVISION}"),
        tags=("gemma-7b-it", "1bpw", "cpu-offload", "gguf", "rtx-3090"),
    ),
    CONFIG,
    export=CompressionExportPolicy(runtime_family="gemma", token_embedding_type="q4_k"),
    restore_completed_blocks=False,
    large_model_guards=True,
    quality_backend=None,
    llamacpp_quality=True,
    llamacpp_quality_parallel=1,
)

experiment_main(__name__, __file__, EXPERIMENT)
