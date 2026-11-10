# Vendored NanoQuant GGUF converter

`convert_nanoquant_to_gguf.py` is vendored from the NanoQuant llama.cpp work at
`da52148384591f4b0d87d58c12862e30f43014f1`. Its required SHA-256 is
`31514ccfaba68fb9a8e4ae952de4c17a9933ac7540d2c0cf040f11695046d53d`, matching
`nanoquant.runtime.packed.PACKED_REFERENCE_CONVERTER_SHA256`.

The converter deliberately remains next to its upstream attribution license. At runtime, the RunPod bootstrap copies
it into a pinned upstream llama.cpp checkout because it imports llama.cpp's `conversion.py`, `convert_hf_to_gguf.py`,
and `gguf-py`. The upstream checkout also supplies the standard `llama-quantize` executable used only to quantize the
token embedding. NanoQuant-specific llama.cpp inference sources are not required for GGUF creation.

Source: <https://github.com/arelath/llama.cpp/commit/da52148384591f4b0d87d58c12862e30f43014f1>
Upstream base: <https://github.com/ggml-org/llama.cpp/commit/68a521b591edd2f36a456809230d63aa81003dfc>

The local extension adds GGUF NanoQuant v3 additive binary residual stages. A staged unit uses
`.nq_stage0_*` for its first term and `.nq_residual_N_*` for its independent residual terms,
with explicit stage counts. This primary-name change makes older native loaders reject the
missing dense/primary contract instead of silently executing only the first term. The packed
word layout and underlying per-term CPU/CUDA operations remain unchanged. The original v1/v2
converter hash remains accepted when reading retained artifacts; new exports record the extended converter.
