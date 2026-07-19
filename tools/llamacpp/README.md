# Vendored NanoQuant GGUF converter

`convert_nanoquant_to_gguf.py` is vendored from the NanoQuant llama.cpp work at
`da52148384591f4b0d87d58c12862e30f43014f1`. Its required SHA-256 is
`c2e1fd064bbd46f38e9e3c5f739865d198ca75bd0bb9db16f72530d378d11304`, matching
`nanoquant.runtime.packed.PACKED_REFERENCE_CONVERTER_SHA256`.

The converter deliberately remains next to its upstream attribution license. At runtime, the RunPod bootstrap copies
it into a pinned upstream llama.cpp checkout because it imports llama.cpp's `conversion.py`, `convert_hf_to_gguf.py`,
and `gguf-py`. The upstream checkout also supplies the standard `llama-quantize` executable used only to quantize the
token embedding. NanoQuant-specific llama.cpp inference sources are not required for GGUF creation.

Source: <https://github.com/arelath/llama.cpp/commit/da52148384591f4b0d87d58c12862e30f43014f1>
Upstream base: <https://github.com/ggml-org/llama.cpp/commit/68a521b591edd2f36a456809230d63aa81003dfc>
