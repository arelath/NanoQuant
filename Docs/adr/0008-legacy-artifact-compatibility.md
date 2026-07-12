# ADR 0008: Legacy artifact compatibility is conversion-based

Status: accepted

The rewrite's native artifacts are immutable, checksummed, non-executable schemas. Compatibility policy:

- Legacy `.pt` checkpoints are import-only because loading pickle can execute code. Import is disabled by default,
  requires explicit trust, runs through the legacy compatibility environment, and emits a new validated native artifact
  with source hash and lineage.
- The legacy packed PyTorch extension layout is not a stable interchange format. It is accepted only by a version-matched
  conversion tool and is never loaded silently by the deployment runtime.
- Modified llama.cpp GGUF/NanoQuant files are supported through explicit, versioned conversion where the logical factors,
  scales, outliers, padding, and tensor names are semantically compatible. The source llama.cpp revision, converter hash,
  GGUF metadata, and `nanoquant.cu` hash are recorded.
- Unknown schema/layout versions are rejected. Migrations create new artifacts; they never modify the source artifact.

