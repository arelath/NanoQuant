# Glossary

**Activation store**  
A storage abstraction for block input/output tensors. Implementations may use GPU memory, pinned or pageable RAM, or memory-mapped disk.

**ADMM**  
Alternating Direction Method of Multipliers, used by NanoQuant to optimize constrained factor representations.

**Artifact**  
An immutable, versioned, checksummed output of a stage, described by a non-executable metadata document.

**Artifact BPW**  
Complete deployable artifact bytes multiplied by eight and divided by the chosen source parameter count. Includes required non-binary components and container data as defined by the report.

**Backend**  
An implementation of packed NanoQuant operations for a particular runtime/device/workload, such as the PyTorch reference or a CUDA binary kernel backend.

**Block replay**  
Re-execution of quantization/tuning behavior using a captured transformer block, activations, targets, and statistics without loading the full source model.

**Calibration**  
Processing representative data to estimate activation, sensitivity, covariance, or related statistics used by quantization.

**Commit boundary**  
A unit of work after which all required outputs are atomically durable and safe to reuse during resume.

**Content address**  
An identity derived from canonical metadata and content hashes rather than a mutable filename.

**Core BPW**  
Logical bits used by quantized target layers divided by the corresponding original target-weight count. It excludes components according to an explicitly stated accounting contract.

**Decode**  
Autoregressive generation after prefill, usually processing one new token per active sequence per step.

**Domain layer**  
The architecture layer containing mathematical concepts, policies, and typed results independent of CLI, storage, or model frameworks.

**Evidence tier**  
The strength/cost level of a result: unit, replay, smoke, quick, standard, or full.

**Executor**  
The component responsible for running stages under a placement/resource strategy such as resident, CPU-offload, streaming, or distributed execution.

**Factorizer**  
A strategy that transforms a source weight and objective into NanoQuant factors and reconstruction metrics.

**Fork**  
A new run derived from a prior run with changed semantic inputs. Unlike resume, it has a new run identity.

**Frozen logical state**  
Immutable backend-independent NanoQuant parameters after tuning and before backend-specific packing.

**Hessian objective**  
A reconstruction objective that incorporates input-channel covariance rather than only independent diagonal importance.

**Layer plan**  
The immutable planned rank, outliers, objective, retry policy, and bit cost for one source layer.

**Logical seed**  
A deterministic seed derived from stable identifiers such as run seed, stage, block, layer, and attempt, independent of execution history.

**Manifest**  
The authoritative self-description and lineage of a run, including intent, resolved inputs, environment, stage outputs, warnings, and results.

**Model adapter**  
The sole owner of model-family-specific block discovery, tensor mapping, prefix/block/suffix execution, and related behavior.

**Objective artifact**  
A versioned representation of the mathematical reconstruction weighting, such as diagonal, block-diagonal, low-rank-plus-diagonal, or dense Hessian.

**Packed state**  
Immutable deployment tensors arranged in a versioned backend-specific bit layout.

**Prefill**  
Processing prompt tokens to create the initial model outputs and KV cache before autoregressive decode.

**Promotion gate**  
A predefined rule deciding whether evidence justifies a more expensive evaluation or full quantization run.

**Reference backend**  
A clear, slower implementation used as the correctness oracle for packed optimized backends.

**Replay fixture**  
A captured, versioned set of tensors and metadata sufficient to reproduce a layer or block experiment cheaply.

**Resident executor**  
An executor that keeps the complete working model, and normally activations, on GPU.

**Resume**  
Continuation of the same run with identical semantic inputs from its latest valid commit.

**Run**  
One auditable attempt with an immutable resolved recipe, manifest, event stream, artifact references, and terminal status.

**Semantic cache key**  
A stage-specific hash of values that affect that stage's numerical output.

**Stage**  
A typed pipeline operation with declared inputs, outputs, resource estimates, validation, cache identity, and commit semantics.

**Streaming executor**  
An executor that loads source weights by block or layer and may use disk-backed activations so memory use is not proportional to total model size.

**Trainable state**  
Latent factors, scales, outliers, and related parameters used during block or model tuning; it is not a deployment representation.

