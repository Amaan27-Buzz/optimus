# Architecture

Optimus owns the inference algorithm and runtime; manufacturer drivers/compilers are permitted only as hardware interfaces.

## Boundary diagram

~~~text
model weights + tokenizer (planned)
             |
 validated loader + model graph (planned)
             |
 prefill / decode scheduler + KV cache (planned)
             |
 memory allocator, scratch lifetime planner (planned)
             |
 +-----------+---------------------------+
 | CPU reference kernels | Intel NPU     |
 | available             | planned       |
 +-----------------------+---------------+
            OS APIs           Level Zero + Intel NPU compiler
~~~

Only `optimus-core` and `optimus-cli` currently exist. Add crates only when a subsystem has working code and a stable boundary.

## CPU-first correctness

Use simple operations as correctness oracles: matmul, stable softmax, RMSNorm, signed INT4 unpacking. Extend to RoPE, embeddings, attention, gated MLP and KV-cache generation. Validate shapes and offsets for every user-supplied model. Do not call CPU reference microbenchmarks optimized inference.

## Direct Intel NPU path

Intel NPU Graph Extension / Level Zero runs **compiled graphs**, not free-form Rust kernels. A workable independent backend must discover device capabilities, construct an acceptable intermediate graph, invoke supported Intel compilation, upload buffers, submit commands, synchronize and validate outputs. OpenVINO GenAI is not an engine dependency. If Intel's compiler doesn't accept our graph representation directly, graph lowering is a prerequisite and a major technical risk.

## Memory-first execution

Plan lifetimes and reserve before allocating for weights, KV cache, activations, scratch and staging. Start with logical accounting (implemented); then enforce real limits and reconcile with process private/committed bytes and working set. Do not confuse Windows NPU "shared memory" with separate dedicated VRAM.

SSD storage tiers:
- **Hot:** active weights, buffers and KV pages required by current decode.
- **Warm:** mapped files/pages and prefetched staging buffers.
- **Cold:** weight archives, compiled blobs and optional unused experts.

Continuous SSD streaming for dense layers is unlikely to preserve throughput. Memory-mapped files and "weightless" caches might reduce load-time allocations but do not guarantee runtime memory savings. Benchmark rather than assume.

## Explicit safeguards

- No silent CPU/GPU fallback when NPU-only is requested.
- No reported zero where a counter is unavailable; mark it N/A.
- Version/compiler-specific graph caches must be invalidated when incompatible.
- Prevent shape overflow, malformed model-file offsets and unsafe allocation paths.
- Report actual model, precision, context, hardware and measured resources.
