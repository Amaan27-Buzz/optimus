# Optimus development plan

**Objective:** original Rust inference runtime achieving useful model quality and inference speed with substantially reduced RAM footprint on consumer machines. Optimize peak physical memory and committed bytes, then energy per token, cold startup, and throughput. Compare only workloads with identical model, quantization, device and context.

The project has no fixed timeline. Milestones advance when correctness and measurements justify it.

## M0 — Foundation (scaffold)

- [x] Rust workspace, reference CPU tensor kernels and INT4 unpacker.
- [x] Logical memory-budget accounting, unit tests, developer CLI.
- [x] Project principles, architecture, benchmark methodology.
- [ ] Pass Windows and Linux CI including format, Clippy, and tests.
- [ ] Add golden and property tests for edge cases and overflow.

**Exit:** tests reproducibly pass and documentation makes unimplemented features explicit.

## M1 — Our own CPU language-model inference

- [ ] Safetensors loader with validation and indexed access to weight ranges.
- [ ] Tokenizer and chat-template handling for a small Qwen checkpoint.
- [ ] Embeddings, RoPE, attention, RMSNorm, gated MLP, output projection and sampler.
- [ ] Incremental KV cache with correct positions.
- [ ] Logits and tokens verified against independent golden reference outputs.

**Exit:** coherent autoregressive output from our own execution path, no dependency on llama.cpp, Ollama, or OpenVINO GenAI.

## M2 — Low-memory engine

- [ ] Enforced allocation admission, arenas and buffer reuse.
- [ ] OS-backed telemetry for private bytes, commit, working set and system delta.
- [ ] Quantized weights with validated numerical error; precision-aware KV allocation.
- [ ] Controlled context growth and graceful rejection on insufficient memory.
- [ ] SSD/mmap prefetch experiments; separate load-time vs decode-time savings.

**Exit:** reproducible lower peak memory on a matched CPU workload without invalid outputs.

## M3 — Intel NPU proof

- [ ] Direct Level Zero NPU discovery, extension negotiation and capabilities.
- [ ] Compile and submit one independently constructed NPU graph through Intel compiler/driver.
- [ ] Validate output vs our CPU reference and device execution via driver trace.
- [ ] Test constant/dynamic shapes, graph loading, state/KV operations and synchronization.

**Exit:** one real NPU subgraph works; NPU enumeration alone is insufficient.

## M4 — Native NPU token generation

- [ ] Own execution graph representation and lowering to supported NPU input.
- [ ] Memory staging, graph lifetime, command submission and verification.
- [ ] Implement compatible transformer operations and prefill/decode.
- [ ] Unsupported operations fail explicitly rather than silently use other hardware.

**Exit:** generation on Intel NPU using our own inference engine and vendor driver/compiler only.

## M5 — Optimize / benchmark

- [ ] Memory layout and weight-buffer duplication audits.
- [ ] Mixed precision and KV-cache policies with fixed quality thresholds.
- [ ] GPU comparison only as a separately labeled hardware scenario.
- [ ] SSD-backed active weight/expert tiering when access patterns justify it.
- [ ] Matched tests against Ollama and OpenVINO with raw samples, spread and power measurements.

**Exit:** independently reproducible advantage for a specified model/device workload.

## M6 — Generalize and productize

- [ ] Support and verify ~1.5B, 4B, 8B/9B candidates conditionally on architecture.
- [ ] Hardware-aware inference plan, no false compatibility claims.
- [ ] Streaming local API, packaging, model manager and minimal UI if needed.

## Reference machine and constraints

HP OMEN Slim 16 (2025); Core Ultra 7 255H, Intel AI Boost NPU, 24 GB RAM, RTX 5060 Laptop GPU 8 GB, NVMe SSD, Windows 11. Initial acceleration goal is NPU-first while using shared system RAM, **without silent NVIDIA fallback**.

**Aspirational engineering threshold:** ≥20% lower *incremental peak system RAM* with ≥90% of generation throughput on a matched configuration. This is a target, not a measured result.
