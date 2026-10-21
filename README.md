# Optimus

**A memory-first, hardware-aware LLM inference engine built from scratch in Rust.**

Optimus aims to run useful models on constrained consumer hardware with low memory pressure, predictable performance, and verified accelerator execution. Unlike a wrapper, Optimus will implement its own model execution, tensor operations, quantization, KV cache, memory planning, and scheduler.

> **Early scaffold, not a working LLM runtime.** Only the reference CPU tensor primitives, logical memory accounting, tests, and a developer CLI are implemented. NPU execution, model loading, and token generation are planned. No performance advantage over Ollama has yet been demonstrated.

## Principles

- No llama.cpp, Ollama, or OpenVINO GenAI in the runtime execution path.
- CPU correctness first; direct Intel NPU integration through vendor Level Zero driver and compiler interfaces later.
- Explicit memory budgets before allocations; measure peak private/committed bytes, working set, and system RAM delta.
- No silent NPU-to-CPU/GPU fallback; report actual execution devices.
- Experiment with memory-mapped/SSD-tiered weights only if measurements show useful savings.
- Measure equal model weights, quantization, context, output length, power state, and device configuration before claiming improvement.
- Small Rust crates; no unnecessary middleware or speculative modules.

## Architecture

~~~text
 CLI / local API (API planned)
           |
  Model loader + tokenizer (planned)
           |
  Transformer / KV cache / scheduler (planned)
           |
  Buffer and memory planner (planned)
           |
  +--------+--------------------+
  | Reference CPU | Intel NPU   |
  | implemented   | planned     |
  +---------------+-------------+
          |               |
      Rust CPU ops   Level Zero + Intel NPU compiler
~~~

The Intel NPU accepts compatible **compiled graphs**, not arbitrary Rust kernels. Vendor compiler/driver use is a hardware boundary, not dependence on another LLM inference runtime.

## Run the scaffold

Requirements: stable Rust and Cargo. No model download is needed.

~~~powershell
git clone https://github.com/Amaan27-Buzz/optimus.git
cd optimus
cargo test --workspace
cargo run -p optimus-cli -- doctor
cargo run -p optimus-cli -- smoke
cargo run -p optimus-cli -- bench
~~~

`doctor` describes **implemented** capability, not hardware detection. `smoke` tests a CPU operation. `bench` times a small CPU reference matrix multiply; it is not a token-generation benchmark.

## Codebase

| Path | Function |
| --- | --- |
| `crates/optimus-core` | Correctness-first f32 kernels, INT4 unpacking, logical memory accounting |
| `crates/optimus-cli` | CLI and reference microbenchmark |
| `docs/PLAN.md` | Milestones and exit criteria |
| `docs/ARCHITECTURE.md` | NPU and memory architecture |
| `docs/BENCHMARKING.md` | Fair measurements and results from earlier experiments |

**First meaningful release milestone:** load a small Qwen checkpoint, generate text using *our own* CPU code, and measure all significant memory allocations. Then verify native NPU graph execution before moving model inference onto the accelerator.

See [the plan](docs/PLAN.md). Licensing is not yet selected; public repository visibility does not grant a license.
