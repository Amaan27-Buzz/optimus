# Benchmarking protocol

## Comparable workloads

Fix exact model weights, model hash, quantization and group size, prompt, context, output count, sampling, power plan, device, temperature and other background workload conditions. Collect cold and warm results, multiple samples, ordered A/B/A tests and variation. Avoid comparing a GPU baseline with an NPU result as proof of an engine-level speedup.

Measure:
- Peak private/committed process memory; peak working set; pre-launch baseline and peak incremental system RAM.
- Model load/cold compile, prefill, time-to-first-token, decode tokens/sec, p50/p95.
- Device evidence, errors/fallbacks, KV memory and graph allocation lifetimes.
- Disk reads/pages during **generation** vs **loading**, not just aggregate disk counters.
- Energy/token only with trusted power measurement.
- Output correctness, logit error and quality consistency.

Unknown counters must be reported as **N/A**, never zero. Unrealistic Windows disk counter spikes require independent validation.

## Prior NPU experiments — NOT Optimus

DeepSeek-R1-Distill-Qwen-1.5B INT4 on HP OMEN Slim 16, OpenVINO GenAI, NPU requested, Oct 2026 (three generations per mode):

| Metric | Standard | Weightless blob |
| --- | ---: | ---: |
| Generation | 9.829 tok/s | 9.410 tok/s |
| Mean TTFT | 826.2 ms | 835.6 ms |
| Peak process RSS | 2.716 GiB | 2.708 GiB |
| Incremental peak system RAM | 3.108 GiB | 3.070 GiB |
| Pipeline load | 8.10 s | 3.12 s |

Different host CPU/background load limits the validity of the observed differences. Weightless loading did **not** achieve a material inference-RAM reduction. An earlier standard run produced 11.71 tok/s at ~2.72 GiB peak process RSS; these are not directly interchangeable measurements.

Task Manager showed NPU activity; automated utilization counters were unavailable. NVIDIA utilization was 0% in sampled runs. Preserve the original CSV/JSON locally for future import; do not commit model files, private machine identifiers or user paths.

`optimus-cli bench` is only a small CPU matrix-multiplication microbenchmark, not an LLM benchmark or comparative claim.

## Intended report schema (future)

~~~json
{
  "engine": "optimus",
  "model_sha256": "required",
  "requested_device": "NPU",
  "actual_device": null,
  "quantization": null,
  "peak_system_ram_delta_gib": null,
  "peak_process_private_gib": null,
  "ttft_ms": null,
  "decode_tok_s": null,
  "energy_j_per_token": null,
  "unavailable_sensors": []
}
~~~
