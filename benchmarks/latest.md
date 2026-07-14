# ArmForge CI benchmark evidence

> **PASS** — native `aarch64` benchmark generated at 2026-07-14T04:31:02.329764+00:00.

| Metric | FP32 baseline | Arm64 INT8 | Improvement |
|---|---:|---:|---:|
| Model size | 86.22 MB | 21.94 MB | **74.55% smaller** |
| Mean latency | 19.31 ms | 7.83 ms | **59.47% lower** |
| P95 latency | 19.92 ms | 7.94 ms | — |
| Throughput | 51.78 items/s | 127.76 items/s | **146.74% higher** |
| Process RSS delta | 135.30 MB | 56.04 MB | **58.58% lower** |

## Quality gate

| Check | Result | Threshold |
|---|---:|---:|
| Output cosine similarity | `0.98344814` | `>= 0.98` |
| Mean absolute error | `0.02242831` | informational |
| Outputs finite | `True` | `true` |
| Native Arm64 runner | `True` | `true` |

## Reproduction details

- Model: `sentence-transformers/all-MiniLM-L6-v2` at revision `94ea1512acaefbfe2e255b2d2ea4bf0d9d7b3dc3`
- Workload: batch `1`, sequence length `128`, `75` measured runs after `15` warmups
- Runtime: ONNX Runtime `1.27.0`, Python `3.12.13`, `4` intra-op threads
- Host: `Linux-6.17.0-1018-azure-aarch64-with-glibc2.39`
- Execution provider: `CPUExecutionProvider`

The workflow downloads a checksum-pinned FP32 model, applies per-channel signed INT8 dynamic quantization, runs both models in isolated processes with identical deterministic inputs, compares their outputs, and publishes the raw JSON alongside this report.
