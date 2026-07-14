from __future__ import annotations

import json
from pathlib import Path


def write_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _metric(value: object, suffix: str = "") -> str:
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def render_markdown(payload: dict[str, object]) -> str:
    baseline = payload["baseline"]
    optimized = payload["optimized"]
    delta = payload["improvement"]
    validation = payload["validation"]
    runtime = optimized["runtime"]
    verdict = payload["verdict"]
    status = "PASS" if verdict["passed"] else "FAIL"
    return f"""# ArmForge CI benchmark evidence

> **{status}** — native `{runtime['architecture']}` benchmark generated at {payload['generated_at']}.

| Metric | FP32 baseline | Arm64 INT8 | Improvement |
|---|---:|---:|---:|
| Model size | {_metric(baseline['model_mb'], ' MB')} | {_metric(optimized['model_mb'], ' MB')} | **{_metric(delta['model_size_reduction_percent'], '% smaller')}** |
| Mean latency | {_metric(baseline['latency']['mean_ms'], ' ms')} | {_metric(optimized['latency']['mean_ms'], ' ms')} | **{_metric(delta['latency_reduction_percent'], '% lower')}** |
| P95 latency | {_metric(baseline['latency']['p95_ms'], ' ms')} | {_metric(optimized['latency']['p95_ms'], ' ms')} | — |
| Throughput | {_metric(baseline['latency']['throughput_items_per_second'], ' items/s')} | {_metric(optimized['latency']['throughput_items_per_second'], ' items/s')} | **{_metric(delta['throughput_gain_percent'], '% higher')}** |
| Process RSS delta | {_metric(baseline['memory']['rss_delta_mb'], ' MB')} | {_metric(optimized['memory']['rss_delta_mb'], ' MB')} | **{_metric(delta['memory_reduction_percent'], '% lower')}** |

## Quality gate

| Check | Result | Threshold |
|---|---:|---:|
| Output cosine similarity | `{validation['cosine_similarity']}` | `>= {verdict['minimum_cosine_similarity']}` |
| Mean absolute error | `{validation['mean_absolute_error']}` | informational |
| Outputs finite | `{validation['finite']}` | `true` |
| Native Arm64 runner | `{verdict['native_arm64']}` | `true` |

## Reproduction details

- Model: `{payload['model']['repository']}` at revision `{payload['model']['revision']}`
- Workload: batch `{payload['workload']['batch_size']}`, sequence length `{payload['workload']['sequence_length']}`, `{payload['workload']['runs']}` measured runs after `{payload['workload']['warmups']}` warmups
- Runtime: ONNX Runtime `{runtime['onnxruntime']}`, Python `{runtime['python']}`, `{runtime['threads']}` intra-op threads
- Host: `{runtime['platform']}`
- Execution provider: `{', '.join(runtime['providers'])}`

The workflow downloads a checksum-pinned FP32 model, applies per-channel signed INT8 dynamic
quantization, runs both models in isolated processes with identical deterministic inputs, compares
their outputs, and publishes the raw JSON alongside this report.
"""


def write_markdown(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_markdown(payload), encoding="utf-8")

