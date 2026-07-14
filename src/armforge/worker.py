from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import psutil

from .metrics import latency_summary


def _dimension(value: object, fallback: int) -> int:
    return int(value) if isinstance(value, int) and value > 0 else fallback


def _numpy_type(onnx_type: str) -> np.dtype:
    if "int64" in onnx_type:
        return np.dtype(np.int64)
    if "int32" in onnx_type:
        return np.dtype(np.int32)
    if "float16" in onnx_type:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def make_inputs(session: object, batch_size: int, sequence_length: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = list(item.shape)
        resolved = [
            _dimension(value, batch_size if index == 0 else sequence_length)
            for index, value in enumerate(shape)
        ]
        dtype = _numpy_type(item.type)
        lowered = item.name.lower()
        if "input_ids" in lowered:
            value = rng.integers(100, 30_000, size=resolved, dtype=dtype)
        elif "attention_mask" in lowered:
            value = np.ones(resolved, dtype=dtype)
        elif "token_type" in lowered:
            value = np.zeros(resolved, dtype=dtype)
        elif "position" in lowered and len(resolved) >= 2:
            value = np.broadcast_to(np.arange(resolved[-1], dtype=dtype), resolved).copy()
        elif np.issubdtype(dtype, np.integer):
            value = np.zeros(resolved, dtype=dtype)
        else:
            value = rng.standard_normal(resolved).astype(dtype)
        feeds[item.name] = value
    return feeds


def execute(
    model: Path,
    output: Path,
    runs: int,
    warmups: int,
    threads: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> dict[str, object]:
    import onnxruntime as ort

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    feeds = make_inputs(session, batch_size, sequence_length, seed)

    outputs: list[np.ndarray] = []
    for _ in range(warmups):
        outputs = session.run(None, feeds)

    samples: list[float] = []
    peak_rss = process.memory_info().rss
    for _ in range(runs):
        started = time.perf_counter_ns()
        outputs = session.run(None, feeds)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        peak_rss = max(peak_rss, process.memory_info().rss)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{f"output_{index}": value for index, value in enumerate(outputs)})
    return {
        "model_bytes": model.stat().st_size,
        "latency": latency_summary(samples, batch_size),
        "memory": {
            "rss_delta_mb": round((peak_rss - rss_before) / 1024 / 1024, 2),
            "peak_rss_mb": round(peak_rss / 1024 / 1024, 2),
        },
        "outputs": [{"shape": list(value.shape), "dtype": str(value.dtype)} for value in outputs],
        "runtime": {
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "logical_cpus": os.cpu_count(),
            "threads": threads,
            "providers": session.get_providers(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = execute(**vars(args))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

