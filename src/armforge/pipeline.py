from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .metrics import compare_outputs, percentage_change
from .model import (
    DEFAULT_FILENAME,
    DEFAULT_REPOSITORY,
    DEFAULT_REVISION,
    DEFAULT_SHA256,
    download_default_model,
    quantize_for_arm64,
)
from .report import write_json, write_markdown


def _load_outputs(path: Path) -> list[np.ndarray]:
    with np.load(path) as data:
        return [data[key] for key in sorted(data.files)]


def _run_worker(model: Path, output: Path, options: dict[str, int]) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "armforge.worker",
        "--model",
        str(model),
        "--output",
        str(output),
    ]
    for key, value in options.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def run_pipeline(
    output_directory: Path,
    model_path: Path | None = None,
    runs: int = 50,
    warmups: int = 10,
    threads: int = 4,
    batch_size: int = 1,
    sequence_length: int = 128,
    seed: int = 42,
    minimum_cosine_similarity: float = 0.98,
    require_arm64: bool = False,
) -> tuple[dict[str, object], bool]:
    cache = Path(".armforge")
    source = model_path or download_default_model(cache / "models" / "minilm-fp32.onnx")
    optimized = cache / "models" / "minilm-qint8-arm64.onnx"
    quantize_for_arm64(source, optimized)

    options = {
        "runs": runs,
        "warmups": warmups,
        "threads": threads,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
    }
    with tempfile.TemporaryDirectory(prefix="armforge-") as temp:
        temporary = Path(temp)
        baseline_outputs = temporary / "baseline.npz"
        optimized_outputs = temporary / "optimized.npz"
        baseline = _run_worker(source, baseline_outputs, options)
        optimized_result = _run_worker(optimized, optimized_outputs, options)
        validation = compare_outputs(
            _load_outputs(baseline_outputs), _load_outputs(optimized_outputs)
        )

    baseline["model_mb"] = round(baseline["model_bytes"] / 1024 / 1024, 2)
    optimized_result["model_mb"] = round(optimized_result["model_bytes"] / 1024 / 1024, 2)
    improvement = {
        "model_size_reduction_percent": percentage_change(
            baseline["model_bytes"], optimized_result["model_bytes"]
        ),
        "latency_reduction_percent": percentage_change(
            baseline["latency"]["mean_ms"], optimized_result["latency"]["mean_ms"]
        ),
        "throughput_gain_percent": round(
            (
                optimized_result["latency"]["throughput_items_per_second"]
                / baseline["latency"]["throughput_items_per_second"]
                - 1.0
            )
            * 100.0,
            2,
        ),
        "memory_reduction_percent": percentage_change(
            baseline["memory"]["rss_delta_mb"], optimized_result["memory"]["rss_delta_mb"]
        ),
    }
    architecture = str(optimized_result["runtime"]["architecture"]).lower()
    native_arm64 = architecture in {"aarch64", "arm64"}
    passed = bool(
        validation["finite"]
        and validation["cosine_similarity"] >= minimum_cosine_similarity
        and (native_arm64 or not require_arm64)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "project": "ArmForge CI",
        "track": "Cloud AI",
        "model": {
            "repository": DEFAULT_REPOSITORY if model_path is None else "user-supplied",
            "revision": DEFAULT_REVISION if model_path is None else "local",
            "filename": DEFAULT_FILENAME if model_path is None else model_path.name,
            "sha256": DEFAULT_SHA256 if model_path is None else "not-pinned",
            "optimization": "per-channel signed INT8 dynamic quantization",
        },
        "workload": options,
        "baseline": baseline,
        "optimized": optimized_result,
        "improvement": improvement,
        "validation": validation,
        "verdict": {
            "passed": passed,
            "native_arm64": native_arm64,
            "minimum_cosine_similarity": minimum_cosine_similarity,
            "arm64_required": require_arm64,
            "local_invocation_architecture": platform.machine(),
        },
    }
    write_json(payload, output_directory / "latest.json")
    write_markdown(payload, output_directory / "latest.md")
    return payload, passed
