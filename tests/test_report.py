from armforge.report import render_markdown


def test_report_contains_evidence() -> None:
    payload = {
        "generated_at": "2026-07-14T00:00:00+00:00",
        "model": {"repository": "model", "revision": "abc"},
        "workload": {"batch_size": 1, "sequence_length": 128, "runs": 10, "warmups": 2},
        "baseline": {
            "model_mb": 90.0,
            "latency": {"mean_ms": 10.0, "p95_ms": 11.0, "throughput_items_per_second": 100.0},
            "memory": {"rss_delta_mb": 100.0},
        },
        "optimized": {
            "model_mb": 23.0,
            "latency": {"mean_ms": 4.0, "p95_ms": 5.0, "throughput_items_per_second": 250.0},
            "memory": {"rss_delta_mb": 40.0},
            "runtime": {
                "architecture": "aarch64",
                "onnxruntime": "1.22",
                "python": "3.12",
                "threads": 4,
                "platform": "Linux",
                "providers": ["CPUExecutionProvider"],
            },
        },
        "improvement": {
            "model_size_reduction_percent": 74.44,
            "latency_reduction_percent": 60.0,
            "throughput_gain_percent": 150.0,
            "memory_reduction_percent": 60.0,
        },
        "validation": {
            "cosine_similarity": 0.999,
            "mean_absolute_error": 0.001,
            "finite": True,
        },
        "verdict": {
            "passed": True,
            "native_arm64": True,
            "minimum_cosine_similarity": 0.98,
        },
    }
    report = render_markdown(payload)
    assert "native `aarch64`" in report
    assert "74.44% smaller" in report
    assert "PASS" in report

