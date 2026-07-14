import numpy as np

from armforge.metrics import compare_outputs, latency_summary, percentage_change


def test_latency_summary_is_deterministic() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 4.0], batch_size=2)
    assert summary["runs"] == 4
    assert summary["mean_ms"] == 2.5
    assert summary["throughput_items_per_second"] == 800.0


def test_compare_outputs_reports_similarity() -> None:
    baseline = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]
    optimized = [np.array([[1.0, 2.01, 2.99]], dtype=np.float32)]
    result = compare_outputs(baseline, optimized)
    assert result["cosine_similarity"] > 0.999
    assert result["finite"] is True


def test_percentage_change() -> None:
    assert percentage_change(100, 25) == 75.0
    assert percentage_change(0, 25) == 0.0

