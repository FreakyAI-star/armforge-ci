from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="armforge",
        description="Quantize ONNX models and produce reproducible native Arm64 evidence.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-cosine", type=float, default=0.98)
    parser.add_argument("--require-arm64", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload, passed = run_pipeline(
        output_directory=args.output,
        model_path=args.model,
        runs=args.runs,
        warmups=args.warmups,
        threads=args.threads,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
        minimum_cosine_similarity=args.minimum_cosine,
        require_arm64=args.require_arm64,
    )
    result = payload["improvement"]
    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict}: {result['model_size_reduction_percent']}% smaller, "
        f"{result['latency_reduction_percent']}% lower mean latency; "
        f"report: {args.output / 'latest.md'}"
    )
    if not passed:
        raise SystemExit(1)

