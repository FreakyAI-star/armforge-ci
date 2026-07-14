# ArmForge CI

[![Native Arm64 evidence](https://github.com/FreakyAI-star/armforge-ci/actions/workflows/benchmark.yml/badge.svg)](https://github.com/FreakyAI-star/armforge-ci/actions/workflows/benchmark.yml)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-c9ff36)](LICENSE)
[![Arm64 native](https://img.shields.io/badge/runner-native%20Arm64-ff6b35)](https://github.com/FreakyAI-star/armforge-ci/actions)

![ArmForge CI cover](assets/armforge-ci-cover.png)

**Optimization claims with receipts.** ArmForge CI turns an ONNX model into an Arm64-targeted
INT8 model, runs a controlled FP32-versus-INT8 benchmark on native Arm hardware, validates output
quality, and publishes both human-readable and machine-readable evidence.

This is a **Cloud AI** entry for the 2026 Arm AI Optimization Challenge. It focuses on two
developer problems: architecture-specific optimization is hard to reproduce, and benchmark claims
often omit the controls needed to trust them.

## What it delivers

- Per-channel signed INT8 dynamic quantization for ONNX language models
- Native four-core Arm64 execution on free public GitHub-hosted runners
- Isolated workers so one model's allocator cannot contaminate the other's RSS evidence
- Mean, P50, P95, min/max latency, throughput, model size, and process memory
- Full-output cosine, mean-error, finite-value, and architecture quality gates
- Checksum-pinned model inputs and deterministic synthetic token workloads
- Raw JSON, a Markdown report, workflow summary, downloadable artifact, and public dashboard
- A reusable composite GitHub Action for other open-source model repositories

## Latest native evidence

- [Readable benchmark report](benchmarks/latest.md)
- [Raw benchmark JSON](benchmarks/latest.json)
- [Public evidence dashboard](https://freakyai-star.github.io/armforge-ci/)
- [Benchmark methodology](docs/methodology.md)
- [System architecture](docs/architecture.md)

The first workflow run replaces the placeholder evidence with results from
`ubuntu-24.04-arm`. Every subsequent run can be reproduced from the Actions tab.

## Run it

```bash
python -m pip install -e .
armforge --runs 50 --warmups 10 --threads 4 --output benchmarks
```

The default workload downloads a checksum-pinned, Apache-2.0
`sentence-transformers/all-MiniLM-L6-v2` ONNX model. To benchmark your own ONNX model:

```bash
armforge --model path/to/model.onnx --output benchmarks
```

To enforce native execution in CI:

```bash
armforge --require-arm64
```

## Use it in another public repository

```yaml
jobs:
  optimize:
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - uses: FreakyAI-star/armforge-ci@main
        with:
          runs: "75"
          sequence-length: "128"
          output: benchmarks
```

## Why Arm64

Public GitHub repositories can run natively on four-core `ubuntu-24.04-arm` workers. That makes
architecture validation available to every open-source maintainer without buying hardware or
managing a cloud account. ArmForge CI converts that access into a repeatable optimization gate,
with reusable evidence suitable for pull-request review and release decisions.

## Fair-benchmark choices

Both models use the same ONNX Runtime version, execution provider, graph optimization level,
thread count, batch, sequence length, seed, generated inputs, warmups, and measurement count. Model
workers are separate processes. Speed is reported rather than used as a pass/fail condition because
shared CI hosts vary; numerical quality and native architecture are enforced.

## Project layout

```text
src/armforge/          quantization, isolated benchmark workers, gates, reports
.github/workflows/     native Arm64 evidence and cross-platform tests
benchmarks/            committed JSON and Markdown proof
site/                  zero-framework public evidence dashboard
docs/                  architecture and methodology
tests/                 deterministic unit tests
action.yml             reusable composite GitHub Action
```

## License and model provenance

ArmForge CI is Apache-2.0 licensed. The default MiniLM model is also Apache-2.0 and is downloaded
from a pinned Hugging Face revision; it is not redistributed in this repository. Arm and related
marks belong to their respective owners. This project is an independent hackathon submission.
