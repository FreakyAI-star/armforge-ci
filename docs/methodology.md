# Benchmark methodology

## What is measured

- End-to-end `InferenceSession.run` wall-clock latency with `perf_counter_ns`
- P50, P95, mean, minimum, maximum, and derived throughput
- Model size on disk
- Process RSS increase from before session creation through inference
- Numerical drift across every output tensor

## Controls

- Native `ubuntu-24.04-arm` GitHub-hosted runner
- Four ONNX Runtime intra-op threads and one inter-op thread
- Sequential execution and the CPU execution provider
- Identical batch size, sequence length, seed, inputs, warmups, and measured runs
- Independent worker process per model
- Checksum-pinned model revision

## Quality gate

The pipeline fails when outputs contain non-finite values, cosine similarity falls below `0.98`, or
the official workflow is not running natively on Arm64. Speed is reported rather than used as a
pass/fail gate because shared CI variance can otherwise create false failures.

## Limitations

GitHub-hosted machines are shared infrastructure and individual runs can vary. The evidence is
therefore reproducible, transparent, and directionally useful—not a replacement for dedicated
hardware lab measurements. Re-run the workflow to collect a fresh sample.

