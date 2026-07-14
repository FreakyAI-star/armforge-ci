# Architecture

```mermaid
flowchart LR
    A["Pinned MiniLM FP32 ONNX"] --> B["Per-channel QInt8 quantizer"]
    B --> C["Arm64 INT8 ONNX"]
    A --> D["Isolated benchmark worker"]
    C --> E["Isolated benchmark worker"]
    D --> F["Latency, throughput, RSS"]
    E --> F
    D --> G["Output tensors"]
    E --> G
    G --> H["Cosine quality gate"]
    F --> I["JSON + Markdown evidence"]
    H --> I
    I --> J["GitHub summary + public dashboard"]
```

ArmForge CI separates orchestration from measurement. Each model runs in a fresh Python process,
which prevents the first session's allocator and memory maps from contaminating the second model's
RSS evidence. Both workers receive identical, seeded token tensors.

The default model is checksum-pinned. A digest mismatch fails before any benchmark begins. ONNX
Runtime graph optimizations remain enabled for both models, while the only model-level change is
per-channel signed INT8 dynamic quantization.

