from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


DEFAULT_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "94ea1512acaefbfe2e255b2d2ea4bf0d9d7b3dc3"
DEFAULT_FILENAME = "onnx/model.onnx"
DEFAULT_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_default_model(destination: Path) -> Path:
    """Download a pinned, Apache-2.0 MiniLM ONNX model and verify its digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == DEFAULT_SHA256:
        return destination

    url = (
        f"https://huggingface.co/{DEFAULT_REPOSITORY}/resolve/"
        f"{DEFAULT_REVISION}/{DEFAULT_FILENAME}?download=true"
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ArmForge-CI/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)

    actual = sha256_file(temporary)
    if actual != DEFAULT_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Model checksum mismatch: expected {DEFAULT_SHA256}, received {actual}")
    temporary.replace(destination)
    return destination


def quantize_for_arm64(source: Path, destination: Path) -> Path:
    """Create a per-channel signed INT8 model suited to Arm64 CPU inference."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        extra_options={"WeightSymmetric": True},
    )
    return destination

