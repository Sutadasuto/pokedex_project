"""Convert the 1,025-class PyTorch checkpoint to Android-compatible ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="pokemon_classifier_1025.onnx")
    parser.add_argument("--labels", default="pokemon_labels_1025.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run ONNX Runtime and compare its output with PyTorch.",
    )
    return parser.parse_args()


def load_dependencies() -> tuple[Any, Any]:
    try:
        import timm
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Install the conversion dependencies with: "
            "pip install torch timm onnxruntime && "
            "pip install --no-deps onnx==1.12.0"
        ) from exc
    return torch, timm


def load_checkpoint(model: Any, checkpoint_path: Path, torch: Any) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint.get("classes") if isinstance(checkpoint, dict) else None
    state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    if not isinstance(classes, list) or len(classes) != model.classifier.out_features:
        raise ValueError(
            "Checkpoint must contain a classes list matching the classifier output"
        )
    return classes


def verify_export(
    model: Any, output_path: Path, torch: Any, image_size: int = 224
) -> None:
    import numpy as np
    import onnxruntime

    sample = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        expected = model(sample).numpy()
    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(["logits"], {"input": sample.numpy()})[0]
    if not np.allclose(expected, actual, rtol=1e-3, atol=1e-4):
        maximum_error = np.max(np.abs(expected - actual))
        raise RuntimeError(
            f"ONNX output differs from PyTorch; maximum absolute error is "
            f"{maximum_error:.6f}"
        )
    print("ONNX verification passed")


def main() -> None:
    args = parse_args()
    torch, timm = load_dependencies()
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    model = timm.create_model(
        "densenet121", pretrained=False, num_classes=1025, drop_rate=0.25
    )
    classes = load_checkpoint(model, checkpoint_path, torch)
    model.eval()
    sample = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        sample,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    Path(args.labels).write_text(json.dumps([
        {"index": index, "id": int(name[:4]), "name": name[5:]}
        for index, name in enumerate(classes)
    ], indent=2), encoding="utf-8")
    print(f"Saved ONNX model to {output_path}")
    print(f"Input: input [batch, 3, 224, 224] float32")
    print(f"Output: logits [batch, {len(classes)}] float32")
    if args.verify:
        verify_export(model, output_path, torch)


if __name__ == "__main__":
    main()
