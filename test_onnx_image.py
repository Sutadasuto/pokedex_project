"""Run top-5 predictions for one image with an ONNX Pokémon classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument(
        "--labels",
        default="pokemon_labels_1025.json",
        help="Path to the labels JSON file",
    )
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Install the dependencies with: pip install numpy pillow onnxruntime"
        ) from exc
    return np, onnxruntime, Image


def preprocess(image: Any, image_cls: Any, np: Any) -> Any:
    image = image.convert("RGB").resize((224, 224), image_cls.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.expand_dims(np.transpose(array, (2, 0, 1)), axis=0)


def main() -> None:
    args = parse_args()
    np, onnxruntime, image_cls = load_dependencies()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    labels = sorted(labels, key=lambda item: item["index"])
    if not labels:
        raise ValueError("The labels file contains no labels")
    if args.topk < 1:
        raise ValueError("--topk must be at least 1")

    session = onnxruntime.InferenceSession(
        args.model, providers=["CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    with image_cls.open(args.image) as image:
        tensor = preprocess(image, image_cls, np)
    logits = session.run([output_name], {input_meta.name: tensor})[0][0]
    top_indices = np.argsort(logits)[::-1][: min(args.topk, len(labels))]
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)

    print(f"Image: {args.image}")
    print("Top predictions:")
    for rank, index in enumerate(top_indices, start=1):
        label = labels[int(index)]
        print(
            f"{rank}. #{label['id']:04d} {label['name']} "
            f"{float(probabilities[index]) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
