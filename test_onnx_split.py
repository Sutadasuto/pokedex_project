"""Evaluate the ONNX classifier on the reproducible held-out split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pokedex_data")
    parser.add_argument("--model", default="pokemon_classifier_1025.onnx")
    parser.add_argument("--labels", default="pokemon_labels_1025.json")
    parser.add_argument("--output", default="onnx_test_split")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--heatmap-tick-step", type=int, default=25)
    return parser.parse_args()


def load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import onnxruntime
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Install the evaluation dependencies with: "
            "pip install numpy pillow onnxruntime matplotlib"
        ) from exc
    return np, onnxruntime, Image, plt


def held_out_samples(
    dataset: Path, classes: list[str], val_fraction: float, seed: int
) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []
    for index, class_name in enumerate(classes):
        files = sorted(
            path
            for path in (dataset / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"Class {class_name!r} has no supported images")
        rng.shuffle(files)
        count = max(1, round(len(files) * val_fraction))
        if count >= len(files):
            count = 1 if len(files) > 1 else 0
        samples.extend((path, index) for path in files[:count])
    if not samples:
        raise ValueError("The held-out split contains no images")
    return samples


def preprocess(image: Any, np: Any, image_cls: Any) -> Any:
    image = image.convert("RGB").resize((224, 224), image_cls.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1))


def probabilities(logits: Any, np: Any) -> Any:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.sum(values)


def main() -> None:
    args = parse_args()
    np, onnxruntime, image_cls, plt = load_dependencies()
    dataset = Path(args.dataset)
    output = Path(args.output)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    classes = [
        f"{item['id']:04d}_{item['name']}"
        for item in sorted(labels, key=lambda item: item["index"])
    ]
    if len(classes) != 1025:
        raise ValueError(f"Expected 1,025 labels, found {len(classes)}")
    samples = held_out_samples(dataset, classes, args.val_fraction, args.seed)
    session = onnxruntime.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if session.get_inputs()[0].shape[-2:] != [224, 224]:
        raise ValueError(f"Expected a 224x224 model input, got {session.get_inputs()[0].shape}")

    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    top5_correct = 0
    top1_correct = 0
    per_class_total = np.zeros(len(classes), dtype=np.int64)
    per_class_top1 = np.zeros(len(classes), dtype=np.int64)
    per_class_top5 = np.zeros(len(classes), dtype=np.int64)
    per_class_predictions = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for sample_index, (path, expected_index) in enumerate(samples, start=1):
        with image_cls.open(path) as image:
            tensor = np.expand_dims(preprocess(image, np, image_cls), axis=0)
        logits = session.run([output_name], {input_name: tensor})[0][0]
        probs = probabilities(logits, np)
        top_indices = np.argsort(logits)[::-1][:args.topk]
        predicted_index = int(top_indices[0])
        top1_correct += predicted_index == expected_index
        top5_correct += expected_index in [int(index) for index in top_indices]
        per_class_total[expected_index] += 1
        per_class_top1[expected_index] += predicted_index == expected_index
        per_class_top5[expected_index] += expected_index in [
            int(index) for index in top_indices
        ]
        confusion[expected_index, predicted_index] += 1
        per_class_predictions[expected_index, predicted_index] += 1

        destination = output / classes[expected_index]
        destination.mkdir(parents=True, exist_ok=True)
        image_destination = destination / path.name
        shutil.copy2(path, image_destination)
        report = [
            f"Expected: #{labels[expected_index]['id']:04d} "
            f"{labels[expected_index]['name']}",
            "",
            "Top predictions:",
        ]
        for rank, index in enumerate(top_indices, start=1):
            label = labels[int(index)]
            report.append(
                f"{rank}. #{label['id']:04d} {label['name']} "
                f"{float(probs[index]) * 100:.2f}%"
            )
        image_destination.with_suffix(".txt").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        if sample_index % 100 == 0 or sample_index == len(samples):
            print(f"Processed {sample_index}/{len(samples)}")

    accuracy = top1_correct / len(samples)
    top5_accuracy = top5_correct / len(samples)
    matrix_path = output / "confusion_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "true_dex,"
            + ",".join(str(item["id"]) for item in labels)
            + "\n"
        )
        for index, row in enumerate(confusion):
            handle.write(
                str(labels[index]["id"])
                + ","
                + ",".join(str(int(value)) for value in row)
                + "\n"
            )

    heatmap_path = output / "confusion_matrix_heatmap.png"
    figure_size = max(14, min(32, len(classes) / 35))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(confusion, interpolation="nearest", cmap="magma")
    tick_step = max(1, args.heatmap_tick_step)
    tick_positions = list(range(0, len(classes), tick_step))
    if tick_positions[-1] != len(classes) - 1:
        tick_positions.append(len(classes) - 1)
    tick_labels = [str(labels[index]["id"]) for index in tick_positions]
    axis.set_xticks(tick_positions, tick_labels, rotation=90)
    axis.set_yticks(tick_positions, tick_labels)
    axis.set_xlabel("Predicted National Dex number")
    axis.set_ylabel("True National Dex number")
    axis.set_title("ONNX confusion matrix (counts)")
    figure.colorbar(image, ax=axis, label="Images")
    figure.tight_layout()
    figure.savefig(heatmap_path, dpi=180)
    plt.close(figure)

    misclassified = []
    for index in range(len(classes)):
        errors = int(per_class_total[index] - per_class_top1[index])
        if errors:
            incorrect_predictions = per_class_predictions[index].copy()
            incorrect_predictions[index] = 0
            rival = int(np.argmax(incorrect_predictions))
            misclassified.append({
                "id": labels[index]["id"],
                "name": labels[index]["name"],
                "images": int(per_class_total[index]),
                "top1_correct": int(per_class_top1[index]),
                "top5_correct": int(per_class_top5[index]),
                "errors": errors,
                "most_common_prediction": {
                    "id": labels[rival]["id"],
                    "name": labels[rival]["name"],
                    "count": int(incorrect_predictions[rival]),
                },
            })
    misclassified.sort(key=lambda item: (-item["errors"], item["id"]))
    summary = {
        "images": len(samples),
        "top1_correct": int(top1_correct),
        "top1_accuracy": accuracy,
        "top5_correct": int(top5_correct),
        "top5_accuracy": top5_accuracy,
        "classes": len(classes),
        "misclassified_species": misclassified,
        "confusion_matrix": "confusion_matrix.csv",
        "confusion_heatmap": "confusion_matrix_heatmap.png",
    }
    (output / "test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    markdown = [
        "# ONNX test summary",
        "",
        f"- Images: {len(samples)}",
        f"- Top-1 accuracy: {accuracy:.4%} ({top1_correct}/{len(samples)})",
        f"- Top-5 accuracy: {top5_accuracy:.4%} ({top5_correct}/{len(samples)})",
        "",
        "## Most misclassified species",
        "",
        "| Dex | Species | Images | Errors | Top-1 | Top-5 | Most common prediction |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in misclassified[:50]:
        rival = item["most_common_prediction"]
        markdown.append(
            f"| {item['id']} | {item['name']} | {item['images']} | "
            f"{item['errors']} | {item['top1_correct'] / item['images']:.2%} | "
            f"{item['top5_correct'] / item['images']:.2%} | "
            f"#{rival['id']} {rival['name']} ({rival['count']}) |"
        )
    (output / "test_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Evaluated {len(samples)} images")
    print(f"Top-1 accuracy: {accuracy:.4%}")
    print(f"Top-5 accuracy: {top5_accuracy:.4%}")
    print(f"Summary: {output / 'test_summary.md'}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
