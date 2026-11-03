"""Fine-tune the published DenseNet121 Pokémon classifier for 1,025 species.

The source checkpoint is the PyTorch/timm DenseNet121 model published at
BiernyVR/pokemon-classifier-mobilenetv3. Its 151-class classifier is discarded
and replaced with a new classifier for the directories found in the dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pokedex_data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="pokemon_densenet1025_best.pth")
    parser.add_argument("--labels-output", default="pokemon_labels_1025.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=2,
        help="Number of final DenseNet feature blocks to unfreeze.",
    )
    return parser.parse_args()


def import_training_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import timm
        import torch
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(
            "This script requires PyTorch, torchvision, timm, and Pillow. "
            "Install them in the training environment with: "
            "pip install torch torchvision timm pillow"
        ) from exc
    return torch, timm, Image, transforms


def discover_classes(dataset_root: Path) -> list[str]:
    classes = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not classes:
        raise ValueError(f"No class directories found in {dataset_root}")
    return classes


def collect_samples(
    dataset_root: Path, classes: list[str], image_extensions: set[str]
) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_index, class_name in enumerate(classes):
        class_files = [
            path
            for path in (dataset_root / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ]
        if not class_files:
            raise ValueError(f"Class {class_name!r} has no supported images")
        samples.extend((path, class_index) for path in sorted(class_files))
    return samples


class ImageSamples:
    def __init__(self, samples: list[tuple[Path, int]], transform: Any):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


def split_samples(
    samples: list[tuple[Path, int]], val_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    rng = random.Random(seed)
    train: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    for class_samples in by_class.values():
        rng.shuffle(class_samples)
        validation_count = max(1, round(len(class_samples) * val_fraction))
        if validation_count >= len(class_samples):
            validation_count = 1 if len(class_samples) > 1 else 0
        validation.extend(class_samples[:validation_count])
        train.extend(class_samples[validation_count:])
    if not train or not validation:
        raise ValueError("The dataset is too small for the requested split")
    return train, validation


def load_backbone_weights(model: Any, checkpoint: Path, torch: Any) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    normalized = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if not key.removeprefix("module.").startswith("classifier.")
    }
    incompatible = model.load_state_dict(normalized, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if key != "classifier"]
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected[:10]}")


def configure_trainable_layers(model: Any, unfreeze_blocks: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if unfreeze_blocks:
        block_names = {
            f"denseblock{index}"
            for index in range(5 - unfreeze_blocks, 5)
        }
        for name, parameter in model.features.named_parameters():
            if name.split(".", 1)[0] in block_names or name.startswith("norm5."):
                parameter.requires_grad = True


def train_epoch(model: Any, loader: Any, loss_fn: Any, optimizer: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: Any, loader: Any, loss_fn: Any, device: Any, torch: Any
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    size = len(loader.dataset)
    return total_loss / size, correct / size


def main() -> None:
    args = parse_args()
    torch, timm, image_cls, transforms = import_training_dependencies()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset_root = Path(args.dataset)
    classes = discover_classes(dataset_root)
    if len(classes) != 1025:
        raise ValueError(f"Expected 1,025 class directories, found {len(classes)}")
    samples = collect_samples(
        dataset_root, classes, {".jpg", ".jpeg", ".png", ".webp"}
    )
    train_samples, val_samples = split_samples(
        samples, args.val_fraction, args.seed
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
        transforms.RandomAffine(
            degrees=35,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
        ),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    train_loader = torch.utils.data.DataLoader(
        ImageSamples(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        ImageSamples(val_samples, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = timm.create_model(
        "densenet121", pretrained=False, num_classes=len(classes), drop_rate=0.25
    )
    load_backbone_weights(model, Path(args.checkpoint), torch)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU device")
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    configure_trainable_layers(model, 0)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.head_lr,
        weight_decay=1e-4,
    )
    best_accuracy = -1.0
    for epoch in range(args.head_epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device, torch)
        print(
            f"head {epoch + 1}/{args.head_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state_dict": model.state_dict(), "classes": classes}, args.output)

    configure_trainable_layers(model, args.unfreeze_blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.fine_tune_lr, weight_decay=1e-4)
    for epoch in range(args.fine_tune_epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device, torch)
        print(
            f"fine {epoch + 1}/{args.fine_tune_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state_dict": model.state_dict(), "classes": classes}, args.output)

    labels = [
        {"index": index, "id": int(name[:4]), "name": name[5:]}
        for index, name in enumerate(classes)
    ]
    Path(args.labels_output).write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Saved {len(classes)}-class checkpoint to {args.output}")


if __name__ == "__main__":
    main()
