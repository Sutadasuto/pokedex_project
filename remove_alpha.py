from PIL import Image
from pathlib import Path

DATASET_DIR = r"./pokedex_data"

dataset_path = Path(DATASET_DIR)

valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]

for path in dataset_path.rglob("*"):

    if path.suffix.lower() not in valid_extensions:
        continue

    try:
        img = Image.open(path).convert("RGBA")

        # White background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))

        # Composite transparency onto white
        merged = Image.alpha_composite(background, img)

        # Convert to RGB
        merged = merged.convert("RGB")

        # Save clean image
        merged.save(path)

        print("Cleaned:", path)

    except Exception as e:
        print("Failed:", path, e)