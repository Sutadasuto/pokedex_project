# Dataset generator

[`dataset_generator.py`](./dataset_generator.py) builds a provenance-aware dataset
for National Pokédex species 1 through 1025. It always creates `labels.json` and
`dataset_manifest.csv`. The manifest records the source page, license metadata
when available, image hash, dimensions, and form information.

## Sources

By default the generator uses:

- **PokeAPI**: all available HOME and Official Artwork sprites, including
  varieties and forms.
- **Pokémon Database**: one dedicated official-artwork gallery page per
  species, parsed only for images hosted by `img.pokemondb.net`.
- **PkmnCards**: a species archive at `/name/<species>/`, downloading each
  card image without an API and cropping the standard illustration panel to a
  centered square JPEG. A card is accepted only when its page title names the
  exact species; variants such as `Eevee ex` are excluded. The original card
  page remains in the manifest.

The current files already in `pokedex_data` are scanned first and retained.
Exact duplicates from new sources are skipped by SHA-256 hash.

Collect Pokémon TCG artwork from PkmnCards as an additional source:

```powershell
python .\dataset_generator.py `
  --sources pokeapi pokemondb pkmncards `
  --search-limit 30
```

Card scans are intentionally opt-in because they are a distinct visual domain
from the game/anime/merchandise images and are subject to the rights of their
publishers. Review the manifest and the applicable terms before redistribution.

- **Brave Images**: an opt-in provider that uses the authenticated Brave
image-search API with the query `"<species>" plush`.

Collect plush search results through the Brave Search API. Put only your API
key, with no surrounding quotes, in `brave_api_key.txt` in the project
directory. The file is read locally and the key is never written to the
manifest or logs:

```powershell
python .\dataset_generator.py `
  --output .\pokedex_data `
  --first 1 --last 1025 `
  --sources brave-images `
  --search-limit 20
```

Brave search results are discovery data, not a license grant. The manifest
records the result source URL, but image ownership and permitted use must be
reviewed before redistribution or model publication. When Brave returns HTTP
429, the generator retries up to three times with a one-minute wait before
each retry. If the limit persists, the generator stops and saves the partial
manifest, then prints the `--first` value needed to resume at the species that
was not collected. When `brave-images` is selected, the normal delay between
species is at least five seconds; a larger `--delay` value is still respected.
The API's `url` field identifies the source page; the provider downloads the
original image from the result's `properties.url` field.

To use a different key-file location:

```powershell
python .\dataset_generator.py --sources brave-images `
  --brave-api-key-file C:\secure\brave_api_key.txt
```

Downloaded images are normalized to PNG. Brave image results are additionally
center-cropped to a square, matching the centered square crop used for TCG
artwork. The `--images-per-source` option accepts either one shared value or
one value for each selected source, in source order:

If a search result returns HTML, a challenge page, or another invalid payload
instead of an image, that candidate is skipped and collection continues.

```powershell
python .\dataset_generator.py --sources pokeapi brave-images --images-per-source 20 10
```

## Fine-tuning the published 151-class model

The published Hugging Face model is a PyTorch/timm **DenseNet121**, despite
the repository name containing `mobilenetv3`. It uses 224x224 RGB inputs,
ImageNet normalization, and a 151-output classifier. The fine-tuning script
preserves the DenseNet backbone, discards that classifier, and creates a new
1025-output classifier based on the dataset directories:

```powershell
pip install torch torchvision timm pillow
curl.exe -L `
  -o .\pokemon_densenet121_best.pth `
  https://huggingface.co/BiernyVR/pokemon-classifier-mobilenetv3/resolve/main/pokemon_densenet121_best.pth
python .\train_finetune.py `
  --dataset .\pokedex_data `
  --checkpoint .\pokemon_densenet121_best.pth `
  --output .\pokemon_densenet1025_best.pth `
  --labels-output .\pokemon_labels_1025.json
```

The script automatically selects CUDA when available. To require the GPU
explicitly, add `--device cuda`; it will fail immediately rather than silently
falling back to the CPU if CUDA is unavailable.

The script requires exactly 1,025 class directories and performs a
class-balanced train/validation split. It trains the new head first, then
unfreezes the final DenseNet blocks for fine-tuning. The original 151-class
checkpoint is not overwritten.

## Converting the trained model to ONNX

The Android app expects an ONNX model with input name `input`, shape
`[1, 3, 224, 224]`, float32 values normalized with ImageNet mean and standard
deviation, and a logits output. Convert the 1,025-class checkpoint with:

```powershell
pip install onnxruntime
pip install --no-deps onnx==1.12.0
python .\convert_to_onnx.py `
  --checkpoint .\pokemon_densenet1025_best.pth `
  --output .\pokemon_classifier_1025.onnx `
  --labels .\pokemon_labels_1025.json `
  --verify
```

The script preserves the classifier output order from the checkpoint and
rewrites the labels file in the format used by the existing Android app.
Update the Android app's model filename, class count, and labels asset from
151 to 1,025 when deploying this model. Preprocessing remains in the Android
client; the ONNX graph expects normalized NCHW tensors rather than raw RGB
pixels.

To evaluate the ONNX model, use the same seed and 15% class-balanced held-out
split used by training:

```powershell
python .\test_onnx_split.py `
  --dataset .\pokedex_data `
  --model .\pokemon_classifier_1025.onnx `
  --labels .\pokemon_labels_1025.json `
  --output .\onnx_test_split
```

The script creates one folder per species, copies each held-out image there,
and writes a same-name `.txt` file containing the expected species and the
top-five predictions with percentages. It also writes `test_summary.md` and
`test_summary.json` with Top-1/Top-5 accuracy and the most misclassified
species, `confusion_matrix.csv`, and
`confusion_matrix_heatmap.png`. Heatmap axes use National Dex numbers; labels
are shown every 25 classes by default.

To test one image directly:

```powershell
python .\test_onnx_image.py `
  --model .\pokemon_classifier_1025.onnx `
  --image .\path\to\image.png `
  --labels .\pokemon_labels_1025.json
```

The command prints the top five National Dex predictions and their
percentages.

## Recommended first run

Run a small range before downloading all 1,025 species:

```powershell
python .\dataset_generator.py `
  --output .\pokedex_data_test `
  --first 1 --last 3 `
  --sources pokeapi pokemondb `
  --images-per-source 10 `
  --verbose
```

After checking the manifest and image quality, run the full collection:

```powershell
python .\dataset_generator.py `
  --output .\pokedex_data `
  --first 1 --last 1025 `
  --sources pokeapi pokemondb `
  --images-per-source 20
```

The downloader is intentionally conservative: it validates image bytes with
Pillow, rejects images smaller than 64x64, skips duplicates, and does not
silently convert or alter source images. Conversion to RGB and train/validation
/test grouping should happen in the training-preparation step, using the
manifest's hash and source fields to prevent near-duplicate leakage.

Progress is displayed on stderr while species are processed, so it remains
visible alongside the normal logging output and does not require `tqdm` or
another additional dependency.

Pokémon assets and names are owned by their respective rights holders. Keep the
source and license columns, review each source's terms, and do not redistribute
downloaded images or use them commercially without permission.
