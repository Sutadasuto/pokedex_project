"""Build a provenance-aware 1,025 species image dataset.

The default sources are PokeAPI and Pokémon Database. PkmnCards and Brave
Images are optional adapters. Every downloaded image is validated, hashed, and
recorded in a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from html.parser import HTMLParser

import requests
from PIL import Image, ImageOps, UnidentifiedImageError


POKEAPI = "https://pokeapi.co/api/v2"
POKEMONDB_ARTWORK = "https://pokemondb.net/artwork"
PKMNCARDS_NAME = "https://pkmncards.com/name"
PKMNCARDS_BASE = "https://pkmncards.com"
BRAVE_IMAGES_API = "https://api.search.brave.com/res/v1/images/search"
BRAVE_SPECIES_DELAY = 1.0
BRAVE_RATE_LIMIT_DELAY = 60.0
DEFAULT_BRAVE_API_KEY_FILE = "brave_api_key.txt"
USER_AGENT = "pokedex-dataset-generator/1.0 (educational research)"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOG = logging.getLogger("dataset_generator")

# Relative bounds of the illustration panel in standard Pokémon TCG cards.
CARD_ART_LEFT = 0.078
CARD_ART_TOP = 0.095
CARD_ART_RIGHT = 0.925
CARD_ART_BOTTOM = 0.477


class BraveRateLimitError(RuntimeError):
    """Raised when Brave continues rate-limiting the public search endpoint."""


@dataclass(frozen=True)
class Species:
    dex_id: int
    name: str

    @property
    def directory(self) -> str:
        return f"{self.dex_id:04d}_{self.name}"


@dataclass
class Candidate:
    url: str
    source: str
    source_url: str
    form_name: str = "unknown"
    license_name: str = ""
    license_url: str = ""
    title: str = ""
    content: Optional[bytes] = None


def request_json(session: requests.Session, url: str, **kwargs) -> dict:
    response = session.get(url, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def load_species(session: requests.Session, first: int, last: int) -> list[Species]:
    result: list[Species] = []
    for dex_id in range(first, last + 1):
        data = request_json(session, f"{POKEAPI}/pokemon-species/{dex_id}/")
        result.append(Species(dex_id, data["name"]))
    return result


def pokeapi_candidates(
    session: requests.Session, species: Species
) -> Iterable[Candidate]:
    species_data = request_json(
        session, f"{POKEAPI}/pokemon-species/{species.dex_id}/"
    )
    for variety in species_data.get("varieties", []):
        variety_name = variety["pokemon"]["name"]
        data = request_json(session, variety["pokemon"]["url"])
        sprites = data.get("sprites", {})
        other = sprites.get("other", {})
        image_sets = {
            "official-artwork": other.get("official-artwork", {}),
            "home": other.get("home", {}),
        }
        for form_name, image_set in image_sets.items():
            for image_type, url in image_set.items():
                if isinstance(url, str) and url:
                    yield Candidate(
                        url=url,
                        source="pokeapi",
                        source_url=variety["pokemon"]["url"],
                        form_name=f"{variety_name}:{form_name}:{image_type}",
                        license_name="PokeAPI source terms",
                        title=variety_name,
                    )


class ImageURLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag != "img":
            return
        attributes = dict(attrs)
        for attribute in ("src", "data-src", "data-original"):
            value = attributes.get(attribute)
            if value and value.startswith("https://img.pokemondb.net/artwork/"):
                self.urls.append(html.unescape(value))


class CardPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.card_links: list[str] = []
        self.image_urls: list[str] = []
        self.card_name = ""
        self._reading_card_title = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, Optional[str]]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "a":
            href = attributes.get("href", "")
            if href.startswith(f"{PKMNCARDS_BASE}/card/"):
                self.card_links.append(href)
        elif tag == "meta" and attributes.get("property") == "og:image":
            content = attributes.get("content", "")
            if content:
                self.image_urls.append(html.unescape(content))
        elif tag == "h1" and "card-title" in attributes.get("class", "").split():
            self._reading_card_title = True

    def handle_data(self, data: str) -> None:
        if self._reading_card_title:
            self.card_name += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self._reading_card_title = False


def pokemondb_candidates(
    session: requests.Session, species: Species
) -> Iterable[Candidate]:
    page_url = f"{POKEMONDB_ARTWORK}/{species.name}"
    response = session.get(page_url, timeout=30)
    response.raise_for_status()
    parser = ImageURLParser()
    parser.feed(response.text)
    seen: set[str] = set()
    for image_url in parser.urls:
        if image_url in seen:
            continue
        seen.add(image_url)
        yield Candidate(
            url=image_url,
            source="pokemondb-artwork",
            source_url=page_url,
            form_name="official-artwork-gallery",
            license_name="Pokémon Database terms; verify before redistribution",
            title=f"{species.name} artwork",
        )


def pkmncards_candidates(
    session: requests.Session, species: Species, limit: int
) -> Iterable[Candidate]:
    """Find species card pages and their full card images without an API."""
    page_number = 1
    seen_pages: set[str] = set()
    yielded = 0
    while yielded < limit:
        url = f"{PKMNCARDS_NAME}/{species.name}/"
        if page_number > 1:
            url = f"{url}page/{page_number}/"
        response = session.get(url, timeout=30)
        if response.status_code == 404:
            LOG.debug("No more PkmnCards pages for %s after page %d", species.name, page_number - 1)
            break
        response.raise_for_status()
        search_parser = CardPageParser()
        search_parser.feed(response.text)
        card_pages = list(dict.fromkeys(search_parser.card_links))
        if not card_pages:
            break
        new_pages = [card_page for card_page in card_pages if card_page not in seen_pages]
        if not new_pages:
            break
        for card_page in new_pages:
            if yielded >= limit:
                break
            seen_pages.add(card_page)
            card_response = session.get(card_page, timeout=30)
            card_response.raise_for_status()
            card_parser = CardPageParser()
            card_parser.feed(card_response.text)
            card_name = card_parser.card_name.split("·", 1)[0].strip()
            if card_name.casefold() != species.name.casefold():
                LOG.debug(
                    "Ignoring non-exact PkmnCards name %r for %s",
                    card_name,
                    species.name,
                )
                continue
            if not card_parser.image_urls:
                continue
            image_url = card_parser.image_urls[0]
            card_slug = card_page.rstrip("/").rsplit("/", 1)[-1]
            yield Candidate(
                url=image_url,
                source="pkmncards-tcg",
                source_url=card_page,
                form_name=f"card-{card_slug}",
                license_name="Pokémon TCG card image; verify usage rights before redistribution",
                title=card_slug,
            )
            yielded += 1
        page_number += 1


def crop_card_artwork(content: bytes) -> bytes:
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB")
        width, height = image.size
        artwork = image.crop((
            round(width * CARD_ART_LEFT),
            round(height * CARD_ART_TOP),
            round(width * CARD_ART_RIGHT),
            round(height * CARD_ART_BOTTOM),
        ))
        square = ImageOps.fit(
            artwork,
            (min(artwork.size), min(artwork.size)),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        output = io.BytesIO()
        square.save(output, format="PNG")
        return output.getvalue()


def crop_center_square(content: bytes) -> bytes:
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGBA")
        side = min(image.size)
        square = ImageOps.fit(
            image,
            (side, side),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        output = io.BytesIO()
        square.save(output, format="PNG")
        return output.getvalue()


def convert_to_png(content: bytes) -> bytes:
    with Image.open(io.BytesIO(content)) as image:
        output = io.BytesIO()
        image.convert("RGBA").save(output, format="PNG")
        return output.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_image(content: bytes) -> tuple[str, int, int, str]:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            image_format = (image.format or "").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image: {exc}") from exc
    if width < 64 or height < 64:
        raise ValueError(f"image is too small: {width}x{height}")
    return image_format, width, height, image.mode


def extension_for(content_type: str, url: str) -> str:
    match = re.search(r"\.(png|jpe?g|webp)(?:$|[?#])", url.lower())
    if match:
        return "." + match.group(1).replace("jpeg", "jpg")
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }.get(content_type.split(";")[0].lower(), ".jpg")


def filename(candidate: Candidate, digest: str, extension: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", candidate.form_name).strip("_")
    return f"{candidate.source}_{safe or 'image'}_{digest[:12]}{extension}"


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path", "national_dex_id", "species_name", "form_name", "source",
        "source_url", "license_name", "license_url", "sha256", "width", "height",
        "mode", "title",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def show_progress(current: int, total: int, species: Species) -> None:
    width = 30
    completed = int(width * current / total) if total else width
    bar = "#" * completed + "-" * (width - completed)
    message = (
        f"\rProgress [{bar}] {current:4d}/{total:4d} "
        f"({current / total:6.2%}) {species.dex_id:04d} {species.name:<16}"
    )
    print(message, end="", file=sys.stderr, flush=True)
    if current == total:
        print(file=sys.stderr)


def scan_existing(root: Path) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    hashes: set[str] = set()
    for path in root.glob("*_*/*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = re.match(r"^(\d{4})_(.+)$", path.parent.name)
        if not match:
            continue
        try:
            content = path.read_bytes()
            _, width, height, mode = validate_image(content)
        except (OSError, ValueError) as exc:
            LOG.warning("Skipping existing file %s: %s", path, exc)
            continue
        digest = sha256(content)
        hashes.add(digest)
        rows.append({
            "image_path": str(path.relative_to(root)),
            "national_dex_id": int(match.group(1)),
            "species_name": match.group(2),
            "form_name": "existing",
            "source": "existing",
            "source_url": "",
            "license_name": "",
            "license_url": "",
            "sha256": digest,
            "width": width,
            "height": height,
            "mode": mode,
            "title": path.name,
        })
    return rows, hashes


def download_candidate(
    session: requests.Session,
    root: Path,
    species: Species,
    candidate: Candidate,
    seen_hashes: set[str],
) -> Optional[dict]:
    try:
        if candidate.content is not None:
            content = candidate.content
            content_type = ""
        else:
            response = session.get(candidate.url, timeout=45)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get("content-type", "")
        if candidate.source == "pkmncards-tcg":
            content = crop_card_artwork(content)
        elif candidate.source == "brave-images":
            content = crop_center_square(content)
        else:
            content = convert_to_png(content)
        _, width, height, mode = validate_image(content)
    except (OSError, requests.RequestException, ValueError) as exc:
        LOG.debug("Skipping %s: %s", candidate.url, exc)
        return None
    digest = sha256(content)
    if digest in seen_hashes:
        return None
    extension = ".png"
    directory = root / species.directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename(candidate, digest, extension)
    path.write_bytes(content)
    seen_hashes.add(digest)
    return {
        "image_path": str(path.relative_to(root)),
        "national_dex_id": species.dex_id,
        "species_name": species.name,
        "form_name": candidate.form_name,
        "source": candidate.source,
        "source_url": candidate.source_url,
        "license_name": candidate.license_name,
        "license_url": candidate.license_url,
        "sha256": digest,
        "width": width,
        "height": height,
        "mode": mode,
        "title": candidate.title,
    }


def build_dataset(args: argparse.Namespace) -> None:
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / args.manifest
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows, seen_hashes = scan_existing(root)
    brave_api_key = ""
    if "brave-images" in args.sources:
        key_path = Path(args.brave_api_key_file)
        if not key_path.is_absolute():
            key_path = Path(__file__).resolve().parent / key_path
        try:
            brave_api_key = key_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"Unable to read Brave API key file {key_path}: {exc}"
            ) from exc
        if not brave_api_key:
            raise RuntimeError(f"Brave API key file is empty: {key_path}")
    species_list = load_species(session, args.first, args.last)
    providers = {
        "pokeapi": lambda species: pokeapi_candidates(session, species),
        "pokemondb": lambda species: pokemondb_candidates(session, species),
        "pkmncards": lambda species: pkmncards_candidates(
            session, species, args.search_limit
        ),
        "brave-images": lambda species: brave_image_candidates(
            session, species, args.search_limit, brave_api_key
        ),
    }
    if len(args.images_per_source) == 1:
        images_per_source = dict.fromkeys(args.sources, args.images_per_source[0])
    elif len(args.images_per_source) == len(args.sources):
        images_per_source = dict(zip(args.sources, args.images_per_source))
    else:
        raise ValueError(
            "--images-per-source accepts one integer or exactly one integer "
            "per selected source."
        )
    total_species = len(species_list)
    for species_index, species in enumerate(species_list, start=1):
        show_progress(species_index, total_species, species)
        LOG.info("Collecting %04d %s", species.dex_id, species.name)
        for provider_name in args.sources:
            try:
                candidates = providers[provider_name](species)
                count = 0
                for candidate in candidates:
                    row = download_candidate(session, root, species, candidate, seen_hashes)
                    if row:
                        rows.append(row)
                        count += 1
                    if count >= images_per_source[provider_name]:
                        break
                LOG.info("  %s: %d new images", provider_name, count)
            except BraveRateLimitError as exc:
                LOG.error(
                    "%s rate limit persisted for %s; stopping the run: %s",
                    provider_name,
                    species.name,
                    exc,
                )
                rows.sort(
                    key=lambda row: (
                        int(row["national_dex_id"]),
                        row["image_path"],
                    )
                )
                write_manifest(manifest_path, rows)
                labels = [
                    {
                        "index": item.dex_id - args.first,
                        "national_dex_id": item.dex_id,
                        "name": item.name,
                    }
                    for item in species_list
                ]
                (root / "labels.json").write_text(
                    json.dumps(labels, indent=2),
                    encoding="utf-8",
                )
                LOG.error(
                    "Partial dataset saved. Re-run with --first %d to resume "
                    "at the missing species (%04d %s).",
                    species.dex_id,
                    species.dex_id,
                    species.name,
                )
                raise SystemExit(2)
            except (requests.RequestException, RuntimeError) as exc:
                LOG.error("%s failed for %s: %s", provider_name, species.name, exc)
        if species_index < total_species:
            species_delay = (
                max(args.delay, BRAVE_SPECIES_DELAY)
                if "brave-images" in args.sources
                else args.delay
            )
            time.sleep(species_delay)
    rows.sort(key=lambda row: (int(row["national_dex_id"]), row["image_path"]))
    write_manifest(manifest_path, rows)
    labels = [
        {"index": species.dex_id - args.first, "national_dex_id": species.dex_id, "name": species.name}
        for species in species_list
    ]
    (root / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    LOG.info("Wrote %d rows to %s", len(rows), manifest_path)


def brave_image_candidates(
    session: requests.Session, species: Species, limit: int, api_key: str
) -> Iterable[Candidate]:
    """Retrieve image URLs from Brave's authenticated image-search API."""
    query = f'"{species.name}" plush'
    search_url = BRAVE_IMAGES_API
    response = None
    for attempt in range(3):
        response = session.get(
            BRAVE_IMAGES_API,
            params={
                "q": query,
                "count": min(limit, 200),
                "safesearch": "strict",
            },
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            timeout=30,
        )
        if response.status_code != 429:
            break
        if attempt == 2:
            break
        LOG.warning(
            "Brave rate limit for %s (attempt %d/3); waiting %.0f seconds before retry",
            species.name,
            attempt + 1,
            BRAVE_RATE_LIMIT_DELAY,
        )
        time.sleep(BRAVE_RATE_LIMIT_DELAY)
    if response is None:
        raise BraveRateLimitError("Brave did not return a response.")
    if response.status_code == 429:
        raise BraveRateLimitError(
            "Brave continued returning HTTP 429 after 3 attempts and two "
            "one-minute waits. "
            "Resume later or reduce request frequency."
        )
    response.raise_for_status()
    data = response.json()
    seen: set[str] = set()
    yielded = 0
    for item in data.get("results", []):
        properties = item.get("properties", {})
        image_url = properties.get("url", "") if isinstance(properties, dict) else ""
        source_url = item.get("url", search_url)
        if image_url in seen or "search.brave.com" in image_url:
            continue
        seen.add(image_url)
        yield Candidate(
            url=image_url,
            source="brave-images",
            source_url=source_url,
            form_name="plush-search",
            license_name="Search result image; verify usage rights before redistribution",
            title=item.get("title", query),
        )
        yielded += 1
        if yielded >= limit:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="pokedex_data")
    parser.add_argument("--manifest", default="dataset_manifest.csv")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=1025)
    parser.add_argument(
        "--sources", nargs="+",
        choices=(
            "pokeapi", "pokemondb", "pkmncards", "brave-images",
        ),
        default=("pokeapi", "pokemondb"),
    )
    parser.add_argument(
        "--images-per-source",
        nargs="+",
        type=int,
        default=[20],
        help="One limit shared by all sources, or one limit per selected source.",
    )
    parser.add_argument("--search-limit", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument(
        "--brave-api-key-file",
        default=DEFAULT_BRAVE_API_KEY_FILE,
        help="Text file containing the Brave Search API key.",
    )
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    if len(arguments.images_per_source) not in (1, len(arguments.sources)):
        parser.error(
            "--images-per-source accepts one integer or exactly one integer "
            "per selected source."
        )
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    build_dataset(arguments)
