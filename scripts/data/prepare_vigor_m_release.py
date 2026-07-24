#!/usr/bin/env python3
"""Stage a self-contained, portable VIGOR-M release for GeoMoE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


CITIES = ("Chicago", "NewYork", "SanFrancisco", "Seattle")
EXPECTED_SPLIT_COUNTS = {
    "same_area_train.csv": 37_895,
    "same_area_test.csv": 37_789,
    "cross_area_train.csv": 42_595,
    "cross_area_test.csv": 33_089,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="VIGOR-M asset root containing Pano/ and level/ (or canonical names).",
    )
    parser.add_argument(
        "--metadata-source",
        type=Path,
        required=True,
        help="Frozen level_pano metadata used for the reported GeoMoE results.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="Use symlink for local validation, hardlink for local staging, or copy for upload.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Permit metadata that does not match the frozen GeoMoE split counts.",
    )
    return parser.parse_args()


def resolve_asset_dir(root: Path, canonical: str, legacy: str) -> Path:
    for candidate in (root / canonical, root / legacy):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Missing {canonical} assets. Expected {root / canonical} or {root / legacy}."
    )


def resolve_bounds_path(root: Path) -> Path:
    candidates = (
        root / "metadata" / "city_bounds.csv",
        root / "figures" / "pano_distribution" / "pano_distribution_summary.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Missing city bounds. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stage_asset_tree(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=True)
        return
    copy_function = os.link if mode == "hardlink" else shutil.copy2
    shutil.copytree(source, destination, copy_function=copy_function)


def portable_ground_path(row: dict[str, str]) -> str:
    value = row.get("ground_path", "").strip()
    if not value:
        return value
    city = row.get("city", "").strip()
    if not city:
        raise ValueError(f"ground_path row has no city: {row}")
    return (Path("panoramas") / city / Path(value).name).as_posix()


def rewrite_metadata(source: Path, destination: Path) -> dict[str, dict[str, str]]:
    csv_paths = sorted(source.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV metadata found under {source}")

    hashes: dict[str, dict[str, str]] = {}
    for source_csv in csv_paths:
        relative = source_csv.relative_to(source)
        destination_csv = destination / relative
        destination_csv.parent.mkdir(parents=True, exist_ok=True)

        with source_csv.open(newline="", encoding="utf-8") as source_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise ValueError(f"Missing CSV header: {source_csv}")
            rows = list(reader)

        if "ground_path" in reader.fieldnames:
            for row in rows:
                row["ground_path"] = portable_ground_path(row)

        with destination_csv.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        hashes[relative.as_posix()] = {
            "source_sha256": file_sha256(source_csv),
            "release_sha256": file_sha256(destination_csv),
        }
    return hashes


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_release(root: Path, allow_count_mismatch: bool) -> dict[str, object]:
    metadata_root = root / "metadata"
    split_counts = {
        name: len(read_rows(metadata_root / name))
        for name in EXPECTED_SPLIT_COUNTS
    }
    if not allow_count_mismatch:
        mismatches = {
            name: (split_counts[name], expected)
            for name, expected in EXPECTED_SPLIT_COUNTS.items()
            if split_counts[name] != expected
        }
        if mismatches:
            raise RuntimeError(f"Frozen split count mismatch: {mismatches}")

    all_rows = read_rows(metadata_root / "all.csv")
    missing: list[str] = []
    absolute_paths: list[str] = []
    for row in all_rows:
        ground_path = Path(row["ground_path"])
        if ground_path.is_absolute():
            absolute_paths.append(str(ground_path))
        elif not (root / ground_path).is_file():
            missing.append(ground_path.as_posix())

        for level in range(4):
            tile_id = row[f"L{level}"]
            city = row["city"]
            tile_path = root / "satellite" / city / f"L{level}" / f"{tile_id}.png"
            if not tile_path.is_file():
                missing.append(tile_path.relative_to(root).as_posix())

        if len(missing) >= 20:
            break

    if absolute_paths:
        raise RuntimeError(f"Release metadata contains absolute paths: {absolute_paths[:3]}")
    if missing:
        raise FileNotFoundError(f"Release references missing files: {missing[:20]}")

    panorama_counts = {
        city: sum(
            path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            for path in (root / "panoramas" / city).iterdir()
        )
        for city in CITIES
    }
    satellite_counts = {
        city: {
            f"L{level}": sum(
                path.is_file()
                for path in (root / "satellite" / city / f"L{level}").iterdir()
            )
            for level in range(4)
        }
        for city in CITIES
    }
    return {
        "all_queries": len(all_rows),
        "split_queries": split_counts,
        "panoramas": panorama_counts,
        "satellite_tiles": satellite_counts,
    }


def write_release_readme(path: Path) -> None:
    path.write_text(
        """# VIGOR-M Release Layout

This directory is the portable VIGOR-M layout consumed by GeoMoE. The files in
`metadata/` contain the frozen train/test membership used for the reported
results; do not regenerate those splits from local paths.

```text
VIGOR-M/
|-- panoramas/<city>/*.jpg
|-- satellite/<city>/L0..L3/*.png
|-- metadata/
|   |-- city_bounds.csv
|   |-- same_area_train.csv
|   |-- same_area_test.csv
|   |-- cross_area_train.csv
|   `-- cross_area_test.csv
`-- manifest.json
```

Use the extracted directory directly:

```bash
python scripts/train/train_vigor_m.py --data-folder /path/to/VIGOR-M
```

The GeoMoE code license does not grant rights to the dataset imagery. Retain the
dataset provider's provenance, attribution, and redistribution terms when
publishing or using this bundle.
""",
        encoding="utf-8",
    )


def build_release(args: argparse.Namespace) -> Path:
    source_root = args.source_root.expanduser().resolve()
    metadata_source = args.metadata_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    if not metadata_source.is_dir():
        raise FileNotFoundError(metadata_source)

    panorama_source = resolve_asset_dir(source_root, "panoramas", "Pano")
    satellite_source = resolve_asset_dir(source_root, "satellite", "level")
    bounds_source = resolve_bounds_path(source_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        stage_asset_tree(panorama_source, temporary / "panoramas", args.mode)
        stage_asset_tree(satellite_source, temporary / "satellite", args.mode)
        metadata_output = temporary / "metadata"
        metadata_hashes = rewrite_metadata(metadata_source, metadata_output)
        shutil.copy2(bounds_source, metadata_output / "city_bounds.csv")
        counts = validate_release(temporary, args.allow_count_mismatch)
        write_release_readme(temporary / "README.md")
        manifest = {
            "schema": "vigor-m-geomoe-release-v1",
            "asset_mode": args.mode,
            "layout": {
                "panoramas": "panoramas/<city>/*.jpg",
                "satellite": "satellite/<city>/L0..L3/*.png",
                "metadata": "metadata/*.csv",
            },
            "counts": counts,
            "metadata_hashes": metadata_hashes,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return output


def main() -> int:
    args = parse_args()
    output = build_release(args)
    print(f"VIGOR-M release ready: {output}")
    print(f"Asset mode: {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
