#!/usr/bin/env python3
"""Build VIGOR-M metadata for the updated Pano street-view images."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "geomoe").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


CITIES = ("Chicago", "NewYork", "SanFrancisco", "Seattle")
DEFAULT_ROOT = Path("data/VIGOR-M")
SOURCE_RE = re.compile(
    r"^(?P<lat>[+-]?\d+(?:\.\d+)?),\s*"
    r"(?P<lon>[+-]?\d+(?:\.\d+)?)_"
    r"(?P<year>\d{4})_(?P<month>\d{2})_"
    r"(?P<panoid>.+)\.(?P<ext>jpe?g)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Grid:
    city: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lat: float, lon: float) -> bool:
        eps = 1e-10
        return (
            self.min_lat - eps <= lat <= self.max_lat + eps
            and self.min_lon - eps <= lon <= self.max_lon + eps
        )

    def tile_id(self, level: int, lat: float, lon: float) -> str:
        n = 4**level
        tile_w = (self.max_lon - self.min_lon) / n
        tile_h = (self.max_lat - self.min_lat) / n
        col = int((lon - self.min_lon) / tile_w)
        row = int((self.max_lat - lat) / tile_h)
        col = max(0, min(n - 1, col))
        row = max(0, min(n - 1, row))
        return f"{self.city}_L{level}_r{row:02d}_c{col:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("data/VIGOR-M/meta/level_pano"))
    parser.add_argument("--same-area-test-ratio", type=float, default=0.5)
    parser.add_argument("--cross-train-cities", nargs="+", default=["NewYork", "Seattle"])
    return parser.parse_args()


def load_grids(root: Path) -> dict[str, Grid]:
    summary_path = root / "figures" / "pano_distribution" / "pano_distribution_summary.csv"
    grids: dict[str, Grid] = {}
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            city = row["city"]
            if city not in CITIES:
                continue
            grids[city] = Grid(
                city=city,
                min_lon=float(row["l0_min_lon"]),
                min_lat=float(row["l0_min_lat"]),
                max_lon=float(row["l0_max_lon"]),
                max_lat=float(row["l0_max_lat"]),
            )
    missing = sorted(set(CITIES) - set(grids))
    if missing:
        raise RuntimeError(f"Missing grid bounds for: {missing}")
    return grids


def hash_score(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def parse_pano(path: Path) -> tuple[float, float, str] | None:
    match = SOURCE_RE.match(path.name)
    if match is None:
        return None
    lat = float(match.group("lat"))
    lon = float(match.group("lon"))
    panoid = match.group("panoid")
    return lat, lon, panoid


def build_rows(root: Path, grids: dict[str, Grid]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []
    pano_root = root / "Pano"

    for city in CITIES:
        grid = grids[city]
        city_dir = pano_root / city
        if not city_dir.exists():
            raise FileNotFoundError(city_dir)

        for path in sorted(city_dir.glob("*.jpg")):
            parsed = parse_pano(path)
            if parsed is None:
                unmatched.append({"city": city, "ground_path": str(path), "reason": "bad_pano_name"})
                continue
            lat, lon, panoid = parsed
            if not grid.contains(lat, lon):
                unmatched.append({"city": city, "ground_path": str(path), "reason": "outside_l0_bounds"})
                continue

            ground = f"{panoid},{lat:.8f},{lon:.8f},.jpg"
            row = {
                "city": city,
                "ground": ground,
                "ground_path": str(path),
                "lat": f"{lat:.8f}",
                "lon": f"{lon:.8f}",
            }
            complete = True
            for level in range(4):
                tile_id = grid.tile_id(level, lat, lon)
                tile_path = root / "level" / city / f"L{level}" / f"{tile_id}.png"
                row[f"L{level}"] = tile_id
                complete = complete and tile_path.exists()

            if complete:
                rows.append(row)
            else:
                unmatched.append({"city": city, "ground_path": str(path), "reason": "missing_tile_file"})

    rows.sort(key=lambda r: (r["city"], r["ground_path"]))
    return rows, unmatched


def split_rows(
    rows: list[dict[str, str]],
    same_area_test_ratio: float,
    cross_train_cities: set[str],
) -> dict[str, list[dict[str, str]]]:
    splits = {
        "same_area_train": [],
        "same_area_test": [],
        "cross_area_train": [],
        "cross_area_test": [],
    }
    for row in rows:
        key = f"{row['city']}/{row['ground_path']}"
        same_split = "same_area_test" if hash_score(key) < same_area_test_ratio else "same_area_train"
        splits[same_split].append(row)

        cross_split = "cross_area_train" if row["city"] in cross_train_cities else "cross_area_test"
        splits[cross_split].append(row)

    for split_rows_ in splits.values():
        split_rows_.sort(key=lambda r: (r["city"], r["ground_path"]))
    return splits


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["city", "ground", "ground_path", "lat", "lon", "L0", "L1", "L2", "L3"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"write: {path} ({len(rows)} rows)")


def write_unmatched(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["city", "ground_path", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"write: {path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not 0 < args.same_area_test_ratio < 1:
        raise ValueError("--same-area-test-ratio must be between 0 and 1")

    grids = load_grids(root)
    rows, unmatched = build_rows(root, grids)
    splits = split_rows(rows, args.same_area_test_ratio, set(args.cross_train_cities))

    write_rows(output / "all.csv", rows)
    write_unmatched(output / "unmatched_pano.csv", unmatched)

    same_train_ids = {id(row) for row in splits["same_area_train"]}
    same_test_ids = {id(row) for row in splits["same_area_test"]}

    for city in CITIES:
        city_rows = [row for row in rows if row["city"] == city]
        write_rows(output / city / f"{city}_all.csv", city_rows)
        write_rows(output / city / f"{city}_train.csv", [row for row in city_rows if id(row) in same_train_ids])
        write_rows(output / city / f"{city}_test.csv", [row for row in city_rows if id(row) in same_test_ids])

    for name, split_rows_ in splits.items():
        write_rows(output / f"{name}.csv", split_rows_)

    print("summary:")
    print(f"  matched: {len(rows)}")
    print(f"  unmatched: {len(unmatched)}")
    for name, split_rows_ in splits.items():
        print(f"  {name}: {len(split_rows_)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
