import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "geomoe").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import time
from pathlib import Path

import cv2

from geomoe.datasets.justzoomin import (
    JustZoomInDatasetEval,
    default_satellite_cache_dir,
    resolve_justzoomin_level,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-generate JustZoomIn satellite reference crops."
    )
    parser.add_argument("--data-folder", default="data/justzoomin")
    parser.add_argument("--levels", nargs="+", default=["L1", "L2"], choices=("L1", "L2", "L3", "L4"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=("train", "val"))
    parser.add_argument("--satellite-zoom", type=int, default=-3)
    parser.add_argument(
        "--satellite-stride-fraction",
        type=float,
        default=None,
        help="Use the same dense reference stride as training, e.g. 0.25.",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_dataset(args, level, split):
    level_cfg = resolve_justzoomin_level(level)
    cache_dir = Path(args.cache_dir) if args.cache_dir else default_satellite_cache_dir(args.data_folder)
    return JustZoomInDatasetEval(
        data_folder=args.data_folder,
        split=split,
        img_type="reference",
        sequence_depth=level_cfg["sequence_depth"],
        satellite_zoom=args.satellite_zoom,
        satellite_crop_meters=level_cfg["satellite_crop_meters"],
        satellite_stride_fraction=args.satellite_stride_fraction,
        satellite_cache_dir=cache_dir,
        satellite_cache_size=args.image_size,
        transforms=None,
    )


def write_cache(args, level):
    datasets = [build_dataset(args, level, split) for split in args.splits]
    ds = datasets[0]
    cache_path = ds.satellite_cache_path
    cache_path.mkdir(parents=True, exist_ok=True)

    meta_path = cache_path / "metadata.json"
    metadata = {
        "level": level,
        "split_source": args.splits,
        "sequence_depth": ds.sequence_depth,
        "satellite_zoom": ds.satellite_zoom,
        "satellite_crop_meters": ds.satellite_crop_meters,
        "satellite_stride_fraction": ds.satellite_stride_fraction,
        "image_size": args.image_size,
        "num_reference_crops": len({tile for dataset in datasets for tile in dataset.idx2tile.values()}),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    written = 0
    skipped = 0
    start = time.perf_counter()

    seen_tile_ids = set()
    for dataset in datasets:
        for idx in sorted(dataset.idx2tile):
            tile_id = dataset.idx2tile[idx]
            if tile_id in seen_tile_ids:
                continue
            seen_tile_ids.add(tile_id)

            out_path = dataset.idx2satellite_cache_path[idx]
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            image = dataset._load_satellite_crop(idx)
            image = cv2.resize(
                image,
                (args.image_size, args.image_size),
                interpolation=cv2.INTER_LINEAR_EXACT,
            )
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(out_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError(f"Failed to write {out_path}")
            written += 1

            if written % 50 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"{level}: written={written} skipped={skipped} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.perf_counter() - start
    print(
        f"{level}: cache={cache_path} written={written} skipped={skipped} "
        f"total={len(seen_tile_ids)} elapsed={elapsed:.1f}s"
    )


def main():
    args = parse_args()
    for level in args.levels:
        write_cache(args, level)


if __name__ == "__main__":
    main()
