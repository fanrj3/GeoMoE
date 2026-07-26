#!/usr/bin/env python3
"""Final VIGOR-M hierarchical ablation evaluation on the updated pano split."""

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "geomoe").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import gc
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geomoe.datasets.vigor_m import (
    VIGOR_M_CITIES,
    VigorMDatasetEval,
    _vigor_m_bounds_path,
    _vigor_m_city_csv_split,
    _vigor_m_split_csv,
)
from geomoe.model import TimmModel
from geomoe.trainer import predict
from geomoe.transforms import get_transforms_val
from geomoe.utils import setup_system


@dataclass
class PredictConfig:
    """Minimal configuration consumed by the shared feature extractor."""

    device: str
    verbose: bool = True
    normalize_features: bool = True


class MetricAccumulator:
    """Accumulate retrieval ranks and geodesic errors without storing scores."""

    def __init__(self, name, num_candidates, center_lat, center_lon):
        """Initialize counters for one scoring method."""
        self.name = name
        self.ranks = [1, 5, 10]
        self.top1_percent_k = max(1, int(num_candidates) // 100)
        self.thresholds = [40, 50, 100, 200, 300]
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.total = 0
        self.rank_hits = {rank: 0 for rank in self.ranks}
        self.rtop1_hits = 0
        self.distance_hits = {threshold: 0 for threshold in self.thresholds}
        self.distances = []

    def update(self, scores, gt_indices, query_lat, query_lon):
        """Add one score batch and its ground-truth query coordinates."""
        if scores.ndim != 2:
            raise ValueError(f"scores must be 2-D, got shape={tuple(scores.shape)}")

        batch_size = scores.shape[0]
        rows = torch.arange(batch_size, device=scores.device)
        gt_scores = scores[rows, gt_indices]
        valid_gt = torch.isfinite(gt_scores)
        rankings = (scores > gt_scores.unsqueeze(1)).sum(dim=1)

        for rank in self.ranks:
            hits = valid_gt & (rankings < rank)
            self.rank_hits[rank] += int(hits.sum().item())

        top1_hits = valid_gt & (rankings < self.top1_percent_k)
        self.rtop1_hits += int(top1_hits.sum().item())

        top1_indices = scores.argmax(dim=1).detach().cpu().numpy()
        pred_lat = self.center_lat[top1_indices]
        pred_lon = self.center_lon[top1_indices]
        distances = haversine_np(query_lat, query_lon, pred_lat, pred_lon)

        self.total += int(batch_size)
        self.distances.append(distances)
        for threshold in self.thresholds:
            self.distance_hits[threshold] += int((distances <= threshold).sum())

    def finalize(self):
        """Convert accumulated counters into report-ready percentages."""
        if self.total == 0:
            raise RuntimeError(f"No samples accumulated for {self.name}")

        distances = np.concatenate(self.distances, axis=0)
        result = {
            "method": self.name,
            "queries": self.total,
            "R1": percent(self.rank_hits[1], self.total),
            "R5": percent(self.rank_hits[5], self.total),
            "R10": percent(self.rank_hits[10], self.total),
            "Rtop1": percent(self.rtop1_hits, self.total),
            "Rtop1_k": self.top1_percent_k,
            **{
                f"R@{threshold}m": percent(self.distance_hits[threshold], self.total)
                for threshold in self.thresholds
            },
            "mean_m": float(np.mean(distances)),
            "median_m": float(np.median(distances)),
        }
        return result


def percent(value, total):
    """Express a count as a percentage of the total."""
    return float(value) / float(total) * 100.0


def parse_args():
    """Parse the legacy hierarchical-ablation command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate VIGOR-M L1/L2/L3 hierarchical ablations."
    )
    parser.add_argument("--data-folder", default="./data/VIGOR-M")
    parser.add_argument("--metadata-folder", default="./data/VIGOR-M/metadata")
    parser.add_argument("--model-root", default="./outputs/checkpoints/vigor_m")
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--same-area", action="store_true", dest="same_area", default=True)
    parser.add_argument("--cross-area", action="store_false", dest="same_area")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help="Optional existing feature_cache directory to load when --reuse-cache is set.",
    )
    parser.add_argument(
        "--fusion-normalization",
        choices=["raw", "minmax"],
        default="minmax",
        help="Normalize each level score per query before multiplicative fusion.",
    )
    parser.add_argument(
        "--checkpoint-choice",
        choices=["end", "best"],
        default=os.environ.get("VIGOR_M_CHECKPOINT_CHOICE", "end"),
        help="Use weights_end.pth or the best validation weights_e*.pth in each run dir.",
    )
    parser.add_argument("--l1-checkpoint", default=os.environ.get("VIGOR_M_L1_CHECKPOINT"))
    parser.add_argument("--l2-checkpoint", default=os.environ.get("VIGOR_M_L2_CHECKPOINT"))
    parser.add_argument("--l3-checkpoint", default=os.environ.get("VIGOR_M_L3_CHECKPOINT"))
    parser.add_argument("--l1-run-dir", default=os.environ.get("VIGOR_M_L1_RUN_DIR"))
    parser.add_argument("--l2-run-dir", default=os.environ.get("VIGOR_M_L2_RUN_DIR"))
    parser.add_argument("--l3-run-dir", default=os.environ.get("VIGOR_M_L3_RUN_DIR"))
    parser.add_argument(
        "--l1-satellite-stride-fraction",
        type=float,
        default=None,
        help="Use dense-stride L1 satellite references, e.g. 0.25 for dense L1.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_path(path):
    """Expand an optional user path to an absolute path."""
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def latest_run_dir(model_root, level, model_name):
    """Find the most recently modified run directory for one level."""
    root = Path(model_root) / level / model_name
    if not root.exists():
        raise FileNotFoundError(f"Missing model directory: {root}")
    run_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {root}")
    return max(run_dirs, key=lambda path: (path.stat().st_mtime, path.name))


def best_checkpoint(run_dir):
    """Select the checkpoint with the highest score encoded in its filename."""
    pattern = re.compile(r"weights_e(\d+)_([0-9.]+)\.pth$")
    candidates = []
    for path in Path(run_dir).glob("weights_e*.pth"):
        match = pattern.match(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        score = float(match.group(2))
        candidates.append((score, epoch, path))
    if not candidates:
        raise FileNotFoundError(f"No weights_e*.pth checkpoints found in {run_dir}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def checkpoint_for_level(args, level):
    """Resolve explicit, final, or best weights for a hierarchy level."""
    explicit = getattr(args, f"{level.lower()}_checkpoint")
    if explicit:
        checkpoint = resolve_path(explicit)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        return checkpoint

    run_dir_arg = getattr(args, f"{level.lower()}_run_dir")
    run_dir = resolve_path(run_dir_arg) if run_dir_arg else latest_run_dir(
        args.model_root, level, args.model
    )
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    end_path = run_dir / "weights_end.pth"
    if args.checkpoint_choice == "end":
        if end_path.exists():
            return end_path
        return best_checkpoint(run_dir)

    return best_checkpoint(run_dir)


def load_model(args, checkpoint_path):
    """Build the single-level encoder and load a release checkpoint."""
    model = TimmModel(args.model, pretrained=False, img_size=args.img_size)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"  Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
    model = model.to(args.device)
    return model


def build_transforms(args):
    """Build validation transforms from the backbone's normalization metadata."""
    model = TimmModel(args.model, pretrained=False, img_size=args.img_size)
    data_config = model.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    del model
    gc.collect()

    image_size_sat = (args.img_size, args.img_size)
    ground_width = args.img_size * 2
    ground_height = int((1024 / 2048) * ground_width)
    image_size_ground = (ground_height, ground_width)
    return get_transforms_val(
        image_size_sat,
        image_size_ground,
        mean=mean,
        std=std,
        ground_cutting=0,
    )


def extract_level_features(args, level, checkpoint, sat_transform, ground_transform):
    """Extract or reuse aligned query/reference features for one level."""
    cache_dir = (
        Path(args.feature_cache_dir)
        if args.feature_cache_dir is not None
        else Path(args.output_dir) / "feature_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{level}_features.pt"

    if args.reuse_cache and cache_path.exists():
        print(f"\n[{level}] Loading feature cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"\n[{level}] Extracting features")
    print(f"  Checkpoint: {checkpoint}")
    model = load_model(args, checkpoint)
    pred_config = PredictConfig(device=args.device, verbose=args.verbose)

    satellite_stride_fraction = (
        args.l1_satellite_stride_fraction if level == "L1" else None
    )

    ref_dataset = VigorMDatasetEval(
        data_folder=args.data_folder,
        split="test",
        img_type="reference",
        same_area=args.same_area,
        data_level=level,
        transforms=sat_transform,
        metadata_folder=args.metadata_folder,
        satellite_stride_fraction=satellite_stride_fraction,
    )
    query_dataset = VigorMDatasetEval(
        data_folder=args.data_folder,
        split="test",
        img_type="query",
        same_area=args.same_area,
        data_level=level,
        transforms=ground_transform,
        metadata_folder=args.metadata_folder,
        satellite_stride_fraction=satellite_stride_fraction,
    )

    pin_memory = args.device.startswith("cuda")
    ref_loader = DataLoader(
        ref_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    ref_features, ref_labels = predict(pred_config, model, ref_loader)
    query_features, query_labels = predict(pred_config, model, query_loader)

    payload = {
        "level": level,
        "checkpoint": str(checkpoint),
        "tile_list": list(ref_dataset.tile_list),
        "query_images": list(query_dataset.images),
        "ref_features": ref_features.detach().cpu(),
        "ref_labels": ref_labels.detach().cpu(),
        "query_features": query_features.detach().cpu(),
        "query_labels": query_labels.detach().cpu(),
    }
    torch.save(payload, cache_path)
    print(f"  Saved cache: {cache_path}")

    del model, ref_features, ref_labels, query_features, query_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


def normalize_scores(scores, mode):
    """Normalize each query's scores before cross-level multiplication."""
    if mode == "raw":
        return scores
    if mode == "minmax":
        score_min = scores.min(dim=1, keepdim=True).values
        score_max = scores.max(dim=1, keepdim=True).values
        return (scores - score_min) / (score_max - score_min).clamp(min=1e-8)
    raise ValueError(f"Unknown score normalization: {mode}")


def parse_tile(tile_id):
    """Parse regular and dense-stride VIGOR-M tile identifiers."""
    parts = tile_id.split("_")
    if len(parts) >= 5 and parts[2].startswith("s"):
        city = parts[0]
        level = parts[1]
        row = int(parts[3][1:])
        col = int(parts[4][1:])
        stride = float(parts[2][1:])
        return city, level, row, col, stride
    if len(parts) < 4:
        raise ValueError(f"Unexpected tile id: {tile_id}")
    city = parts[0]
    level = parts[1]
    row = int(parts[2][1:])
    col = int(parts[3][1:])
    return city, level, row, col, None


def parent_tile(tile_id, target_level):
    """Map a regular tile identifier to an ancestor level."""
    city, level, row, col, _stride = parse_tile(tile_id)
    source_depth = int(level[1:])
    target_depth = int(target_level[1:])
    if target_depth > source_depth:
        raise ValueError(f"{target_level} is not a parent of {tile_id}")
    divisor = 4 ** (source_depth - target_depth)
    return f"{city}_{target_level}_r{row // divisor:02d}_c{col // divisor:02d}"


def dense_l1_parent_tile(tile_id, stride_fraction):
    """Map a child tile center to its nearest dense L1 reference."""
    city, level, row, col, _stride = parse_tile(tile_id)
    source_depth = int(level[1:])
    axis_scale = 4 ** (source_depth - 1)
    row_pos = (row + 0.5) / axis_scale
    col_pos = (col + 0.5) / axis_scale
    dense_axis = int(round(4.0 / stride_fraction))
    dense_row = int(round(row_pos / stride_fraction - 0.5))
    dense_col = int(round(col_pos / stride_fraction - 0.5))
    dense_row = max(0, min(dense_axis - 1, dense_row))
    dense_col = max(0, min(dense_axis - 1, dense_col))
    return f"{city}_L1_s{stride_fraction:g}_r{dense_row:02d}_c{dense_col:02d}"


def build_parent_indices(child_tiles, parent_tiles, parent_level):
    """Return the parent-reference index for every child tile."""
    parent_to_idx = {tile: idx for idx, tile in enumerate(parent_tiles)}
    dense_l1_stride = None
    if parent_level == "L1":
        for tile in parent_tiles:
            _city, _level, _row, _col, stride = parse_tile(tile)
            if stride is not None:
                dense_l1_stride = stride
                break

    indices = []
    missing = []
    for tile in child_tiles:
        if dense_l1_stride is not None:
            parent = dense_l1_parent_tile(tile, dense_l1_stride)
        else:
            parent = parent_tile(tile, parent_level)
        if parent not in parent_to_idx:
            missing.append((tile, parent))
            continue
        indices.append(parent_to_idx[parent])
    if missing:
        examples = ", ".join(f"{child}->{parent}" for child, parent in missing[:5])
        raise KeyError(f"Missing {parent_level} parents for {len(missing)} tiles: {examples}")
    return torch.tensor(indices, dtype=torch.long)


def load_query_rows(data_folder, metadata_folder, split="test", same_area=True):
    """Load valid geotagged panorama rows in dataset query order."""
    rows = []
    cities = VIGOR_M_CITIES if same_area else ["Chicago", "SanFrancisco"]
    for city in cities:
        csv_split = _vigor_m_city_csv_split(same_area, split)
        csv_path = _vigor_m_split_csv(data_folder, city, csv_split, metadata_folder)
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if pd.notna(row["L3"]) and row["L3"]:
                rows.append(row.to_dict())
    return rows


def load_city_bounds(data_folder):
    """Load each city's L0 geographic extent."""
    summary_path = _vigor_m_bounds_path(data_folder)
    df = pd.read_csv(summary_path)
    bounds = {}
    for _, row in df.iterrows():
        bounds[row["city"]] = (
            float(row["l0_min_lon"]),
            float(row["l0_min_lat"]),
            float(row["l0_max_lon"]),
            float(row["l0_max_lat"]),
        )
    return bounds


def tile_center_latlon(tile_id, bounds):
    """Convert a tile identifier to the latitude/longitude of its center."""
    city, level, row, col, _stride = parse_tile(tile_id)
    axis = 4 ** int(level[1:])
    min_lon, min_lat, max_lon, max_lat = bounds[city]
    lon = min_lon + (col + 0.5) / axis * (max_lon - min_lon)
    lat = max_lat - (row + 0.5) / axis * (max_lat - min_lat)
    return lat, lon


def haversine_np(lat1, lon1, lat2, lon2):
    """Compute vectorized great-circle distances in metres."""
    radius_m = 6371000.0
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return radius_m * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def validate_alignment(level_features, query_rows):
    """Reject caches whose query order or L3 labels differ across levels."""
    query_images = level_features["L1"]["query_images"]
    for level in ["L2", "L3"]:
        if level_features[level]["query_images"] != query_images:
            raise RuntimeError(f"Query order mismatch between L1 and {level}")

    l3_tiles = level_features["L3"]["tile_list"]
    l3_labels = level_features["L3"]["query_labels"][:, 0].numpy()
    if len(query_rows) != len(l3_labels):
        raise RuntimeError(
            f"Query row count mismatch: rows={len(query_rows)} labels={len(l3_labels)}"
        )

    mismatches = 0
    for idx, row in enumerate(query_rows):
        if l3_tiles[int(l3_labels[idx])] != row["L3"]:
            mismatches += 1
            if mismatches <= 5:
                print(
                    "  Label mismatch example: "
                    f"idx={idx} label={l3_tiles[int(l3_labels[idx])]} csv={row['L3']}"
                )
    if mismatches:
        raise RuntimeError(f"L3 query labels mismatch CSV rows: {mismatches}")


def evaluate_methods(args, features, query_rows):
    """Compare flat, score-fusion, and hard-cascade retrieval protocols."""
    l1_tiles = features["L1"]["tile_list"]
    l2_tiles = features["L2"]["tile_list"]
    l3_tiles = features["L3"]["tile_list"]
    p_l1_for_l3 = build_parent_indices(l3_tiles, l1_tiles, "L1")
    p_l2_for_l3 = build_parent_indices(l3_tiles, l2_tiles, "L2")
    p_l1_for_l2 = build_parent_indices(l2_tiles, l1_tiles, "L1")

    bounds = load_city_bounds(args.data_folder)
    centers = np.array([tile_center_latlon(tile, bounds) for tile in l3_tiles], dtype=np.float64)
    center_lat = centers[:, 0]
    center_lon = centers[:, 1]
    query_lat = np.array([float(row["lat"]) for row in query_rows], dtype=np.float64)
    query_lon = np.array([float(row["lon"]) for row in query_rows], dtype=np.float64)

    accumulators = {
        "L3": MetricAccumulator("L3", len(l3_tiles), center_lat, center_lon),
        "L3*L2": MetricAccumulator("L3*L2", len(l3_tiles), center_lat, center_lon),
        "L3*L1": MetricAccumulator("L3*L1", len(l3_tiles), center_lat, center_lon),
        "L3*L2*L1": MetricAccumulator("L3*L2*L1", len(l3_tiles), center_lat, center_lon),
        "Cascade L1->L2->L3": MetricAccumulator(
            "Cascade L1->L2->L3", len(l3_tiles), center_lat, center_lon
        ),
    }

    device = torch.device(args.device)
    ref_l1 = features["L1"]["ref_features"].to(device)
    ref_l2 = features["L2"]["ref_features"].to(device)
    ref_l3 = features["L3"]["ref_features"].to(device)
    qry_l1 = features["L1"]["query_features"]
    qry_l2 = features["L2"]["query_features"]
    qry_l3 = features["L3"]["query_features"]
    gt_l3 = features["L3"]["query_labels"][:, 0].long()

    p_l1_for_l3 = p_l1_for_l3.to(device)
    p_l2_for_l3 = p_l2_for_l3.to(device)
    p_l1_for_l2 = p_l1_for_l2.to(device)

    total_queries = len(qry_l3)
    print("\nScoring final L3 candidates:")
    print(f"  Queries: {total_queries}")
    print(f"  References: L1={len(l1_tiles)} L2={len(l2_tiles)} L3={len(l3_tiles)}")
    print(f"  Rtop1 uses top {max(1, len(l3_tiles) // 100)} L3 references")
    print(f"  Fusion normalization: {args.fusion_normalization}")

    with torch.no_grad():
        for start in tqdm(range(0, total_queries, args.score_batch_size), unit="chunk"):
            end = min(start + args.score_batch_size, total_queries)
            q1 = qry_l1[start:end].to(device)
            q2 = qry_l2[start:end].to(device)
            q3 = qry_l3[start:end].to(device)
            gt = gt_l3[start:end].to(device)
            q_lat = query_lat[start:end]
            q_lon = query_lon[start:end]

            sim_l1 = q1 @ ref_l1.T
            sim_l2 = q2 @ ref_l2.T
            sim_l3 = q3 @ ref_l3.T
            fuse_l1 = normalize_scores(sim_l1, args.fusion_normalization)
            fuse_l2 = normalize_scores(sim_l2, args.fusion_normalization)
            fuse_l3 = normalize_scores(sim_l3, args.fusion_normalization)

            accumulators["L3"].update(sim_l3, gt, q_lat, q_lon)

            score_l3_l2 = fuse_l3 * fuse_l2.index_select(1, p_l2_for_l3)
            accumulators["L3*L2"].update(score_l3_l2, gt, q_lat, q_lon)
            del score_l3_l2

            score_l3_l1 = fuse_l3 * fuse_l1.index_select(1, p_l1_for_l3)
            accumulators["L3*L1"].update(score_l3_l1, gt, q_lat, q_lon)
            del score_l3_l1

            score_l3_l2_l1 = (
                fuse_l3
                * fuse_l2.index_select(1, p_l2_for_l3)
                * fuse_l1.index_select(1, p_l1_for_l3)
            )
            accumulators["L3*L2*L1"].update(score_l3_l2_l1, gt, q_lat, q_lon)
            del score_l3_l2_l1

            pred_l1 = fuse_l1.argmax(dim=1)
            l2_mask = p_l1_for_l2.unsqueeze(0).eq(pred_l1.unsqueeze(1))
            cascade_l2_scores = fuse_l2.masked_fill(~l2_mask, -float("inf"))
            pred_l2 = cascade_l2_scores.argmax(dim=1)
            l3_mask = p_l2_for_l3.unsqueeze(0).eq(pred_l2.unsqueeze(1))
            cascade_l3_scores = fuse_l3.masked_fill(~l3_mask, -float("inf"))
            accumulators["Cascade L1->L2->L3"].update(
                cascade_l3_scores, gt, q_lat, q_lon
            )

            del q1, q2, q3, gt, sim_l1, sim_l2, sim_l3
            del fuse_l1, fuse_l2, fuse_l3
            del l2_mask, cascade_l2_scores, pred_l1, pred_l2, l3_mask, cascade_l3_scores

    results = [accumulators[name].finalize() for name in accumulators]
    return results


def write_reports(args, checkpoints, results):
    """Write machine-readable and Markdown ablation reports."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        level: str(path)
        for level, path in checkpoints.items()
    }
    with open(output_dir / "checkpoints.json", "w", encoding="utf-8") as handle:
        json.dump(checkpoint_payload, handle, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    table_path = output_dir / "metrics.csv"
    pd.DataFrame(results).to_csv(table_path, index=False)

    distance_columns = [
        column
        for column in results[0]
        if column.startswith("R@") and column.endswith("m")
    ]
    columns = ["method", "R1", "R5", "R10", "Rtop1", *distance_columns]
    print("\nFinal VIGOR-M Hierarchical Ablation:")
    print("  " + "  ".join(f"{col:>18}" for col in columns))
    for row in results:
        values = [row["method"]] + [f"{row[col]:.4f}" for col in columns[1:]]
        print("  " + "  ".join(f"{value:>18}" for value in values))

    print(f"\nSaved metrics: {table_path}")
    print(f"Saved details: {output_dir / 'metrics.json'}")


def main():
    """Extract level features, score all ablations, and save reports."""
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(
            Path("outputs") / "eval" /
            f"vigor_m_hier_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    setup_system(seed=1, cudnn_benchmark=True, cudnn_deterministic=False)
    sat_transform, ground_transform = build_transforms(args)

    checkpoints = {
        "L1": checkpoint_for_level(args, "L1"),
        "L2": checkpoint_for_level(args, "L2"),
        "L3": checkpoint_for_level(args, "L3"),
    }
    print("Checkpoints:")
    for level, checkpoint in checkpoints.items():
        print(f"  {level}: {checkpoint}")

    features = {}
    for level in ["L1", "L2", "L3"]:
        features[level] = extract_level_features(
            args,
            level,
            checkpoints[level],
            sat_transform,
            ground_transform,
        )

    query_rows = load_query_rows(
        args.data_folder,
        args.metadata_folder,
        split="test",
        same_area=args.same_area,
    )
    validate_alignment(features, query_rows)

    results = evaluate_methods(args, features, query_rows)
    write_reports(args, checkpoints, results)


if __name__ == "__main__":
    main()
