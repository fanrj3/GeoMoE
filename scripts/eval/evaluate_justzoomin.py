#!/usr/bin/env python3
"""Evaluate the JustZoomIn B11/E5 model with fixed K=4 beam search and PRC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "geomoe").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geomoe.datasets.justzoomin import (  # noqa: E402
    JustZoomInDatasetEval,
    _epsg26985_from_latlon,
    default_satellite_cache_dir,
    resolve_justzoomin_level,
)
from geomoe.model import LevelMoEFFNTimmModel, TimmModel  # noqa: E402
from geomoe.transforms import get_transforms_val  # noqa: E402


LEVELS = ("L1", "L2", "L3", "L4")
TEMPERATURE = 0.07
BEAM_WIDTH = 4
EARTH_RADIUS_METERS = 6.378137e6
PATH_FEATURE_NAMES = (
    "total_logp",
    "l1_logp",
    "l2_local_logp",
    "l3_local_logp",
    "l4_local_logp",
    "l2_raw_similarity",
    "l3_raw_similarity",
    "l4_raw_similarity",
    "l2_path_logp",
    "l3_path_logp",
    "l1_rank_norm",
    "l2_rank_norm",
    "l3_rank_norm",
    "l4_rank_norm",
    "root_entropy_norm",
    "root_margin",
    "root_top1_probability",
    "l2_entropy_norm",
    "l2_margin",
    "l3_entropy_norm",
    "l3_margin",
    "l4_child_fraction",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class PathCalibrator(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, int(index)


@dataclass
class SplitBundle:
    split: str
    query: torch.Tensor
    references: dict[str, torch.Tensor]
    gt_l4: torch.Tensor
    protocol: "Protocol"


@dataclass
class Protocol:
    l1_l2_indices: torch.Tensor
    l1_l2_mask: torch.Tensor
    l2_l3_indices: torch.Tensor
    l2_l3_mask: torch.Tensor
    l3_l4_indices: torch.Tensor
    l3_l4_mask: torch.Tensor
    l4_parent_l1: torch.Tensor
    l4_parent_l2: torch.Tensor
    l4_parent_l3: torch.Tensor
    query_xy: np.ndarray
    reference_xy: np.ndarray
    query_latlon: np.ndarray
    reference_latlon: np.ndarray


@dataclass
class SearchOutput:
    candidate_ids: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_features: torch.Tensor
    base_scores: torch.Tensor
    final_candidate_count: torch.Tensor
    l1_indices: torch.Tensor
    l2_indices: torch.Tensor
    l3_indices: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights/justzoomin/geomoe_b11_e5_e60.pth",
    )
    parser.add_argument("--data-folder", type=Path, default=Path("data/justzoomin"))
    parser.add_argument("--satellite-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("outputs/cache/justzoomin_b11_e5_features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/eval/justzoomin_b11_e5_current_method",
    )
    parser.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--moe-start-block", type=int, default=11)
    parser.add_argument("--num-experts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--search-batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--calibrator-checkpoint",
        type=Path,
        default=None,
        help="Load an existing train-only PRC checkpoint instead of fitting it again.",
    )
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--max-val-queries", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data_folder = args.data_folder.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.feature_cache_dir = args.feature_cache_dir.expanduser().resolve()
    if args.satellite_cache_dir is None:
        args.satellite_cache_dir = default_satellite_cache_dir(args.data_folder)
    args.satellite_cache_dir = args.satellite_cache_dir.expanduser().resolve()
    if args.calibrator_checkpoint is not None:
        args.calibrator_checkpoint = args.calibrator_checkpoint.expanduser().resolve()
        if not args.calibrator_checkpoint.is_file():
            raise FileNotFoundError(args.calibrator_checkpoint)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.beam_width != 4:
        raise ValueError("The locked current-method protocol requires --beam-width 4.")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be in (0, 1)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def build_model(args: argparse.Namespace, device: torch.device) -> LevelMoEFFNTimmModel:
    model = LevelMoEFFNTimmModel(
        args.model,
        pretrained=False,
        img_size=args.img_size,
        levels=LEVELS,
        moe_start_block=args.moe_start_block,
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_jitter=0.0,
        router_condition="none",
        expert_layout="routed",
        default_level="L4",
    )
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                state = state[key]
                break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")
    return model.to(device).eval()


def build_transforms(args: argparse.Namespace):
    probe = TimmModel(args.model, pretrained=False, img_size=args.img_size)
    config = probe.get_config()
    del probe
    satellite_size = (args.img_size, args.img_size)
    ground_size = (int((288 / 512) * args.img_size * 2), args.img_size * 2)
    transforms = get_transforms_val(
        satellite_size,
        ground_size,
        mean=config["mean"],
        std=config["std"],
        ground_cutting=0,
    )
    metadata = {
        "satellite_resize_hw": list(satellite_size),
        "ground_resize_hw": list(ground_size),
        "mean": [float(value) for value in config["mean"]],
        "std": [float(value) for value in config["std"]],
        "ground_cutting": 0,
    }
    return transforms[0], transforms[1], metadata


def make_dataset(
    args: argparse.Namespace,
    split: str,
    level: str,
    img_type: str,
    transform,
) -> JustZoomInDatasetEval:
    config = resolve_justzoomin_level(level)
    dense = level in {"L1", "L2"}
    return JustZoomInDatasetEval(
        str(args.data_folder),
        split=split,
        img_type=img_type,
        sequence_depth=config["sequence_depth"],
        satellite_zoom=-3,
        satellite_crop_meters=config["satellite_crop_meters"],
        transforms=transform,
        satellite_stride_fraction=0.25 if dense else None,
        satellite_cache_dir=str(args.satellite_cache_dir) if dense else None,
        satellite_cache_size=args.img_size,
    )


def cache_path(
    args: argparse.Namespace,
    checkpoint_sha256: str,
    split: str,
    level: str,
    img_type: str,
    count: int,
) -> Path:
    dense_tag = "dense0p25" if level in {"L1", "L2"} else "native"
    return args.feature_cache_dir / (
        f"{split}_{img_type}_{level}_{dense_tag}_size{args.img_size}_"
        f"n{count}_{checkpoint_sha256[:12]}.pt"
    )


def make_loader(dataset: Dataset, args: argparse.Namespace, device: torch.device):
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(IndexedDataset(dataset), **kwargs)


@torch.inference_mode()
def extract_or_load(
    model: LevelMoEFFNTimmModel,
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_sha256: str,
    split: str,
    level: str,
    img_type: str,
    max_queries: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if img_type == "query" and max_queries is not None:
        dataset = Subset(dataset, range(min(max_queries, len(dataset))))
    path = cache_path(args, checkpoint_sha256, split, level, img_type, len(dataset))
    if path.exists() and not args.force_extract:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        features = payload["features"]
        labels = payload["labels"]
        if features.shape != (len(dataset), 768) or len(labels) != len(dataset):
            raise RuntimeError(f"Invalid feature cache: {path}")
        print(f"Loaded {path}: features={tuple(features.shape)}", flush=True)
        return features, labels

    features, labels, indices = [], [], []
    started = time.time()
    description = f"Extract {split} {img_type} {level}"
    for images, batch_labels, batch_indices in tqdm(
        make_loader(dataset, args, device), desc=description, unit="batch"
    ):
        images = images.to(device, non_blocking=device.type == "cuda")
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = F.normalize(model(images, levels=level), dim=-1)
        features.append(output.float().cpu())
        labels.append(batch_labels.long().cpu())
        indices.append(torch.as_tensor(batch_indices).long().cpu())
    actual_indices = torch.cat(indices)
    if not torch.equal(actual_indices, torch.arange(len(dataset))):
        raise RuntimeError(f"Feature order changed for {description}")
    feature_tensor = torch.cat(features).contiguous()
    label_tensor = torch.cat(labels).contiguous()
    if not torch.isfinite(feature_tensor).all():
        raise RuntimeError(f"Non-finite features for {description}")
    payload = {
        "schema": "justzoomin-b11-e5-current-method-feature-v1",
        "features": feature_tensor,
        "labels": label_tensor,
        "metadata": {
            "split": split,
            "level": level,
            "img_type": img_type,
            "checkpoint_sha256": checkpoint_sha256,
            "items": len(dataset),
        },
    }
    atomic_torch_save(path, payload)
    print(
        f"Saved {path}: features={tuple(feature_tensor.shape)} "
        f"elapsed_min={(time.time() - started) / 60:.2f}",
        flush=True,
    )
    return feature_tensor, label_tensor


def dense_idx_for_center(dataset, center_east: float, center_north: float) -> int:
    left = dataset.initial_center[0] - dataset.initial_size / 2.0
    top = dataset.initial_center[1] + dataset.initial_size / 2.0
    col = int(round((float(center_east) - left) / dataset.dense_stride_meters - 0.5))
    row = int(round((top - float(center_north)) / dataset.dense_stride_meters - 0.5))
    col = max(0, min(dataset.dense_cells_per_axis - 1, col))
    row = max(0, min(dataset.dense_cells_per_axis - 1, row))
    return row * dataset.dense_cells_per_axis + col


def child_to_dense_parent(child_dataset, parent_dataset) -> torch.Tensor:
    return torch.tensor(
        [
            dense_idx_for_center(parent_dataset, *child_dataset.idx2tile_center[int(index)])
            for index in child_dataset.images
        ],
        dtype=torch.long,
    )


def groups_from_parents(parent_indices: torch.Tensor, parent_count: int) -> list[list[int]]:
    groups = [[] for _ in range(parent_count)]
    for child, parent in enumerate(parent_indices.tolist()):
        groups[int(parent)].append(int(child))
    return groups


def pad_groups(groups: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max((len(group) for group in groups), default=0)
    if maximum <= 0:
        raise RuntimeError("Hierarchy has no child groups")
    indices = torch.zeros((len(groups), maximum), dtype=torch.long)
    mask = torch.zeros((len(groups), maximum), dtype=torch.bool)
    for parent, group in enumerate(groups):
        if len(group) != len(set(group)):
            raise RuntimeError(f"Duplicate child under parent {parent}")
        if group:
            indices[parent, : len(group)] = torch.tensor(group, dtype=torch.long)
            mask[parent, : len(group)] = True
    return indices, mask


def query_xy(dataset: JustZoomInDatasetEval, count: int) -> np.ndarray:
    coordinates = np.zeros((count, 2), dtype=np.float64)
    for output_index, (_, row) in enumerate(dataset.df.iloc[:count].iterrows()):
        coordinates[output_index] = _epsg26985_from_latlon(
            row["latitude"], row["longitude"]
        )
    return coordinates


def reference_xy(dataset: JustZoomInDatasetEval) -> np.ndarray:
    return np.asarray(
        [dataset.idx2tile_center[int(index)] for index in dataset.images],
        dtype=np.float64,
    )


def move_from_latlon(latlon: np.ndarray, bearing_degrees: float, distance: float) -> np.ndarray:
    bearing = np.radians(bearing_degrees)
    source = np.radians(np.asarray(latlon, dtype=np.float64))
    angular_distance = float(distance) / EARTH_RADIUS_METERS
    target_latitude = np.arcsin(
        np.sin(source[0]) * np.cos(angular_distance)
        + np.cos(source[0]) * np.sin(angular_distance) * np.cos(bearing)
    )
    target_longitude = source[1] + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(source[0]),
        np.cos(angular_distance) - np.sin(source[0]) * np.sin(target_latitude),
    )
    target_longitude = (target_longitude + np.pi) % (2.0 * np.pi) - np.pi
    return np.asarray(
        [np.degrees(target_latitude), np.degrees(target_longitude)], dtype=np.float64
    )


def sequence_center_latlon(sequence) -> np.ndarray:
    center = move_from_latlon(np.asarray([38.8936, -77.0116]), 90.0, 2000.0)
    size = 10000.0
    for action in sequence:
        patch_size = size / 4.0
        row, column = divmod(int(action), 4)
        east = (column + 0.5) * patch_size - size / 2.0
        north = -((row + 0.5) * patch_size - size / 2.0)
        center = move_from_latlon(center, 90.0, east)
        center = move_from_latlon(center, 0.0, north)
        size = patch_size
    return center


def query_latlon(dataset: JustZoomInDatasetEval, count: int) -> np.ndarray:
    return dataset.df.loc[: count - 1, ["latitude", "longitude"]].to_numpy(
        dtype=np.float64, copy=True
    )


def reference_latlon(dataset: JustZoomInDatasetEval) -> np.ndarray:
    return np.asarray(
        [sequence_center_latlon(dataset.idx2tile[int(index)]) for index in dataset.images],
        dtype=np.float64,
    )


def build_protocol(
    ref_datasets: dict[str, JustZoomInDatasetEval],
    query_dataset: JustZoomInDatasetEval,
    query_count: int,
) -> Protocol:
    l1, l2, l3, l4 = (ref_datasets[level] for level in LEVELS)
    if not l1.use_dense_satellite_grid or not l2.use_dense_satellite_grid:
        raise RuntimeError("L1 and L2 must use dense stride-0.25 galleries")
    l2_parent_l1 = child_to_dense_parent(l2, l1)
    l3_parent_l2 = child_to_dense_parent(l3, l2)
    l4_parent_l1, l4_parent_l2, l4_parent_l3 = [], [], []
    for index in l4.images:
        center = l4.idx2tile_center[int(index)]
        l4_parent_l1.append(dense_idx_for_center(l1, *center))
        l4_parent_l2.append(dense_idx_for_center(l2, *center))
        l3_key = tuple(l4.idx2tile[int(index)][:3])
        if l3_key not in l3.tile2idx:
            raise RuntimeError(f"Missing L3 parent for L4 tile {l4.idx2tile[int(index)]}")
        l4_parent_l3.append(int(l3.tile2idx[l3_key]))
    l4_parent_l1 = torch.tensor(l4_parent_l1, dtype=torch.long)
    l4_parent_l2 = torch.tensor(l4_parent_l2, dtype=torch.long)
    l4_parent_l3 = torch.tensor(l4_parent_l3, dtype=torch.long)

    l3_l4_groups = groups_from_parents(l4_parent_l3, len(l3.images))
    valid_l3 = {index for index, group in enumerate(l3_l4_groups) if group}
    l2_l3_groups = [[] for _ in l2.images]
    for l3_index, l2_index in enumerate(l3_parent_l2.tolist()):
        if l3_index in valid_l3:
            l2_l3_groups[int(l2_index)].append(l3_index)
    valid_l2 = {index for index, group in enumerate(l2_l3_groups) if group}
    l1_l2_groups = [[] for _ in l1.images]
    for l2_index, l1_index in enumerate(l2_parent_l1.tolist()):
        if l2_index in valid_l2:
            l1_l2_groups[int(l1_index)].append(l2_index)
    l1_l2_indices, l1_l2_mask = pad_groups(l1_l2_groups)
    l2_l3_indices, l2_l3_mask = pad_groups(l2_l3_groups)
    l3_l4_indices, l3_l4_mask = pad_groups(l3_l4_groups)

    if max(map(len, l1_l2_groups)) > 16:
        raise RuntimeError("A dense L1 node has more than 16 valid dense L2 children")
    if max(map(len, l2_l3_groups)) > 1:
        raise RuntimeError("A dense L2 node maps to more than one native L3 child")
    if max(map(len, l3_l4_groups)) > 16:
        raise RuntimeError("A native L3 node has more than 16 L4 children")
    print(
        "Hierarchy: "
        f"L1={len(l1.images)} -> L2={len(l2.images)} -> "
        f"L3={len(l3.images)} -> L4={len(l4.images)}; "
        f"nonempty L1/L2/L3 parents="
        f"{sum(bool(group) for group in l1_l2_groups)}/"
        f"{sum(bool(group) for group in l2_l3_groups)}/"
        f"{sum(bool(group) for group in l3_l4_groups)}",
        flush=True,
    )
    return Protocol(
        l1_l2_indices=l1_l2_indices,
        l1_l2_mask=l1_l2_mask,
        l2_l3_indices=l2_l3_indices,
        l2_l3_mask=l2_l3_mask,
        l3_l4_indices=l3_l4_indices,
        l3_l4_mask=l3_l4_mask,
        l4_parent_l1=l4_parent_l1,
        l4_parent_l2=l4_parent_l2,
        l4_parent_l3=l4_parent_l3,
        query_xy=query_xy(query_dataset, query_count),
        reference_xy=reference_xy(l4),
        query_latlon=query_latlon(query_dataset, query_count),
        reference_latlon=reference_latlon(l4),
    )


def build_split_bundle(
    args: argparse.Namespace,
    model: LevelMoEFFNTimmModel,
    device: torch.device,
    checkpoint_sha256: str,
    split: str,
    satellite_transform,
    ground_transform,
    shared_dense_references: dict[str, torch.Tensor] | None = None,
) -> SplitBundle:
    ref_datasets = {
        level: make_dataset(args, split, level, "reference", satellite_transform)
        for level in LEVELS
    }
    query_dataset = make_dataset(args, split, "L4", "query", ground_transform)
    maximum = args.max_train_queries if split == "train" else args.max_val_queries
    query, query_labels = extract_or_load(
        model,
        query_dataset,
        args,
        device,
        checkpoint_sha256,
        split,
        "L4",
        "query",
        max_queries=maximum,
    )
    references = {}
    for level in LEVELS:
        if shared_dense_references is not None and level in {"L1", "L2"}:
            references[level] = shared_dense_references[level]
            if len(references[level]) != len(ref_datasets[level]):
                raise RuntimeError(f"Shared {level} gallery size changed across splits")
            continue
        references[level], _ = extract_or_load(
            model,
            ref_datasets[level],
            args,
            device,
            checkpoint_sha256,
            split,
            level,
            "reference",
        )
    protocol = build_protocol(ref_datasets, query_dataset, len(query))
    gt_l4 = query_labels[:, 0].long() if query_labels.ndim > 1 else query_labels.long()
    if int(gt_l4.min()) < 0 or int(gt_l4.max()) >= len(references["L4"]):
        raise RuntimeError(f"{split} L4 labels outside gallery")
    return SplitBundle(split, query, references, gt_l4, protocol)


def masked_log_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int
) -> torch.Tensor:
    has_candidate = mask.any(dim=dim, keepdim=True)
    masked = logits.masked_fill(~mask, -float("inf"))
    safe = torch.where(has_candidate, masked, torch.zeros_like(masked))
    return F.log_softmax(safe, dim=dim).masked_fill(~mask, -float("inf"))


def gather_scores(
    query: torch.Tensor, references: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    selected = references.index_select(0, indices.reshape(-1)).reshape(
        *indices.shape, references.shape[1]
    )
    expanded = query.reshape(query.shape[0], *([1] * (indices.ndim - 1)), -1)
    return (expanded * selected).sum(dim=-1)


def normalized_entropy(probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(dim=1).clamp_min(2).float()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / count.log()


def distribution_state(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(scores)
    probabilities = F.softmax(scores.masked_fill(~finite, -float("inf")), dim=1)
    entropy = normalized_entropy(probabilities, finite)
    top_scores = scores.topk(k=2, dim=1).values
    margin = torch.where(
        finite.sum(dim=1) >= 2,
        top_scores[:, 0] - top_scores[:, 1],
        torch.zeros_like(top_scores[:, 0]),
    )
    return finite, entropy, margin


def provenance(
    flat_ids: torch.Tensor,
    flat_scores: torch.Tensor,
    selected_ids: torch.Tensor,
) -> torch.Tensor:
    matches = flat_ids.unsqueeze(1).eq(selected_ids.unsqueeze(2))
    candidate_scores = flat_scores.unsqueeze(1).expand(-1, selected_ids.shape[1], -1)
    return candidate_scores.masked_fill(~matches, -float("inf")).argmax(dim=2)


@torch.inference_mode()
def search_candidates(
    query: torch.Tensor,
    references: dict[str, torch.Tensor],
    protocol: Protocol,
    temperature: float,
    beam_width: int,
) -> SearchOutput:
    device = query.device
    l1_l2_indices = protocol.l1_l2_indices.to(device)
    l1_l2_mask = protocol.l1_l2_mask.to(device)
    l2_l3_indices = protocol.l2_l3_indices.to(device)
    l2_l3_mask = protocol.l2_l3_mask.to(device)
    l3_l4_indices = protocol.l3_l4_indices.to(device)
    l3_l4_mask = protocol.l3_l4_mask.to(device)
    r1, r2, r3, r4 = (references[level] for level in LEVELS)

    root_valid = l1_l2_mask.any(dim=1)
    root_mask = root_valid.unsqueeze(0).expand(len(query), -1)
    root_logp = masked_log_softmax(
        (query @ r1.T) / temperature, root_mask, dim=1
    )
    root_probabilities = root_logp.exp()
    root_entropy = normalized_entropy(
        root_probabilities, torch.ones_like(root_probabilities, dtype=torch.bool)
    )
    root_top = root_logp.topk(k=beam_width, dim=1)
    l1_scores, l1_indices = root_top.values, root_top.indices
    root_margin = l1_scores[:, 0] - l1_scores[:, 1]
    root_top1_probability = root_probabilities.gather(1, l1_indices[:, :1]).squeeze(1)

    l2_candidates = l1_l2_indices[l1_indices]
    l2_valid = l1_l2_mask[l1_indices]
    l2_raw = gather_scores(query, r2, l2_candidates)
    l2_local = masked_log_softmax(l2_raw / temperature, l2_valid, dim=2)
    l2_path = l1_scores.unsqueeze(2) + l2_local
    flat_l2_ids = l2_candidates.flatten(1)
    flat_l2_valid = l2_valid.flatten(1)
    flat_l2_path = l2_path.flatten(1).masked_fill(~flat_l2_valid, -float("inf"))
    flat_l1_score = l1_scores.unsqueeze(2).expand_as(l2_local).flatten(1)
    flat_l2_local = l2_local.flatten(1)
    flat_l2_raw = l2_raw.flatten(1)
    merged_l2 = query.new_full((len(query), len(r2)), -float("inf"))
    merged_l2.scatter_reduce_(
        1, flat_l2_ids, flat_l2_path, reduce="amax", include_self=True
    )
    l2_scores, l2_indices = merged_l2.topk(k=beam_width, dim=1)
    if not torch.isfinite(l2_scores).all():
        raise RuntimeError("No finite L2 beam")
    l2_positions = provenance(flat_l2_ids, flat_l2_path, l2_indices)
    selected_l1_score = flat_l1_score.gather(1, l2_positions)
    selected_l2_local = flat_l2_local.gather(1, l2_positions)
    selected_l2_raw = flat_l2_raw.gather(1, l2_positions)
    selected_l1_rank = torch.div(
        l2_positions, l2_candidates.shape[2], rounding_mode="floor"
    ).float()
    _, l2_entropy, l2_margin = distribution_state(merged_l2)

    l3_candidates = l2_l3_indices[l2_indices]
    l3_valid = l2_l3_mask[l2_indices]
    l3_raw = gather_scores(query, r3, l3_candidates)
    l3_local = masked_log_softmax(l3_raw / temperature, l3_valid, dim=2)
    l3_path = l2_scores.unsqueeze(2) + l3_local
    flat_l3_ids = l3_candidates.flatten(1)
    flat_l3_valid = l3_valid.flatten(1)
    flat_l3_path = l3_path.flatten(1).masked_fill(~flat_l3_valid, -float("inf"))
    flat_l3_local = l3_local.flatten(1)
    flat_l3_raw = l3_raw.flatten(1)
    expand_l3 = lambda value: value.unsqueeze(2).expand_as(l3_local).flatten(1)
    merged_l3 = query.new_full((len(query), len(r3)), -float("inf"))
    merged_l3.scatter_reduce_(
        1, flat_l3_ids, flat_l3_path, reduce="amax", include_self=True
    )
    l3_scores, l3_indices = merged_l3.topk(k=beam_width, dim=1)
    if not torch.isfinite(l3_scores[:, 0]).all():
        raise RuntimeError("No finite L3 beam")
    selected_l3_valid = torch.isfinite(l3_scores)
    l3_positions = provenance(flat_l3_ids, flat_l3_path, l3_indices)
    selected_l1_score = expand_l3(selected_l1_score).gather(1, l3_positions)
    selected_l2_local = expand_l3(selected_l2_local).gather(1, l3_positions)
    selected_l2_raw = expand_l3(selected_l2_raw).gather(1, l3_positions)
    selected_l2_path = expand_l3(l2_scores).gather(1, l3_positions)
    selected_l1_rank = expand_l3(selected_l1_rank).gather(1, l3_positions)
    flat_l2_rank = (
        torch.arange(beam_width, device=device)
        .float()[None, :, None]
        .expand_as(l3_local)
        .flatten(1)
    )
    selected_l2_rank = flat_l2_rank.gather(1, l3_positions)
    selected_l3_local = flat_l3_local.gather(1, l3_positions)
    selected_l3_raw = flat_l3_raw.gather(1, l3_positions)
    _, l3_entropy, l3_margin = distribution_state(merged_l3)

    l4_candidates = l3_l4_indices[l3_indices]
    l4_valid = l3_l4_mask[l3_indices] & selected_l3_valid.unsqueeze(2)
    l4_raw = gather_scores(query, r4, l4_candidates)
    l4_local = masked_log_softmax(l4_raw / temperature, l4_valid, dim=2)
    l4_path = l3_scores.unsqueeze(2) + l4_local
    final_ids = l4_candidates.flatten(1)
    final_mask = l4_valid.flatten(1)
    base_scores = l4_path.flatten(1).masked_fill(~final_mask, -float("inf"))
    local_order = torch.argsort(l4_local, dim=2, descending=True)
    local_rank = torch.argsort(local_order, dim=2).float()
    l4_child_count = l4_valid.sum(dim=2).float()
    final_shape = l4_path.shape
    expand_final = lambda value: value.unsqueeze(2).expand(final_shape).flatten(1)
    rank_denominator = float(max(1, beam_width - 1))
    child_denominator = float(l4_candidates.shape[2])
    l4_rank_denominator = float(max(1, l4_candidates.shape[2] - 1))
    features = torch.stack(
        [
            base_scores,
            expand_final(selected_l1_score),
            expand_final(selected_l2_local),
            expand_final(selected_l3_local),
            l4_local.flatten(1),
            expand_final(selected_l2_raw),
            expand_final(selected_l3_raw),
            l4_raw.flatten(1),
            expand_final(selected_l2_path),
            expand_final(l3_scores),
            expand_final(selected_l1_rank / rank_denominator),
            expand_final(selected_l2_rank / rank_denominator),
            expand_final(
                torch.arange(beam_width, device=device)
                .float()[None, :]
                .expand(len(query), -1)
                / rank_denominator
            ),
            (local_rank / l4_rank_denominator).flatten(1),
            root_entropy[:, None].expand_as(base_scores),
            root_margin[:, None].expand_as(base_scores),
            root_top1_probability[:, None].expand_as(base_scores),
            l2_entropy[:, None].expand_as(base_scores),
            l2_margin[:, None].expand_as(base_scores),
            l3_entropy[:, None].expand_as(base_scores),
            l3_margin[:, None].expand_as(base_scores),
            expand_final(l4_child_count / child_denominator),
        ],
        dim=2,
    )
    if features.shape[2] != len(PATH_FEATURE_NAMES):
        raise RuntimeError("Path feature dimension mismatch")
    features = features.masked_fill(~final_mask.unsqueeze(2), 0.0)
    return SearchOutput(
        candidate_ids=final_ids,
        candidate_mask=final_mask,
        candidate_features=features,
        base_scores=base_scores,
        final_candidate_count=final_mask.sum(dim=1),
        l1_indices=l1_indices,
        l2_indices=l2_indices,
        l3_indices=l3_indices.masked_fill(~selected_l3_valid, -1),
    )


def split_indices(total: int, seed: int, val_fraction: float):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=generator)
    val_count = int(round(total * val_fraction))
    return order[val_count:], order[:val_count]


def feature_statistics(
    bundle: SplitBundle,
    train_indices: torch.Tensor,
    references: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(len(PATH_FEATURE_NAMES), dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    count = 0
    for start in tqdm(
        range(0, len(train_indices), args.search_batch_size),
        desc="PRC feature statistics",
        unit="batch",
    ):
        indices = train_indices[start : start + args.search_batch_size]
        query = bundle.query[indices].to(device, non_blocking=True)
        output = search_candidates(
            query, references, bundle.protocol, args.temperature, args.beam_width
        )
        valid = output.candidate_mask
        values = output.candidate_features[valid].double()
        total += values.sum(dim=0)
        square += (values * values).sum(dim=0)
        count += len(values)
    mean = total / count
    variance = (square / count - mean * mean).clamp_min(0.0)
    std = variance.sqrt().clamp_min(1e-4)
    return mean.float(), std.float()


@torch.inference_mode()
def exact_accuracy(
    bundle: SplitBundle,
    indices: torch.Tensor,
    references: dict[str, torch.Tensor],
    args: argparse.Namespace,
    calibrator: PathCalibrator | None,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    correct = covered = total = 0
    calibrator_was_training = calibrator.training if calibrator is not None else False
    if calibrator is not None:
        calibrator.eval()
    for start in range(0, len(indices), args.search_batch_size):
        batch_indices = indices[start : start + args.search_batch_size]
        query = bundle.query[batch_indices].to(device, non_blocking=True)
        output = search_candidates(
            query, references, bundle.protocol, args.temperature, args.beam_width
        )
        scores = output.base_scores
        if calibrator is not None:
            normalized = (output.candidate_features - feature_mean) / feature_std
            scores = scores + calibrator(normalized).masked_fill(
                ~output.candidate_mask, 0.0
            )
        scores = scores.masked_fill(~output.candidate_mask, -float("inf"))
        prediction = output.candidate_ids.gather(
            1, scores.argmax(dim=1, keepdim=True)
        ).squeeze(1)
        gt = bundle.gt_l4[batch_indices].to(device)
        correct += int(prediction.eq(gt).sum())
        covered += int(
            (output.candidate_mask & output.candidate_ids.eq(gt.unsqueeze(1)))
            .any(dim=1)
            .sum()
        )
        total += len(batch_indices)
    if calibrator is not None and calibrator_was_training:
        calibrator.train()
    return {"R1": 100.0 * correct / total, "coverage": 100.0 * covered / total}


def train_calibrator(
    bundle: SplitBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PathCalibrator, torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_indices, val_indices = split_indices(
        len(bundle.query), args.seed, args.val_fraction
    )
    references = {
        level: bundle.references[level].to(device, non_blocking=True) for level in LEVELS
    }
    feature_mean, feature_std = feature_statistics(
        bundle, train_indices, references, args, device
    )
    model = PathCalibrator(len(PATH_FEATURE_NAMES), args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state = None
    best_r1 = -1.0
    history = []

    baseline = exact_accuracy(
        bundle,
        val_indices,
        references,
        args,
        None,
        feature_mean,
        feature_std,
        device,
    )
    print(f"Internal train validation baseline: {json.dumps(baseline)}", flush=True)
    generator = torch.Generator().manual_seed(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        losses = []
        covered_queries = 0
        for start in range(0, len(order), args.search_batch_size):
            indices = order[start : start + args.search_batch_size]
            query = bundle.query[indices].to(device, non_blocking=True)
            output = search_candidates(
                query, references, bundle.protocol, args.temperature, args.beam_width
            )
            gt = bundle.gt_l4[indices].to(device)
            coverage = (
                output.candidate_mask
                & output.candidate_ids.eq(gt.unsqueeze(1))
            ).any(dim=1)
            if not bool(coverage.any()):
                continue
            features = output.candidate_features[coverage]
            valid = output.candidate_mask[coverage]
            base = output.base_scores[coverage]
            ids = output.candidate_ids[coverage]
            covered_gt = gt[coverage]
            normalized = (features - feature_mean) / feature_std
            scores = (base + model(normalized)).masked_fill(~valid, -float("inf"))
            target = (valid & ids.eq(covered_gt.unsqueeze(1))).float().argmax(dim=1)
            loss = F.cross_entropy(scores, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            covered_queries += int(coverage.sum())
        validation = exact_accuracy(
            bundle,
            val_indices,
            references,
            args,
            model,
            feature_mean,
            feature_std,
            device,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "covered_train_queries": covered_queries,
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["R1"] > best_r1:
            best_r1 = validation["R1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("PRC training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    metadata = {
        "schema": "justzoomin-b11-e5-four-level-path-calibrator-v1",
        "hidden": args.hidden,
        "feature_names": PATH_FEATURE_NAMES,
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "state_dict": best_state,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters()),
        "train_queries": len(train_indices),
        "internal_validation_queries": len(val_indices),
        "internal_validation_baseline": baseline,
        "best_internal_validation_r1": best_r1,
        "history": history,
    }
    return model, feature_mean, feature_std, metadata


def metrics_from_predictions(
    name: str,
    prediction: torch.Tensor,
    gt: torch.Tensor,
    candidate_count: torch.Tensor,
    protocol: Protocol,
    stage_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    coverage: torch.Tensor | None = None,
) -> dict[str, Any]:
    prediction_np = prediction.cpu().numpy()
    projected_distances = np.linalg.norm(
        protocol.reference_xy[prediction_np] - protocol.query_xy[: len(prediction_np)],
        axis=1,
    )
    predicted_latlon = protocol.reference_latlon[prediction_np]
    query_latlon_values = protocol.query_latlon[: len(prediction_np)]
    predicted_radians = np.radians(predicted_latlon)
    query_radians = np.radians(query_latlon_values)
    delta = predicted_radians - query_radians
    haversine = (
        np.sin(delta[:, 0] / 2.0) ** 2
        + np.cos(predicted_radians[:, 0])
        * np.cos(query_radians[:, 0])
        * np.sin(delta[:, 1] / 2.0) ** 2
    )
    distances = 2.0 * EARTH_RADIUS_METERS * np.arctan2(
        np.sqrt(haversine), np.sqrt(1.0 - haversine)
    )
    result = {
        "method": name,
        "queries": len(prediction),
        "R@1": 100.0 * float(prediction.eq(gt).float().mean()),
        "R@40m": 100.0 * float(np.mean(distances <= 40.0)),
        "R@50m": 100.0 * float(np.mean(distances <= 50.0)),
        "R@100m": 100.0 * float(np.mean(distances <= 100.0)),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "avg_final_candidates": float(candidate_count.float().mean()),
        "p95_final_candidates": float(torch.quantile(candidate_count.float(), 0.95)),
        "projected_R@40m": 100.0 * float(np.mean(projected_distances <= 40.0)),
        "projected_R@50m": 100.0 * float(np.mean(projected_distances <= 50.0)),
        "projected_R@100m": 100.0 * float(np.mean(projected_distances <= 100.0)),
        "projected_mean_m": float(np.mean(projected_distances)),
        "projected_median_m": float(np.median(projected_distances)),
    }
    if coverage is not None:
        result["coverage"] = 100.0 * float(coverage.float().mean())
    if stage_indices is not None:
        l1_indices, l2_indices, l3_indices = stage_indices
        gt_l1 = protocol.l4_parent_l1[gt]
        gt_l2 = protocol.l4_parent_l2[gt]
        gt_l3 = protocol.l4_parent_l3[gt]
        result["L1_reachable"] = 100.0 * float(
            l1_indices.eq(gt_l1.unsqueeze(1)).any(dim=1).float().mean()
        )
        result["L2_survives"] = 100.0 * float(
            l2_indices.eq(gt_l2.unsqueeze(1)).any(dim=1).float().mean()
        )
        result["L3_survives"] = 100.0 * float(
            l3_indices.eq(gt_l3.unsqueeze(1)).any(dim=1).float().mean()
        )
    return result


@torch.inference_mode()
def evaluate_beam(
    bundle: SplitBundle,
    args: argparse.Namespace,
    calibrator: PathCalibrator,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    references = {
        level: bundle.references[level].to(device, non_blocking=True) for level in LEVELS
    }
    calibrator.eval()
    base_predictions, calibrated_predictions = [], []
    candidate_counts, coverages = [], []
    stage_l1, stage_l2, stage_l3 = [], [], []
    for start in tqdm(
        range(0, len(bundle.query), args.search_batch_size),
        desc=f"Evaluate {bundle.split} Beam K=4",
        unit="batch",
    ):
        end = min(len(bundle.query), start + args.search_batch_size)
        query = bundle.query[start:end].to(device, non_blocking=True)
        output = search_candidates(
            query, references, bundle.protocol, args.temperature, args.beam_width
        )
        base = output.base_scores.masked_fill(
            ~output.candidate_mask, -float("inf")
        )
        normalized = (output.candidate_features - feature_mean) / feature_std
        calibrated = (
            output.base_scores
            + calibrator(normalized).masked_fill(~output.candidate_mask, 0.0)
        ).masked_fill(~output.candidate_mask, -float("inf"))
        base_predictions.append(
            output.candidate_ids.gather(1, base.argmax(dim=1, keepdim=True)).squeeze(1).cpu()
        )
        calibrated_predictions.append(
            output.candidate_ids
            .gather(1, calibrated.argmax(dim=1, keepdim=True))
            .squeeze(1)
            .cpu()
        )
        gt = bundle.gt_l4[start:end].to(device)
        coverages.append(
            (
                output.candidate_mask
                & output.candidate_ids.eq(gt.unsqueeze(1))
            ).any(dim=1).cpu()
        )
        candidate_counts.append(output.final_candidate_count.cpu())
        stage_l1.append(output.l1_indices.cpu())
        stage_l2.append(output.l2_indices.cpu())
        stage_l3.append(output.l3_indices.cpu())
    gt = bundle.gt_l4
    counts = torch.cat(candidate_counts)
    coverage = torch.cat(coverages)
    stages = (torch.cat(stage_l1), torch.cat(stage_l2), torch.cat(stage_l3))
    base_prediction = torch.cat(base_predictions)
    calibrated_prediction = torch.cat(calibrated_predictions)
    metrics = [
        metrics_from_predictions(
            "Fixed Beam K=4",
            base_prediction,
            gt,
            counts,
            bundle.protocol,
            stages,
            coverage,
        ),
        metrics_from_predictions(
            "Path calibrator + fixed K=4",
            calibrated_prediction,
            gt,
            counts,
            bundle.protocol,
            stages,
            coverage,
        ),
    ]
    predictions = {
        "beam_k4": base_prediction,
        "beam_k4_prc": calibrated_prediction,
        "candidate_count": counts,
        "coverage": coverage,
        "l1_indices": stages[0],
        "l2_indices": stages[1],
        "l3_indices": stages[2],
    }
    return metrics, predictions


@torch.inference_mode()
def evaluate_flat(
    bundle: SplitBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    reference = bundle.references["L4"].to(device, non_blocking=True)
    predictions = []
    for start in tqdm(
        range(0, len(bundle.query), args.search_batch_size),
        desc=f"Evaluate {bundle.split} flat L4",
        unit="batch",
    ):
        end = min(len(bundle.query), start + args.search_batch_size)
        query = bundle.query[start:end].to(device, non_blocking=True)
        predictions.append((query @ reference.T).argmax(dim=1).cpu())
    gallery = torch.full((len(bundle.query),), len(reference), dtype=torch.long)
    prediction = torch.cat(predictions)
    metrics = metrics_from_predictions(
        "Flat exhaustive L4",
        prediction,
        bundle.gt_l4,
        gallery,
        bundle.protocol,
    )
    return metrics, prediction


def load_calibrator_checkpoint(
    path: Path,
    checkpoint_sha256: str,
    device: torch.device,
) -> tuple[PathCalibrator, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "justzoomin-b11-e5-four-level-path-calibrator-v1":
        raise RuntimeError(f"Invalid calibrator checkpoint: {path}")
    if payload.get("backbone_checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("Calibrator/backbone checkpoint mismatch")
    if tuple(payload.get("feature_names", ())) != PATH_FEATURE_NAMES:
        raise RuntimeError("Calibrator path-feature definition mismatch")
    model = PathCalibrator(len(PATH_FEATURE_NAMES), int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return (
        model.to(device).eval(),
        payload["feature_mean"].to(device),
        payload["feature_std"].to(device),
        payload,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    satellite_transform, ground_transform, preprocess = build_transforms(args)
    model = build_model(args, device)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "data_folder": str(args.data_folder),
                "satellite_cache_dir": str(args.satellite_cache_dir),
                "feature_cache_dir": str(args.feature_cache_dir),
                "output_dir": str(args.output_dir),
                "preprocess": preprocess,
                "beam_width": args.beam_width,
                "temperature": args.temperature,
                "path_features": PATH_FEATURE_NAMES,
            },
            indent=2,
        ),
        flush=True,
    )

    train_bundle = build_split_bundle(
        args,
        model,
        device,
        checkpoint_sha256,
        "train",
        satellite_transform,
        ground_transform,
    )
    val_bundle = build_split_bundle(
        args,
        model,
        device,
        checkpoint_sha256,
        "val",
        satellite_transform,
        ground_transform,
        shared_dense_references=train_bundle.references,
    )
    del model
    torch.cuda.empty_cache()
    if args.dry_run:
        print("Dry run complete.")
        return

    if args.calibrator_checkpoint is None:
        calibrator, feature_mean, feature_std, checkpoint = train_calibrator(
            train_bundle, args, device
        )
        checkpoint.update(
            {
                "backbone_checkpoint": str(args.checkpoint),
                "backbone_checkpoint_sha256": checkpoint_sha256,
                "beam_width": args.beam_width,
                "temperature": args.temperature,
                "preprocess": preprocess,
                "data_protocol": "JustZoomIn official train -> official val",
            }
        )
        calibrator_path = args.output_dir / "calibrator.pt"
        atomic_torch_save(calibrator_path, checkpoint)
    else:
        calibrator_path = args.calibrator_checkpoint
        calibrator, feature_mean, feature_std, checkpoint = load_calibrator_checkpoint(
            calibrator_path, checkpoint_sha256, device
        )
    flat_metrics, flat_prediction = evaluate_flat(val_bundle, args, device)
    beam_metrics, beam_predictions = evaluate_beam(
        val_bundle,
        args,
        calibrator,
        feature_mean,
        feature_std,
        device,
    )
    results = [flat_metrics, *beam_metrics]
    output = {
        "schema": "justzoomin-b11-e5-current-method-results-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "calibrator": str(calibrator_path.resolve()),
        "protocol": {
            "fit_split": "train",
            "evaluation_split": "val",
            "query_head": "L4",
            "levels": LEVELS,
            "dense_levels": ("L1", "L2"),
            "satellite_stride_fraction": 0.25,
            "beam_width": args.beam_width,
            "temperature": args.temperature,
            "preprocess": preprocess,
            "path_feature_names": PATH_FEATURE_NAMES,
            "calibrator_trainable_params": checkpoint["trainable_params"],
            "distance_definition": "tiledwebmaps-compatible spherical distance from reconstructed L4 center to query GPS",
            "projected_distance_definition": "EPSG:26985 Euclidean distance from dataset L4 center to query GPS",
        },
        "metrics": results,
    }
    prediction_payload = {
        "schema": "justzoomin-b11-e5-current-method-predictions-v1",
        "checkpoint_sha256": checkpoint_sha256,
        "gt_l4": val_bundle.gt_l4,
        "flat_l4": flat_prediction,
        **beam_predictions,
    }
    atomic_torch_save(args.output_dir / "predictions.pt", prediction_payload)
    atomic_json(args.output_dir / "metrics.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
