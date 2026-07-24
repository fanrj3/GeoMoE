#!/usr/bin/env python3
"""Build and evaluate the VIGOR-M fixed-beam path-residual pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import math
import os
import random
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "geomoe").is_dir()
)
BASE_EVALUATOR = Path(__file__).with_name("_vigor_m_protocol.py")
DATA_ROOT = PROJECT_ROOT / "data/VIGOR-M"
METADATA_ROOT = PROJECT_ROOT / "data/VIGOR-M/meta/level_pano"
BUNDLE_SCHEMA = "vigorm-b11-e5-adaptive-search-features-v1"
TABLE_SCHEMA = "vigorm-b11-e5-adaptive-search-action-table-v1"
CHECKPOINT_SHA256 = "13d75e2a456e346138e3aac62b707739f800b77fc3f7648492862d72191a3463"
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
LEVELS = ("L1", "L2", "L3")
CITIES = ("Chicago", "NewYork", "SanFrancisco", "Seattle")
WIDTHS = tuple(range(1, 9))
ACTIONS = tuple((k1, k2) for k1 in WIDTHS for k2 in WIDTHS)
TEMPERATURE = 0.07
MAX_CHILDREN = 16
PATH_FEATURE_NAMES = (
    "total_logp",
    "l1_logp",
    "l2_local_logp",
    "l3_local_logp",
    "l2_raw_similarity",
    "l3_raw_similarity",
    "l2_path_logp",
    "l1_rank_norm",
    "l2_rank_norm",
    "l3_rank_norm",
    "root_entropy_norm",
    "root_margin",
    "root_top1_probability",
    "l2_entropy_norm",
    "l2_margin",
    "l3_child_fraction",
)
DENSE_TILE_RE = re.compile(
    r"^(?P<city>[^_]+)_L1_s0\.25_r(?P<row>\d+)_c(?P<col>\d+)$"
)
HALF_TILE_RE = re.compile(
    r"^(?P<city>[^_]+)_(?P<code>L15|L25|L35)_r(?P<row>\d+)_c(?P<col>\d+)$"
)
HALF_AXES = {"L15": 8, "L25": 32, "L35": 128}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_EVALUATOR, "vigorm_adaptive_search_base")


@dataclass
class Protocol:
    dense_tiles: list[str]
    l2_tiles: list[str]
    l3_tiles: list[str]
    dense_l2_indices: torch.Tensor
    dense_l2_mask: torch.Tensor
    l2_l3_indices: torch.Tensor
    l2_l3_mask: torch.Tensor
    dense_l2_groups: list[list[int]]
    l2_l3_groups: list[list[int]]
    gt_l2: torch.Tensor
    gt_l3: torch.Tensor
    query_rows: list[dict[str, Any]]
    l3_centers: np.ndarray


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


class WidthController(nn.Module):
    def __init__(self, input_dim: int, actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class BeamExpansionGate(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch(path: Path, payload: Any) -> None:
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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = load_torch(path)
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise RuntimeError(f"Invalid feature bundle: {path}")
    checkpoint_sha256 = bundle["checkpoint"]["sha256"]
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        source = bundle.get("feature_source", {})
        expected_model = {
            "name": MODEL_NAME,
            "moe_start_block": 11,
            "num_experts": 5,
            "top_k": 2,
            "router_condition": "none",
            "expert_layout": "routed",
            "feature_dim": 768,
        }
        if (
            source.get("kind") != "same-run-checkpoint-snapshot"
            or source.get("target_checkpoint_sha256") != CHECKPOINT_SHA256
            or bundle.get("model") != expected_model
        ):
            raise RuntimeError("Feature bundle checkpoint mismatch")
    query = bundle["query_features"]
    if query.dtype != torch.float32 or query.ndim != 2 or query.shape[1] != 768:
        raise RuntimeError(f"Invalid query features: {query.shape}")
    for level in LEVELS:
        data = bundle["levels"][level]
        refs = data["ref_features"]
        labels = data["query_labels"]
        if refs.dtype != torch.float32 or refs.shape != (len(data["tile_list"]), 768):
            raise RuntimeError(f"Invalid {level} references")
        if labels.shape != (len(query), 4):
            raise RuntimeError(f"Invalid {level} labels")
    return bundle


def parse_dense_tile(tile_id: str) -> tuple[str, int, int]:
    match = DENSE_TILE_RE.fullmatch(tile_id)
    if match is None:
        raise ValueError(tile_id)
    city = match.group("city")
    row = int(match.group("row"))
    col = int(match.group("col"))
    if city not in CITIES or not (0 <= row < 16 and 0 <= col < 16):
        raise ValueError(tile_id)
    return city, row, col


def fixed_window_start(index: int) -> int:
    return min(max(index - 2, 0), 12)


def parse_half_tile(tile_id: str, expected_code: str) -> tuple[str, int, int]:
    match = HALF_TILE_RE.fullmatch(tile_id)
    if match is None or match.group("code") != expected_code:
        raise ValueError(tile_id)
    city = match.group("city")
    row = int(match.group("row"))
    col = int(match.group("col"))
    axis = HALF_AXES[expected_code]
    if city not in CITIES or not (0 <= row < axis and 0 <= col < axis):
        raise ValueError(tile_id)
    return city, row, col


def half_parent(tile_id: str, child_code: str, parent_code: str) -> str:
    city, row, col = parse_half_tile(tile_id, child_code)
    return f"{city}_{parent_code}_r{row // 4:02d}_c{col // 4:02d}"


def half_tile_from_latlon(
    city: str, lat: float, lon: float, code: str, bounds: dict[str, tuple]
) -> str:
    min_lon, min_lat, max_lon, max_lat = bounds[city]
    axis = HALF_AXES[code]
    epsilon = 1e-7
    col_position = (float(lon) - min_lon) / (max_lon - min_lon) * axis
    row_position = (max_lat - float(lat)) / (max_lat - min_lat) * axis
    col = int(math.floor(max(0.0, min(axis - epsilon, col_position))))
    row = int(math.floor(max(0.0, min(axis - epsilon, row_position))))
    return f"{city}_{code}_r{row:02d}_c{col:02d}"


def half_tile_center(tile_id: str, bounds: dict[str, tuple]) -> tuple[float, float]:
    match = HALF_TILE_RE.fullmatch(tile_id)
    if match is None:
        raise ValueError(tile_id)
    code = match.group("code")
    city, row, col = parse_half_tile(tile_id, code)
    min_lon, min_lat, max_lon, max_lat = bounds[city]
    axis = HALF_AXES[code]
    lon = min_lon + (col + 0.5) / axis * (max_lon - min_lon)
    lat = max_lat - (row + 0.5) / axis * (max_lat - min_lat)
    return lat, lon


def pad_groups(groups: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    if not groups or any(not group or len(group) > MAX_CHILDREN for group in groups):
        raise RuntimeError("Invalid child groups")
    indices = torch.zeros((len(groups), MAX_CHILDREN), dtype=torch.long)
    mask = torch.zeros((len(groups), MAX_CHILDREN), dtype=torch.bool)
    for row, group in enumerate(groups):
        if len(group) != len(set(group)):
            raise RuntimeError(f"Duplicate child in group {row}")
        indices[row, : len(group)] = torch.tensor(group)
        mask[row, : len(group)] = True
    return indices, mask


def build_protocol(bundle: dict[str, Any]) -> Protocol:
    dense_tiles = list(bundle["levels"]["L1"]["tile_list"])
    l2_tiles = list(bundle["levels"]["L2"]["tile_list"])
    l3_tiles = list(bundle["levels"]["L3"]["tile_list"])
    l2_index = {tile: index for index, tile in enumerate(l2_tiles)}
    hierarchy = bundle.get("hierarchy", {"kind": "native_dense_l1"})
    if hierarchy["kind"] == "half_levels":
        root_index = {tile: index for index, tile in enumerate(dense_tiles)}
        dense_l2_groups = [[] for _ in dense_tiles]
        for child_index, tile in enumerate(l2_tiles):
            parent = half_parent(tile, "L25", "L15")
            if parent not in root_index:
                raise RuntimeError(f"Missing L15 parent for {tile}")
            dense_l2_groups[root_index[parent]].append(child_index)
    elif hierarchy["kind"] == "native_dense_l1":
        dense_l2_groups = []
        for dense in dense_tiles:
            city, row, col = parse_dense_tile(dense)
            row_start = fixed_window_start(row)
            col_start = fixed_window_start(col)
            group = [
                l2_index[tile]
                for child_row in range(row_start, row_start + 4)
                for child_col in range(col_start, col_start + 4)
                if (tile := f"{city}_L2_r{child_row:02d}_c{child_col:02d}") in l2_index
            ]
            dense_l2_groups.append(group)
    else:
        raise RuntimeError(f"Unsupported hierarchy: {hierarchy}")
    dense_l2_indices, dense_l2_mask = pad_groups(dense_l2_groups)

    l2_l3_groups = [[] for _ in l2_tiles]
    for child_index, tile in enumerate(l3_tiles):
        parent = (
            half_parent(tile, "L35", "L25")
            if hierarchy["kind"] == "half_levels"
            else BASE.parent_tile(tile, "L2")
        )
        if parent not in l2_index:
            raise RuntimeError(f"Missing L2 parent for {tile}")
        l2_l3_groups[l2_index[parent]].append(child_index)
    l2_l3_indices, l2_l3_mask = pad_groups(l2_l3_groups)

    split = bundle["split"]
    query_rows = BASE.load_query_rows(
        str(DATA_ROOT), str(METADATA_ROOT), split=split, same_area=True
    )
    source_indices = bundle.get("source_indices")
    if source_indices is not None:
        source_indices = torch.as_tensor(source_indices).long()
        if (
            source_indices.ndim != 1
            or len(source_indices) != len(bundle["query_features"])
            or len(source_indices.unique()) != len(source_indices)
            or int(source_indices.min()) < 0
            or int(source_indices.max()) >= len(query_rows)
        ):
            raise RuntimeError("Invalid bundle source_indices")
        query_rows = [query_rows[int(index)] for index in source_indices]
    if len(query_rows) != len(bundle["query_features"]):
        raise RuntimeError("Metadata and feature query counts differ")
    gt_l2 = bundle["levels"]["L2"]["query_labels"][:, 0].long()
    gt_l3 = bundle["levels"]["L3"]["query_labels"][:, 0].long()
    bounds = BASE.load_city_bounds(str(DATA_ROOT))
    for index, row in enumerate(query_rows):
        if hierarchy["kind"] == "half_levels":
            expected_l2 = half_tile_from_latlon(
                row["city"], row["lat"], row["lon"], "L25", bounds
            )
            expected_l3 = half_tile_from_latlon(
                row["city"], row["lat"], row["lon"], "L35", bounds
            )
        else:
            expected_l2 = row["L2"]
            expected_l3 = row["L3"]
        if l2_tiles[int(gt_l2[index])] != expected_l2:
            raise RuntimeError(f"L2 alignment mismatch at {index}")
        if l3_tiles[int(gt_l3[index])] != expected_l3:
            raise RuntimeError(f"L3 alignment mismatch at {index}")
    centers = np.asarray(
        [
            half_tile_center(tile, bounds)
            if hierarchy["kind"] == "half_levels"
            else BASE.tile_center_latlon(tile, bounds)
            for tile in l3_tiles
        ],
        dtype=np.float64,
    )
    return Protocol(
        dense_tiles=dense_tiles,
        l2_tiles=l2_tiles,
        l3_tiles=l3_tiles,
        dense_l2_indices=dense_l2_indices,
        dense_l2_mask=dense_l2_mask,
        l2_l3_indices=l2_l3_indices,
        l2_l3_mask=l2_l3_mask,
        dense_l2_groups=dense_l2_groups,
        l2_l3_groups=l2_l3_groups,
        gt_l2=gt_l2,
        gt_l3=gt_l3,
        query_rows=query_rows,
        l3_centers=centers,
    )


def gather_scores(query: torch.Tensor, refs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected = refs.index_select(0, indices.reshape(-1)).reshape(*indices.shape, -1)
    expanded = query.reshape(query.shape[0], *([1] * (indices.ndim - 1)), -1)
    return (expanded * selected).sum(dim=-1)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    has_candidate = mask.any(dim=dim, keepdim=True)
    masked_logits = logits.masked_fill(~mask, -float("inf"))
    # Sparse observed hierarchies can have padded beam slots with no children.
    # Give those slots a harmless finite input, then mask their output entirely.
    safe_logits = torch.where(has_candidate, masked_logits, torch.zeros_like(logits))
    output = F.log_softmax(safe_logits, dim=dim)
    return output.masked_fill(~mask, -float("inf"))


def normalized_entropy(probabilities: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    p = probabilities.masked_fill(~mask, 0.0)
    entropy = -(p.clamp_min(1e-12).log() * p).sum(dim=dim)
    count = mask.sum(dim=dim).clamp_min(2).float()
    return entropy / count.log()


def top_state(logp: torch.Tensor, valid: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    values = logp.masked_fill(~valid, -float("inf"))
    top = torch.topk(values, k=min(k, values.shape[1]), dim=1).values
    if top.shape[1] < k:
        top = F.pad(top, (0, k - top.shape[1]), value=-float("inf"))
    finite = torch.isfinite(top)
    probs = top.exp().masked_fill(~finite, 0.0)
    # Keep controller inputs finite while preserving zero probability for padding.
    return top.masked_fill(~finite, -30.0), probs


def root_features(query: torch.Tensor, sim_l1: torch.Tensor, temperature: float):
    logp = F.log_softmax(sim_l1 / temperature, dim=1)
    probs = logp.exp()
    top_logp, top_probs = top_state(logp, torch.ones_like(logp, dtype=torch.bool))
    entropy = normalized_entropy(probs, torch.ones_like(probs, dtype=torch.bool), dim=1)
    margin = top_logp[:, 0] - top_logp[:, 1]
    stats = torch.cat(
        [
            top_logp,
            top_probs,
            entropy[:, None],
            margin[:, None],
            sim_l1.mean(dim=1, keepdim=True),
            sim_l1.std(dim=1, keepdim=True),
            sim_l1.max(dim=1, keepdim=True).values,
        ],
        dim=1,
    )
    return logp, top_logp, top_probs, entropy, margin, torch.cat([query, stats], dim=1)


@torch.inference_mode()
def search_action(
    query: torch.Tensor,
    sim_l1: torch.Tensor,
    ref_l2: torch.Tensor,
    ref_l3: torch.Tensor,
    protocol: Protocol,
    gt_l2: torch.Tensor,
    k1: int,
    k2: int,
    temperature: float,
    calibrator: PathCalibrator | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = query.device
    dense_l2_indices = protocol.dense_l2_indices.to(device)
    dense_l2_mask = protocol.dense_l2_mask.to(device)
    l2_l3_indices = protocol.l2_l3_indices.to(device)
    l2_l3_mask = protocol.l2_l3_mask.to(device)

    root_logp, root_top_logp, root_top_probs, root_entropy, root_margin, _ = root_features(
        query, sim_l1, temperature
    )
    l1_scores, l1_indices = torch.topk(root_logp, k=k1, dim=1)
    l2_candidates = dense_l2_indices[l1_indices]
    l2_valid = dense_l2_mask[l1_indices]
    l2_raw = gather_scores(query, ref_l2, l2_candidates)
    l2_local = masked_log_softmax(l2_raw / temperature, l2_valid, dim=2)
    l2_path = l1_scores.unsqueeze(2) + l2_local
    flat_l2_indices = l2_candidates.flatten(1)
    flat_l2_valid = l2_valid.flatten(1)
    flat_l2_path = l2_path.flatten(1).masked_fill(~flat_l2_valid, -float("inf"))
    flat_l1_score = l1_scores.unsqueeze(2).expand_as(l2_local).flatten(1)
    flat_l2_local = l2_local.flatten(1)
    flat_l2_raw = l2_raw.flatten(1)

    merged_l2 = query.new_full((query.shape[0], len(protocol.l2_tiles)), -float("inf"))
    merged_l2.scatter_reduce_(
        1, flat_l2_indices, flat_l2_path, reduce="amax", include_self=True
    )
    l2_scores, l2_indices = torch.topk(merged_l2, k=k2, dim=1)
    selected_l2_valid = torch.isfinite(l2_scores)
    if not bool(selected_l2_valid.any(dim=1).all()):
        raise RuntimeError("No finite L2 path")

    matches = flat_l2_indices.unsqueeze(1).eq(l2_indices.unsqueeze(2))
    matches &= selected_l2_valid.unsqueeze(2)
    provenance_scores = flat_l2_path.unsqueeze(1).expand(-1, k2, -1)
    best_positions = provenance_scores.masked_fill(~matches, -float("inf")).argmax(dim=2)
    selected_l1_score = flat_l1_score.gather(1, best_positions)
    selected_l2_local = flat_l2_local.gather(1, best_positions)
    selected_l2_raw = flat_l2_raw.gather(1, best_positions)
    selected_l1_rank = torch.div(best_positions, MAX_CHILDREN, rounding_mode="floor")

    finite_l2 = torch.isfinite(merged_l2)
    l2_distribution = F.softmax(merged_l2.masked_fill(~finite_l2, -float("inf")), dim=1)
    l2_entropy = normalized_entropy(l2_distribution, finite_l2, dim=1)
    l2_top_logp, l2_top_probs = top_state(
        F.log_softmax(merged_l2.masked_fill(~finite_l2, -float("inf")), dim=1),
        finite_l2,
    )
    l2_margin = l2_top_logp[:, 0] - l2_top_logp[:, 1]

    l3_candidates = l2_l3_indices[l2_indices]
    l3_valid = l2_l3_mask[l2_indices] & selected_l2_valid.unsqueeze(2)
    l3_raw = gather_scores(query, ref_l3, l3_candidates)
    l3_local = masked_log_softmax(l3_raw / temperature, l3_valid, dim=2)
    l3_path = l2_scores.unsqueeze(2) + l3_local
    flat_l3_ids = l3_candidates.flatten(1)
    flat_l3_valid = l3_valid.flatten(1)
    base_scores = l3_path.flatten(1).masked_fill(~flat_l3_valid, -float("inf"))

    local_order = torch.argsort(l3_local, dim=2, descending=True)
    local_rank = torch.argsort(local_order, dim=2).float()
    child_count = l3_valid.sum(dim=2).float()
    shape = l3_path.shape
    expand = lambda value: value.unsqueeze(2).expand(shape).flatten(1)
    features = torch.stack(
        [
            base_scores,
            expand(selected_l1_score),
            expand(selected_l2_local),
            l3_local.flatten(1),
            expand(selected_l2_raw),
            l3_raw.flatten(1),
            expand(l2_scores),
            expand(selected_l1_rank.float() / 7.0),
            expand(torch.arange(k2, device=device).float()[None, :].expand(query.shape[0], -1) / 7.0),
            (local_rank / 15.0).flatten(1),
            root_entropy[:, None].expand_as(base_scores),
            root_margin[:, None].expand_as(base_scores),
            root_top_probs[:, :1].expand_as(base_scores),
            l2_entropy[:, None].expand_as(base_scores),
            l2_margin[:, None].expand_as(base_scores),
            expand(child_count / 16.0),
        ],
        dim=2,
    )
    features = features.masked_fill(~flat_l3_valid.unsqueeze(2), 0.0)
    ranking_scores = base_scores
    if calibrator is not None:
        if feature_mean is None or feature_std is None:
            raise RuntimeError("Calibrator normalization is missing")
        normalized = (features - feature_mean) / feature_std
        ranking_scores = base_scores + calibrator(normalized).masked_fill(
            ~flat_l3_valid, 0.0
        )
        ranking_scores = ranking_scores.masked_fill(~flat_l3_valid, -float("inf"))

    order = torch.argsort(ranking_scores, dim=1, descending=True)
    ranked_l3 = flat_l3_ids.gather(1, order)
    ranked_valid = flat_l3_valid.gather(1, order)
    ranked_scores = ranking_scores.gather(1, order)
    ranked_valid &= torch.isfinite(ranked_scores)

    return {
        "l1_indices": l1_indices,
        "l2_indices": l2_indices,
        "ranked_l3": ranked_l3,
        "ranked_valid": ranked_valid,
        "candidate_ids": flat_l3_ids,
        "candidate_mask": flat_l3_valid,
        "candidate_features": features,
        "base_scores": base_scores,
        "ranking_scores": ranking_scores,
        "l1_reachable": (
            flat_l2_indices.eq(gt_l2.to(device).unsqueeze(1))
            & flat_l2_valid
        ).any(dim=1),
        "l2_union_count": finite_l2.sum(dim=1),
        "final_candidate_count": flat_l3_valid.sum(dim=1),
        "root_top_logp": root_top_logp,
        "root_top_probs": root_top_probs,
        "root_entropy": root_entropy,
        "root_margin": root_margin,
        "l2_top_logp": l2_top_logp,
        "l2_top_probs": l2_top_probs,
        "l2_entropy": l2_entropy,
        "l2_margin": l2_margin,
    }


def calibrator_from_checkpoint(path: Path, device: torch.device):
    payload = load_torch(path)
    if payload.get("schema") != "vigorm-b11-e5-path-calibrator-v1":
        raise RuntimeError(f"Invalid calibrator checkpoint: {path}")
    model = PathCalibrator(len(PATH_FEATURE_NAMES), int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    mean = payload["feature_mean"].to(device)
    std = payload["feature_std"].to(device)
    return model, mean, std, payload


def distances_for_predictions(
    protocol: Protocol,
    predictions: torch.Tensor,
    query_start: int = 0,
    query_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    count = int(predictions.shape[0])
    if query_indices is None:
        rows = protocol.query_rows[query_start : query_start + count]
    else:
        if len(query_indices) != count:
            raise RuntimeError("Prediction/query-index count mismatch")
        rows = [protocol.query_rows[int(index)] for index in query_indices]
    query_lat = np.asarray([float(row["lat"]) for row in rows])
    query_lon = np.asarray([float(row["lon"]) for row in rows])
    pred = predictions.cpu().numpy()
    distances = BASE.haversine_np(
        query_lat,
        query_lon,
        protocol.l3_centers[pred, 0],
        protocol.l3_centers[pred, 1],
    )
    return torch.from_numpy(distances).float()


def build_action_table(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.bundle)
    protocol = build_protocol(bundle)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    calibrator = feature_mean = feature_std = None
    calibrator_meta = None
    if args.calibrator is not None:
        calibrator, feature_mean, feature_std, calibrator_meta = calibrator_from_checkpoint(
            args.calibrator, device
        )

    query_features = bundle["query_features"]
    ref_l1 = bundle["levels"]["L1"]["ref_features"].to(device)
    ref_l2 = bundle["levels"]["L2"]["ref_features"].to(device)
    ref_l3 = bundle["levels"]["L3"]["ref_features"].to(device)
    query_start = args.start_query
    if not 0 <= query_start < len(query_features):
        raise ValueError(f"Invalid --start-query {query_start}")
    available = len(query_features) - query_start
    total = available if args.max_queries is None else min(available, args.max_queries)
    query_end = query_start + total
    fields = (
        "prediction",
        "exact",
        "hit5",
        "hit10",
        "coverage",
        "l1_reachable",
        "l2_survives",
        "l2_candidates",
        "l3_candidates",
    )
    stores = {name: [] for name in fields}
    root_inputs = []
    root_top_logp = []
    root_top_probs = []
    root_entropy = []
    root_margin = []
    l2_top_logp_by_k1 = [[] for _ in WIDTHS]
    l2_top_probs_by_k1 = [[] for _ in WIDTHS]
    l2_entropy_by_k1 = [[] for _ in WIDTHS]
    l2_margin_by_k1 = [[] for _ in WIDTHS]
    candidate_store = {name: [] for name in (
        "features", "mask", "ids", "base_scores", "ranking_scores", "gt"
    )}

    iterator = range(0, total, args.batch_size)
    for local_start in tqdm(iterator, desc=f"Action table {bundle['split']}", unit="batch"):
        local_end = min(total, local_start + args.batch_size)
        start = query_start + local_start
        end = query_start + local_end
        query = query_features[start:end].to(device)
        sim_l1 = query @ ref_l1.T
        _, top_logp, top_probs, entropy, margin, inputs = root_features(
            query, sim_l1, args.temperature
        )
        root_inputs.append(inputs.cpu())
        root_top_logp.append(top_logp.cpu())
        root_top_probs.append(top_probs.cpu())
        root_entropy.append(entropy.cpu())
        root_margin.append(margin.cpu())
        batch_gt_l2 = protocol.gt_l2[start:end].to(device)
        batch_gt_l3 = protocol.gt_l3[start:end].to(device)
        batch_fields = {name: [] for name in fields}

        for action_index, (k1, k2) in enumerate(ACTIONS):
            output = search_action(
                query,
                sim_l1,
                ref_l2,
                ref_l3,
                protocol,
                batch_gt_l2,
                k1,
                k2,
                args.temperature,
                calibrator,
                feature_mean,
                feature_std,
            )
            ranked = output["ranked_l3"]
            valid = output["ranked_valid"]
            exact = valid[:, 0] & ranked[:, 0].eq(batch_gt_l3)
            hit5 = (
                valid[:, :5] & ranked[:, :5].eq(batch_gt_l3.unsqueeze(1))
            ).any(dim=1)
            hit10 = (
                valid[:, :10] & ranked[:, :10].eq(batch_gt_l3.unsqueeze(1))
            ).any(dim=1)
            coverage = (
                output["candidate_mask"]
                & output["candidate_ids"].eq(batch_gt_l3.unsqueeze(1))
            ).any(dim=1)
            l2_survives = output["l2_indices"].eq(batch_gt_l2.unsqueeze(1)).any(dim=1)
            batch_fields["prediction"].append(ranked[:, 0].cpu())
            batch_fields["exact"].append(exact.cpu())
            batch_fields["hit5"].append(hit5.cpu())
            batch_fields["hit10"].append(hit10.cpu())
            batch_fields["coverage"].append(coverage.cpu())
            batch_fields["l1_reachable"].append(output["l1_reachable"].cpu())
            batch_fields["l2_survives"].append(l2_survives.cpu())
            batch_fields["l2_candidates"].append(output["l2_union_count"].cpu())
            batch_fields["l3_candidates"].append(output["final_candidate_count"].cpu())

            if k2 == 1:
                index = k1 - 1
                l2_top_logp_by_k1[index].append(output["l2_top_logp"].cpu())
                l2_top_probs_by_k1[index].append(output["l2_top_probs"].cpu())
                l2_entropy_by_k1[index].append(output["l2_entropy"].cpu())
                l2_margin_by_k1[index].append(output["l2_margin"].cpu())
            if (k1, k2) == (3, 3):
                candidate_store["features"].append(output["candidate_features"].cpu())
                candidate_store["mask"].append(output["candidate_mask"].cpu())
                candidate_store["ids"].append(output["candidate_ids"].cpu())
                candidate_store["base_scores"].append(output["base_scores"].cpu())
                candidate_store["ranking_scores"].append(output["ranking_scores"].cpu())
                candidate_store["gt"].append(batch_gt_l3.cpu())

        for name in fields:
            stores[name].append(torch.stack(batch_fields[name], dim=1))

    table = {name: torch.cat(parts, dim=0) for name, parts in stores.items()}
    predictions = table["prediction"]
    distances = []
    for action_index in range(len(ACTIONS)):
        distances.append(
            distances_for_predictions(
                protocol, predictions[:, action_index], query_start=query_start
            )
        )
    table["distance_m"] = torch.stack(distances, dim=1)
    table["within100"] = table["distance_m"].le(100.0)
    payload = {
        "schema": TABLE_SCHEMA,
        "bundle": str(args.bundle.resolve()),
        "bundle_signature": bundle["signature"],
        "split": bundle["split"],
        "queries": total,
        "query_start": query_start,
        "query_end": query_end,
        "temperature": args.temperature,
        "widths": WIDTHS,
        "actions": ACTIONS,
        "calibrator": calibrator_meta,
        "source_indices": (
            None
            if bundle.get("source_indices") is None
            else torch.as_tensor(bundle["source_indices"])[query_start:query_end].long()
        ),
        "root_input": torch.cat(root_inputs),
        "root_top_logp": torch.cat(root_top_logp),
        "root_top_probs": torch.cat(root_top_probs),
        "root_entropy": torch.cat(root_entropy),
        "root_margin": torch.cat(root_margin),
        "l2_top_logp_by_k1": torch.stack(
            [torch.cat(parts) for parts in l2_top_logp_by_k1], dim=1
        ),
        "l2_top_probs_by_k1": torch.stack(
            [torch.cat(parts) for parts in l2_top_probs_by_k1], dim=1
        ),
        "l2_entropy_by_k1": torch.stack(
            [torch.cat(parts) for parts in l2_entropy_by_k1], dim=1
        ),
        "l2_margin_by_k1": torch.stack(
            [torch.cat(parts) for parts in l2_margin_by_k1], dim=1
        ),
        "table": table,
        "candidate_k3": {
            name: torch.cat(parts) for name, parts in candidate_store.items()
        },
    }
    atomic_torch(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "query_start": query_start,
                "query_end": query_end,
                "queries": total,
                "actions": len(ACTIONS),
            },
            indent=2,
        )
    )


def calibrator_identity(payload: dict[str, Any]) -> tuple[str, str] | None:
    calibrator = payload.get("calibrator")
    if calibrator is None:
        return None
    return str(calibrator.get("schema")), str(calibrator.get("table"))


def merge_action_tables(args: argparse.Namespace) -> None:
    parts = [load_torch(path) for path in args.inputs]
    if not parts or any(part.get("schema") != TABLE_SCHEMA for part in parts):
        raise RuntimeError("All inputs must be adaptive-search action tables")
    parts.sort(key=lambda part: int(part.get("query_start", 0)))
    first = parts[0]
    expected_start = int(first.get("query_start", 0))
    for part in parts:
        start = int(part.get("query_start", 0))
        end = int(part.get("query_end", start + int(part["queries"])))
        if start != expected_start or end - start != int(part["queries"]):
            raise RuntimeError(f"Non-contiguous action-table shard at {start}:{end}")
        expected_start = end
        for key in ("bundle", "bundle_signature", "split", "temperature", "widths", "actions"):
            if part[key] != first[key]:
                raise RuntimeError(f"Shard metadata mismatch: {key}")
        if calibrator_identity(part) != calibrator_identity(first):
            raise RuntimeError("Shard calibrator mismatch")

    merged = dict(first)
    merged["queries"] = sum(int(part["queries"]) for part in parts)
    merged["query_start"] = int(parts[0].get("query_start", 0))
    merged["query_end"] = expected_start
    tensor_fields = (
        "root_input",
        "root_top_logp",
        "root_top_probs",
        "root_entropy",
        "root_margin",
        "l2_top_logp_by_k1",
        "l2_top_probs_by_k1",
        "l2_entropy_by_k1",
        "l2_margin_by_k1",
    )
    for key in tensor_fields:
        merged[key] = torch.cat([part[key] for part in parts], dim=0)
    source_parts = [part.get("source_indices") for part in parts]
    if all(value is None for value in source_parts):
        merged["source_indices"] = None
    elif any(value is None for value in source_parts):
        raise RuntimeError("Shard source-index metadata mismatch")
    else:
        merged["source_indices"] = torch.cat(source_parts, dim=0)
    merged["table"] = {
        key: torch.cat([part["table"][key] for part in parts], dim=0)
        for key in first["table"]
    }
    merged["candidate_k3"] = {
        key: torch.cat([part["candidate_k3"][key] for part in parts], dim=0)
        for key in first["candidate_k3"]
    }
    if merged["query_end"] - merged["query_start"] != merged["queries"]:
        raise RuntimeError("Merged query range is inconsistent")
    atomic_torch(args.output, merged)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shards": len(parts),
                "query_start": merged["query_start"],
                "query_end": merged["query_end"],
                "queries": merged["queries"],
            },
            indent=2,
        )
    )


def subset_action_table(args: argparse.Namespace) -> None:
    payload = load_torch(args.input)
    if payload.get("schema") != TABLE_SCHEMA:
        raise RuntimeError(f"Invalid action table: {args.input}")
    total = int(payload["queries"])
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(total, generator=generator)
    calibration_count = int(round(total * args.fraction))
    if not 0 < calibration_count < total:
        raise ValueError("Partition fraction creates an empty subset")
    if args.partition == "calibration":
        indices = order[:calibration_count].sort().values
    else:
        indices = order[calibration_count:].sort().values

    base_source = payload.get("source_indices")
    if base_source is None:
        start = int(payload.get("query_start", 0))
        base_source = torch.arange(start, start + total)
    subset = dict(payload)
    subset["queries"] = len(indices)
    subset["query_start"] = None
    subset["query_end"] = None
    subset["source_indices"] = base_source[indices]
    subset["partition"] = {
        "name": args.partition,
        "fraction": args.fraction,
        "seed": args.seed,
        "parent": str(args.input.resolve()),
        "parent_queries": total,
    }
    tensor_fields = (
        "root_input",
        "root_top_logp",
        "root_top_probs",
        "root_entropy",
        "root_margin",
        "l2_top_logp_by_k1",
        "l2_top_probs_by_k1",
        "l2_entropy_by_k1",
        "l2_margin_by_k1",
    )
    for key in tensor_fields:
        subset[key] = payload[key][indices]
    subset["table"] = {
        key: value[indices] for key, value in payload["table"].items()
    }
    subset["candidate_k3"] = {
        key: value[indices] for key, value in payload["candidate_k3"].items()
    }
    atomic_torch(args.output, subset)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "partition": args.partition,
                "queries": len(indices),
                "source_index_sha256": hashlib.sha256(
                    subset["source_indices"].numpy().tobytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


def split_indices(total: int, seed: int = 17, val_fraction: float = 0.2):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=generator)
    val_count = int(round(total * val_fraction))
    return order[val_count:], order[:val_count]


def train_calibrator(args: argparse.Namespace) -> None:
    data = load_torch(args.table)
    candidates = data["candidate_k3"]
    features = candidates["features"].float()
    mask = candidates["mask"].bool()
    ids = candidates["ids"].long()
    base_scores = candidates["base_scores"].float()
    gt = candidates["gt"].long()
    coverage = (mask & ids.eq(gt.unsqueeze(1))).any(dim=1)
    train_idx, val_idx = split_indices(len(gt), args.seed)
    valid_train = mask[train_idx]
    flat_train = features[train_idx][valid_train]
    feature_mean = flat_train.mean(dim=0)
    feature_std = flat_train.std(dim=0).clamp_min(1e-4)
    device = torch.device(args.device)
    model = PathCalibrator(features.shape[2], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state = None
    best_r1 = -1.0
    history = []

    def evaluate(indices: torch.Tensor):
        model.eval()
        correct = covered = total = 0
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                idx = indices[start : start + args.batch_size]
                feat = ((features[idx] - feature_mean) / feature_std).to(device)
                valid = mask[idx].to(device)
                base = base_scores[idx].to(device)
                target_ids = ids[idx].to(device)
                target_gt = gt[idx].to(device)
                scores = base + model(feat)
                scores = scores.masked_fill(~valid, -float("inf"))
                pred = target_ids.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
                correct += int(pred.eq(target_gt).sum())
                covered += int((valid & target_ids.eq(target_gt.unsqueeze(1))).any(dim=1).sum())
                total += len(idx)
        return {"R1": 100.0 * correct / total, "coverage": 100.0 * covered / total}

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            use = idx[coverage[idx]]
            if len(use) == 0:
                continue
            feat = ((features[use] - feature_mean) / feature_std).to(device)
            valid = mask[use].to(device)
            base = base_scores[use].to(device)
            target_ids = ids[use].to(device)
            target_gt = gt[use].to(device)
            target = (valid & target_ids.eq(target_gt.unsqueeze(1))).float().argmax(dim=1)
            scores = (base + model(feat)).masked_fill(~valid, -float("inf"))
            loss = F.cross_entropy(scores, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(val_idx)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": val})
        print(json.dumps(history[-1]), flush=True)
        if val["R1"] > best_r1:
            best_r1 = val["R1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Calibrator training produced no checkpoint")
    model.load_state_dict(best_state)
    payload = {
        "schema": "vigorm-b11-e5-path-calibrator-v1",
        "table": str(args.table.resolve()),
        "hidden": args.hidden,
        "feature_names": PATH_FEATURE_NAMES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "state_dict": best_state,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "best_val_r1": best_r1,
    }
    atomic_torch(args.output, payload)
    print(json.dumps({"output": str(args.output), "best_val_r1": best_r1}, indent=2))


def oracle_labels(table: dict[str, torch.Tensor]) -> torch.Tensor:
    exact = table["exact"].bool()
    within = table["within100"].bool()
    distance = table["distance_m"].float()
    cost = table["l3_candidates"].float()
    labels = []
    for index in range(len(exact)):
        exact_actions = torch.where(exact[index])[0]
        if len(exact_actions):
            labels.append(int(exact_actions[cost[index, exact_actions].argmin()]))
            continue
        near_actions = torch.where(within[index])[0]
        if len(near_actions):
            labels.append(int(near_actions[cost[index, near_actions].argmin()]))
            continue
        minimum = distance[index].min()
        candidates = torch.where(distance[index].eq(minimum))[0]
        labels.append(int(candidates[cost[index, candidates].argmin()]))
    return torch.tensor(labels, dtype=torch.long)


def metrics_for_selection(payload: dict[str, Any], selection: torch.Tensor, name: str):
    table = payload["table"]
    rows = torch.arange(len(selection))
    chosen = {key: value[rows, selection] for key, value in table.items()}
    widths = payload["actions"]
    k1 = torch.tensor([widths[int(index)][0] for index in selection])
    k2 = torch.tensor([widths[int(index)][1] for index in selection])
    return {
        "method": name,
        "queries": len(selection),
        "R1": 100.0 * float(chosen["exact"].float().mean()),
        "R5": 100.0 * float(chosen["hit5"].float().mean()),
        "R10": 100.0 * float(chosen["hit10"].float().mean()),
        "R@100m": 100.0 * float(chosen["within100"].float().mean()),
        "R@200m": 100.0 * float(chosen["distance_m"].le(200.0).float().mean()),
        "R@300m": 100.0 * float(chosen["distance_m"].le(300.0).float().mean()),
        "coverage": 100.0 * float(chosen["coverage"].float().mean()),
        "l1_reachable": 100.0 * float(chosen["l1_reachable"].float().mean()),
        "l2_survives": 100.0 * float(chosen["l2_survives"].float().mean()),
        "avg_l2_candidates": float(chosen["l2_candidates"].float().mean()),
        "avg_final_candidates": float(chosen["l3_candidates"].float().mean()),
        "p95_final_candidates": float(torch.quantile(chosen["l3_candidates"].float(), 0.95)),
        "mean_distance_m": float(chosen["distance_m"].float().mean()),
        "median_distance_m": float(chosen["distance_m"].float().median()),
        "avg_k1": float(k1.float().mean()),
        "avg_k2": float(k2.float().mean()),
        "width_distribution": {
            f"{a},{b}": int(((k1 == a) & (k2 == b)).sum())
            for a, b in ACTIONS
            if int(((k1 == a) & (k2 == b)).sum()) > 0
        },
    }


def evaluate_current_method(args: argparse.Namespace) -> None:
    payload = load_torch(args.calibrated_test_table)
    if payload.get("schema") != TABLE_SCHEMA or payload.get("split") != "test":
        raise RuntimeError("Expected a calibrated VIGOR-M test action table")
    if payload.get("calibrator") is None:
        raise RuntimeError("The test table was built without a path calibrator")
    action = ACTIONS.index((4, 4))
    selection = torch.full((payload["queries"],), action, dtype=torch.long)
    metrics = metrics_for_selection(
        payload, selection, "GeoMoE B11/E5 + fixed K=4 + PRC"
    )
    output = {
        "schema": "geomoe-vigorm-b11-e5-current-method-v1",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "calibrated_test_table": str(args.calibrated_test_table),
        "protocol": {
            "fit_split": "official train",
            "evaluation_split": "official full test",
            "dense_levels": ["L1"],
            "l1_stride_fraction": 0.25,
            "beam_width": 4,
            "temperature": 0.07,
            "path_residual_features": list(PATH_FEATURE_NAMES),
        },
        "metrics": metrics,
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


def gate_features(payload: dict[str, Any], narrow_width: int) -> torch.Tensor:
    # Query embeddings are intentionally excluded; the gate uses only search-state uncertainty.
    root_stats = payload["root_input"][:, -21:].float()
    l2_stats = torch.cat(
        [
            payload["l2_top_logp_by_k1"][:, narrow_width - 1].float(),
            payload["l2_top_probs_by_k1"][:, narrow_width - 1].float(),
            payload["l2_entropy_by_k1"][:, narrow_width - 1 : narrow_width].float(),
            payload["l2_margin_by_k1"][:, narrow_width - 1 : narrow_width].float(),
        ],
        dim=1,
    )
    features = torch.cat([root_stats, l2_stats], dim=1)
    if features.shape[1] != 39 or not torch.isfinite(features).all():
        raise RuntimeError("Invalid expansion-gate features")
    return features


def tune_gate_threshold(
    payload: dict[str, Any],
    val_idx: torch.Tensor,
    probabilities: torch.Tensor,
    narrow_width: int,
    wide_width: int,
):
    narrow_action = ACTIONS.index((narrow_width, narrow_width))
    wide_action = ACTIONS.index((wide_width, wide_width))
    subset = {
        **payload,
        "table": {key: value[val_idx] for key, value in payload["table"].items()},
    }
    best = None
    for threshold in torch.linspace(0.0, 1.0, 201):
        selection = torch.where(
            probabilities >= threshold,
            torch.full_like(probabilities, wide_action, dtype=torch.long),
            torch.full_like(probabilities, narrow_action, dtype=torch.long),
        )
        table = subset["table"]
        rows = torch.arange(len(selection))
        r1 = 100.0 * float(table["exact"][rows, selection].float().mean())
        r100 = 100.0 * float(table["within100"][rows, selection].float().mean())
        cost = float(table["l3_candidates"][rows, selection].float().mean())
        key = (r1, r100, -cost)
        if best is None or key > best[0]:
            best = (key, float(threshold), selection)
    result = metrics_for_selection(subset, best[2], "validation")
    return best[0], best[1], result


def train_gate(args: argparse.Namespace) -> None:
    payload = load_torch(args.table)
    if payload.get("schema") != TABLE_SCHEMA:
        raise RuntimeError(f"Invalid action table: {args.table}")
    if (
        args.narrow_width not in WIDTHS
        or args.wide_width not in WIDTHS
        or args.narrow_width >= args.wide_width
    ):
        raise ValueError("Gate widths must satisfy 1 <= narrow < wide <= 8")
    features = gate_features(payload, args.narrow_width)
    table = payload["table"]
    narrow_action = ACTIONS.index((args.narrow_width, args.narrow_width))
    wide_action = ACTIONS.index((args.wide_width, args.wide_width))
    exact_narrow = table["exact"][:, narrow_action].bool()
    exact_wide = table["exact"][:, wide_action].bool()
    decisive = exact_narrow.ne(exact_wide)
    targets = (exact_wide & ~exact_narrow).float()
    train_idx, val_idx = split_indices(len(features), args.seed)
    decisive_train = train_idx[decisive[train_idx]]
    if len(decisive_train) < 32 or targets[decisive_train].unique().numel() != 2:
        raise RuntimeError("Insufficient decisive K1/K3 training examples")
    mean = features[train_idx].mean(dim=0)
    std = features[train_idx].std(dim=0).clamp_min(1e-4)
    device = torch.device(args.device)
    model = BeamExpansionGate(features.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_state = None
    best_key = None
    best_threshold = None
    best_validation = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = decisive_train[torch.randperm(len(decisive_train))]
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            x = ((features[idx] - mean) / std).to(device)
            loss = F.binary_cross_entropy_with_logits(
                model(x), targets[idx].to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            probabilities = model(
                ((features[val_idx] - mean) / std).to(device)
            ).sigmoid().cpu()
        key, threshold, validation = tune_gate_threshold(
            payload,
            val_idx,
            probabilities,
            args.narrow_width,
            args.wide_width,
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_R1": validation["R1"],
                "val_candidates": validation["avg_final_candidates"],
                "threshold": threshold,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    checkpoint = {
        "schema": "vigorm-b11-e5-expansion-gate-v1",
        "table": str(args.table.resolve()),
        "actions": ACTIONS,
        "calibrator_identity": calibrator_identity(payload),
        "narrow_width": args.narrow_width,
        "wide_width": args.wide_width,
        "input_dim": features.shape[1],
        "hidden": args.hidden,
        "feature_mean": mean,
        "feature_std": std,
        "threshold": best_threshold,
        "validation_result": best_validation,
        "state_dict": best_state,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters()),
        "decisive_train": len(decisive_train),
        "positive_train": int(targets[decisive_train].sum()),
        "negative_train": int(len(decisive_train) - targets[decisive_train].sum()),
        "history": history,
    }
    atomic_torch(args.output, checkpoint)
    print(json.dumps({"output": str(args.output), "validation": best_validation}, indent=2))


def gate_selection(payload: dict[str, Any], checkpoint_path: Path, device: torch.device):
    checkpoint = load_torch(checkpoint_path)
    if checkpoint.get("schema") != "vigorm-b11-e5-expansion-gate-v1":
        raise RuntimeError(f"Invalid expansion gate: {checkpoint_path}")
    if checkpoint.get("calibrator_identity") != calibrator_identity(payload):
        raise RuntimeError("Expansion-gate/calibrator identity mismatch")
    model = BeamExpansionGate(checkpoint["input_dim"], checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    narrow_width = int(checkpoint.get("narrow_width", 1))
    wide_width = int(checkpoint.get("wide_width", 3))
    features = gate_features(payload, narrow_width)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            x = (
                features[start : start + 1024] - checkpoint["feature_mean"]
            ) / checkpoint["feature_std"]
            probabilities.append(model(x.to(device)).sigmoid().cpu())
    probabilities = torch.cat(probabilities)
    narrow_action = ACTIONS.index((narrow_width, narrow_width))
    wide_action = ACTIONS.index((wide_width, wide_width))
    selection = torch.where(
        probabilities >= checkpoint["threshold"],
        torch.full_like(probabilities, wide_action, dtype=torch.long),
        torch.full_like(probabilities, narrow_action, dtype=torch.long),
    )
    return selection, checkpoint


def tune_controller_logits(
    payload: dict[str, Any],
    val_idx: torch.Tensor,
    logits: torch.Tensor,
    cost_prior: torch.Tensor,
    budget: float,
):
    table = payload["table"]
    rows = torch.arange(len(val_idx))
    exact = table["exact"][val_idx]
    near = table["within100"][val_idx]
    costs = table["l3_candidates"][val_idx].float()
    best = None
    for penalty in torch.linspace(0.0, 4.0, 81):
        score = logits - float(penalty) * (cost_prior / budget).unsqueeze(0)
        selection = score.argmax(dim=1)
        r1 = 100.0 * float(exact[rows, selection].float().mean())
        r100 = 100.0 * float(near[rows, selection].float().mean())
        cost = float(costs[rows, selection].mean())
        feasible = cost <= budget * 1.01
        key = (int(feasible), r1, r100, -cost)
        if best is None or key > best[0]:
            best = (key, float(penalty), selection)
    subset = {
        **payload,
        "table": {key: value[val_idx] for key, value in table.items()},
    }
    result = metrics_for_selection(subset, best[2], "validation")
    return best[0], best[1], result


def train_controller(args: argparse.Namespace) -> None:
    payload = load_torch(args.table)
    if payload.get("schema") != TABLE_SCHEMA:
        raise RuntimeError(f"Invalid action table: {args.table}")
    features = payload["root_input"].float()
    table = payload["table"]
    labels = oracle_labels(table)
    train_idx, val_idx = split_indices(len(features), args.seed)
    mean = features[train_idx].mean(dim=0)
    std = features[train_idx].std(dim=0).clamp_min(1e-4)
    cost_prior = table["l3_candidates"][train_idx].float().mean(dim=0)
    fixed_k3 = ACTIONS.index((3, 3))
    budget = float(table["l3_candidates"][val_idx, fixed_k3].float().mean())
    device = torch.device(args.device)
    model = WidthController(features.shape[1], len(ACTIONS), args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state = None
    best_policy_key = None
    best_penalty = None
    best_validation = None
    history = []
    exact_targets = table["exact"].float()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            x = ((features[idx] - mean) / std).to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, labels[idx].to(device))
            bce = F.binary_cross_entropy_with_logits(logits, exact_targets[idx].to(device))
            loss = ce + 0.25 * bce
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            logits = model(((features[val_idx] - mean) / std).to(device)).cpu()
        accuracy = float(logits.argmax(dim=1).eq(labels[val_idx]).float().mean())
        policy_key, penalty, validation = tune_controller_logits(
            payload, val_idx, logits, cost_prior, budget
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_action_accuracy": accuracy,
                "val_policy_R1": validation["R1"],
                "val_policy_candidates": validation["avg_final_candidates"],
                "cost_penalty": penalty,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if best_policy_key is None or policy_key > best_policy_key:
            best_policy_key = policy_key
            best_penalty = penalty
            best_validation = validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Controller training produced no checkpoint")
    checkpoint = {
        "schema": "vigorm-b11-e5-width-controller-v1",
        "table": str(args.table.resolve()),
        "actions": ACTIONS,
        "calibrator_identity": calibrator_identity(payload),
        "input_dim": features.shape[1],
        "hidden": args.hidden,
        "feature_mean": mean,
        "feature_std": std,
        "cost_prior": cost_prior,
        "cost_penalty": best_penalty,
        "validation_budget": budget,
        "validation_result": best_validation,
        "state_dict": best_state,
        "trainable_params": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    atomic_torch(args.output, checkpoint)
    print(json.dumps({"output": str(args.output), "validation": best_validation}, indent=2))


def controller_selection(payload: dict[str, Any], checkpoint_path: Path, device: torch.device):
    checkpoint = load_torch(checkpoint_path)
    if checkpoint.get("schema") != "vigorm-b11-e5-width-controller-v1":
        raise RuntimeError(f"Invalid controller checkpoint: {checkpoint_path}")
    if tuple(checkpoint["actions"]) != tuple(payload["actions"]):
        raise RuntimeError("Controller action set mismatch")
    if checkpoint.get("calibrator_identity") != calibrator_identity(payload):
        raise RuntimeError("Controller/calibrator identity mismatch")
    model = WidthController(checkpoint["input_dim"], len(ACTIONS), checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    features = payload["root_input"].float()
    selections = []
    with torch.inference_mode():
        for start in range(0, len(features), 1024):
            x = (features[start : start + 1024] - checkpoint["feature_mean"]) / checkpoint["feature_std"]
            logits = model(x.to(device)).cpu()
            score = logits - checkpoint["cost_penalty"] * (
                checkpoint["cost_prior"] / checkpoint["validation_budget"]
            ).unsqueeze(0)
            selections.append(score.argmax(dim=1))
    return torch.cat(selections), checkpoint


def count_from_top_p(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
    cumulative = probabilities.cumsum(dim=1)
    return (cumulative < threshold).sum(dim=1).add(1).clamp(1, 8)


def count_from_margin(logp: torch.Tensor, delta: float) -> torch.Tensor:
    return (logp >= logp[:, :1] - delta).sum(dim=1).clamp(1, 8)


def entropy_width(entropy: torch.Tensor, thresholds: tuple[float, float, float]) -> torch.Tensor:
    output = torch.ones_like(entropy, dtype=torch.long)
    output[entropy >= thresholds[0]] = 3
    output[entropy >= thresholds[1]] = 5
    output[entropy >= thresholds[2]] = 8
    return output


def pair_selection(payload: dict[str, Any], k1: torch.Tensor, k2: torch.Tensor):
    mapping = {pair: index for index, pair in enumerate(payload["actions"])}
    return torch.tensor([mapping[(int(a), int(b))] for a, b in zip(k1, k2)], dtype=torch.long)


def tune_static_policy(
    train_payload: dict[str, Any],
    val_idx: torch.Tensor,
    policy: str,
):
    subset = {
        **train_payload,
        "table": {key: value[val_idx] for key, value in train_payload["table"].items()},
        "root_top_probs": train_payload["root_top_probs"][val_idx],
        "root_top_logp": train_payload["root_top_logp"][val_idx],
        "root_entropy": train_payload["root_entropy"][val_idx],
        "l2_top_probs_by_k1": train_payload["l2_top_probs_by_k1"][val_idx],
        "l2_top_logp_by_k1": train_payload["l2_top_logp_by_k1"][val_idx],
        "l2_entropy_by_k1": train_payload["l2_entropy_by_k1"][val_idx],
    }
    k3_index = ACTIONS.index((3, 3))
    budget = float(subset["table"]["l3_candidates"][:, k3_index].float().mean())
    candidates = []
    if policy == "top_p":
        parameters = [(float(value),) for value in torch.linspace(0.40, 0.995, 80)]
    elif policy == "margin":
        parameters = [(float(value),) for value in torch.linspace(0.05, 6.0, 100)]
    else:
        values = torch.quantile(subset["root_entropy"], torch.linspace(0.1, 0.9, 9)).tolist()
        parameters = [
            (float(values[a]), float(values[b]), float(values[c]))
            for a in range(len(values))
            for b in range(a + 1, len(values))
            for c in range(b + 1, len(values))
        ]
    for parameter in parameters:
        if policy == "top_p":
            k1 = count_from_top_p(subset["root_top_probs"], parameter[0])
            l2 = subset["l2_top_probs_by_k1"][torch.arange(len(k1)), k1 - 1]
            k2 = count_from_top_p(l2, parameter[0])
        elif policy == "margin":
            k1 = count_from_margin(subset["root_top_logp"], parameter[0])
            l2 = subset["l2_top_logp_by_k1"][torch.arange(len(k1)), k1 - 1]
            k2 = count_from_margin(l2, parameter[0])
        else:
            k1 = entropy_width(subset["root_entropy"], parameter)
            l2 = subset["l2_entropy_by_k1"][torch.arange(len(k1)), k1 - 1]
            k2 = entropy_width(l2, parameter)
        selection = pair_selection(subset, k1, k2)
        result = metrics_for_selection(subset, selection, f"validation_{policy}")
        feasible = result["avg_final_candidates"] <= budget * 1.01
        key = (int(feasible), result["R1"], result["R@100m"], -result["avg_final_candidates"])
        candidates.append((key, parameter, result))
    return max(candidates, key=lambda item: item[0]), budget


def apply_static_policy(payload: dict[str, Any], policy: str, parameter: tuple[float, ...]):
    if policy == "top_p":
        k1 = count_from_top_p(payload["root_top_probs"], parameter[0])
        l2 = payload["l2_top_probs_by_k1"][torch.arange(len(k1)), k1 - 1]
        k2 = count_from_top_p(l2, parameter[0])
    elif policy == "margin":
        k1 = count_from_margin(payload["root_top_logp"], parameter[0])
        l2 = payload["l2_top_logp_by_k1"][torch.arange(len(k1)), k1 - 1]
        k2 = count_from_margin(l2, parameter[0])
    else:
        thresholds = (parameter[0], parameter[1], parameter[2])
        k1 = entropy_width(payload["root_entropy"], thresholds)
        l2 = payload["l2_entropy_by_k1"][torch.arange(len(k1)), k1 - 1]
        k2 = entropy_width(l2, thresholds)
    return pair_selection(payload, k1, k2)


def budget_oracle(payload: dict[str, Any], budget: float):
    table = payload["table"]
    exact = table["exact"].float()
    near = table["within100"].float()
    cost = table["l3_candidates"].float()
    best = None
    for penalty in torch.logspace(-5, 0, 301):
        utility = exact + 0.05 * near - float(penalty) * (cost / budget)
        selection = utility.argmax(dim=1)
        result = metrics_for_selection(payload, selection, "Oracle budget adaptive")
        feasible = result["avg_final_candidates"] <= budget * 1.001
        key = (int(feasible), result["R1"], result["R@100m"], -result["avg_final_candidates"])
        if best is None or key > best[0]:
            best = (key, float(penalty), selection, result)
    return best


@torch.inference_mode()
def flat_l3_metrics(
    bundle_path: Path,
    device: torch.device,
    batch_size: int = 512,
    query_indices: torch.Tensor | None = None,
):
    bundle = load_bundle(bundle_path)
    protocol = build_protocol(bundle)
    all_query = bundle["query_features"]
    if query_indices is None:
        query_indices = torch.arange(len(all_query))
    query = all_query[query_indices]
    refs = bundle["levels"]["L3"]["ref_features"].to(device)
    predictions = []
    hit5 = []
    hit10 = []
    gt = protocol.gt_l3[query_indices]
    for start in tqdm(range(0, len(query), batch_size), desc="Flat L3", unit="batch"):
        end = min(len(query), start + batch_size)
        scores = query[start:end].to(device) @ refs.T
        top = scores.topk(k=10, dim=1).indices.cpu()
        batch_gt = gt[start:end].unsqueeze(1)
        predictions.append(top[:, 0])
        hit5.append(top[:, :5].eq(batch_gt).any(dim=1))
        hit10.append(top.eq(batch_gt).any(dim=1))
    predictions = torch.cat(predictions)
    distances = distances_for_predictions(
        protocol, predictions, query_indices=query_indices
    )
    gallery = len(protocol.l3_tiles)
    return {
        "method": "Flat exhaustive L3",
        "queries": len(query),
        "R1": 100.0 * float(predictions.eq(gt).float().mean()),
        "R5": 100.0 * float(torch.cat(hit5).float().mean()),
        "R10": 100.0 * float(torch.cat(hit10).float().mean()),
        "R@100m": 100.0 * float(distances.le(100.0).float().mean()),
        "coverage": 100.0,
        "l1_reachable": None,
        "l2_survives": None,
        "avg_l2_candidates": None,
        "avg_final_candidates": float(gallery),
        "p95_final_candidates": float(gallery),
        "mean_distance_m": float(distances.mean()),
        "median_distance_m": float(distances.median()),
        "avg_k1": None,
        "avg_k2": None,
        "width_distribution": {},
    }


def evaluate_tables(args: argparse.Namespace) -> None:
    train = load_torch(args.train_table)
    test = load_torch(args.test_table)
    calibrated = load_torch(args.calibrated_test_table)
    if (
        train.get("schema") != TABLE_SCHEMA
        or test.get("schema") != TABLE_SCHEMA
        or calibrated.get("schema") != TABLE_SCHEMA
    ):
        raise RuntimeError("Invalid action tables")
    if test["split"] != "test" or calibrated["split"] != "test":
        raise RuntimeError("Evaluation tables must come from the test split")
    if test["bundle_signature"] != calibrated["bundle_signature"]:
        raise RuntimeError("Base and calibrated test tables use different bundles")
    if test.get("calibrator") is not None or calibrated.get("calibrator") is None:
        raise RuntimeError("Expected an uncalibrated base table and calibrated test table")
    _, val_idx = split_indices(train["queries"], args.seed)
    device = torch.device(args.device)
    results = [
        flat_l3_metrics(
            args.test_bundle,
            device,
            query_indices=test.get("source_indices"),
        )
    ]
    for width in WIDTHS:
        action = ACTIONS.index((width, width))
        selection = torch.full((test["queries"],), action, dtype=torch.long)
        results.append(metrics_for_selection(test, selection, f"Fixed Beam K={width}"))

    static_config = {}
    for policy in ("top_p", "margin", "entropy"):
        best, budget = tune_static_policy(train, val_idx, policy)
        selection = apply_static_policy(test, policy, best[1])
        result = metrics_for_selection(test, selection, f"Adaptive {policy}")
        result["tuned_parameter"] = list(best[1])
        result["validation_budget"] = budget
        results.append(result)
        static_config[policy] = {"parameter": list(best[1]), "validation": best[2]}

    selection, controller = controller_selection(test, args.controller, device)
    controller_result = metrics_for_selection(test, selection, "Learned width controller")
    results.append(controller_result)
    gate = None
    if args.gate is not None:
        gate_selection_test, gate = gate_selection(test, args.gate, device)
        gate_name = f"Learned K{gate.get('narrow_width', 1)}/K{gate.get('wide_width', 3)} expansion gate"
        results.append(
            metrics_for_selection(
                test, gate_selection_test, gate_name
            )
        )

    for width in WIDTHS:
        action = ACTIONS.index((width, width))
        fixed_selection = torch.full((test["queries"],), action, dtype=torch.long)
        results.append(
            metrics_for_selection(
                calibrated,
                fixed_selection,
                f"Path calibrator + fixed K={width}",
            )
        )

    joint_selection, joint_controller = controller_selection(
        calibrated, args.joint_controller, device
    )
    joint_result = metrics_for_selection(
        calibrated, joint_selection, "Path calibrator + learned controller"
    )
    results.append(joint_result)
    joint_gate = None
    if args.joint_gate is not None:
        joint_gate_selection, joint_gate = gate_selection(
            calibrated, args.joint_gate, device
        )
        joint_gate_name = (
            "Path calibrator + learned "
            f"K{joint_gate.get('narrow_width', 1)}/K{joint_gate.get('wide_width', 3)} expansion gate"
        )
        results.append(
            metrics_for_selection(
                calibrated,
                joint_gate_selection,
                joint_gate_name,
            )
        )

    fixed_k3_result = next(row for row in results if row["method"] == "Fixed Beam K=3")
    budget = fixed_k3_result["avg_final_candidates"]
    oracle = budget_oracle(test, budget)
    results.append(oracle[3])
    unconstrained_selection = oracle_labels(test["table"])
    results.append(
        metrics_for_selection(test, unconstrained_selection, "Oracle per-query action")
    )

    if args.best_first_metrics is not None:
        best_first = json.loads(args.best_first_metrics.read_text(encoding="utf-8"))
        results.append(best_first["metrics"])

    output = {
        "schema": "vigorm-b11-e5-adaptive-search-results-v1",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "train_table": str(args.train_table.resolve()),
        "test_table": str(args.test_table.resolve()),
        "test_bundle": str(args.test_bundle.resolve()),
        "calibrated_test_table": str(args.calibrated_test_table.resolve()),
        "controller": str(args.controller.resolve()),
        "joint_controller": str(args.joint_controller.resolve()),
        "static_policy_config": static_config,
        "controller_validation": controller["validation_result"],
        "joint_controller_validation": joint_controller["validation_result"],
        "gate_validation": None if gate is None else gate["validation_result"],
        "joint_gate_validation": (
            None if joint_gate is None else joint_gate["validation_result"]
        ),
        "oracle_budget_penalty": oracle[1],
        "metrics": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "metrics.json", output)
    flat_rows = []
    for row in results:
        flat_rows.append({key: value for key, value in row.items() if not isinstance(value, dict)})
    atomic_csv(args.output_dir / "metrics.csv", flat_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": flat_rows}, indent=2))


def np_log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values)
    shifted = values - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def best_first_one(
    root_logp: np.ndarray,
    sim_l2: np.ndarray,
    sim_l3: np.ndarray,
    protocol: Protocol,
    gt_l2: int,
    gt_l3: int,
    temperature: float,
):
    root_order = np.argsort(root_logp)[::-1]
    root_position = 0
    heap: list[tuple[float, int, int]] = []
    closed_l2: set[int] = set()
    l1_expanded = l2_scored = l2_expanded = l3_scored = 0
    l1_reachable = l2_survives = coverage = False
    while root_position < len(root_order) or heap:
        next_root_score = (
            float(root_logp[root_order[root_position]])
            if root_position < len(root_order)
            else -float("inf")
        )
        next_heap_score = -heap[0][0] if heap else -float("inf")
        if next_root_score >= next_heap_score:
            root = int(root_order[root_position])
            root_position += 1
            children = protocol.dense_l2_groups[root]
            local = np_log_softmax(sim_l2[children] / temperature)
            l1_expanded += 1
            l2_scored += len(children)
            if gt_l2 in children:
                l1_reachable = True
            for child, local_score in zip(children, local):
                heapq.heappush(
                    heap, (-(next_root_score + float(local_score)), 1, int(child))
                )
            continue

        negative_score, kind, node = heapq.heappop(heap)
        path_score = -negative_score
        if kind == 1:
            if node in closed_l2:
                continue
            closed_l2.add(node)
            children = protocol.l2_l3_groups[node]
            local = np_log_softmax(sim_l3[children] / temperature)
            l2_expanded += 1
            l3_scored += len(children)
            if node == gt_l2:
                l2_survives = True
            if gt_l3 in children:
                coverage = True
            for child, local_score in zip(children, local):
                heapq.heappush(
                    heap, (-(path_score + float(local_score)), 2, int(child))
                )
            continue
        return {
            "prediction": node,
            "l1_expanded": l1_expanded,
            "l2_candidates": l2_scored,
            "l2_expanded": l2_expanded,
            "l3_candidates": l3_scored,
            "l1_reachable": l1_reachable,
            "l2_survives": l2_survives,
            "coverage": coverage,
        }
    raise RuntimeError("Best-first search exhausted without a leaf")


def run_best_first(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.bundle)
    protocol = build_protocol(bundle)
    device = torch.device(args.device)
    query_features = bundle["query_features"]
    ref_l1 = bundle["levels"]["L1"]["ref_features"].to(device)
    ref_l2 = bundle["levels"]["L2"]["ref_features"].to(device)
    ref_l3 = bundle["levels"]["L3"]["ref_features"].to(device)
    total = len(query_features) if args.max_queries is None else min(len(query_features), args.max_queries)
    outputs = []
    for start in tqdm(range(0, total, args.batch_size), desc="Exact best-first", unit="batch"):
        end = min(total, start + args.batch_size)
        query = query_features[start:end].to(device)
        with torch.inference_mode():
            sim1 = (query @ ref_l1.T).cpu().numpy()
            sim2 = (query @ ref_l2.T).cpu().numpy()
            sim3 = (query @ ref_l3.T).cpu().numpy()
        root = torch.log_softmax(torch.from_numpy(sim1) / args.temperature, dim=1).numpy()
        for local_index in range(end - start):
            index = start + local_index
            outputs.append(
                best_first_one(
                    root[local_index],
                    sim2[local_index],
                    sim3[local_index],
                    protocol,
                    int(protocol.gt_l2[index]),
                    int(protocol.gt_l3[index]),
                    args.temperature,
                )
            )
    predictions = torch.tensor([row["prediction"] for row in outputs])
    gt = protocol.gt_l3[:total]
    distances = distances_for_predictions(protocol, predictions)[:total]
    def values(name):
        return torch.tensor([row[name] for row in outputs], dtype=torch.float32)
    metrics = {
        "method": "Exact best-first MAP",
        "queries": total,
        "R1": 100.0 * float(predictions.eq(gt).float().mean()),
        "R5": None,
        "R10": None,
        "R@100m": 100.0 * float(distances.le(100).float().mean()),
        "coverage": 100.0 * float(values("coverage").mean()),
        "l1_reachable": 100.0 * float(values("l1_reachable").mean()),
        "l2_survives": 100.0 * float(values("l2_survives").mean()),
        "avg_l2_candidates": float(values("l2_candidates").mean()),
        "avg_final_candidates": float(values("l3_candidates").mean()),
        "p95_final_candidates": float(torch.quantile(values("l3_candidates"), 0.95)),
        "mean_distance_m": float(distances.mean()),
        "median_distance_m": float(distances.median()),
        "avg_k1": float(values("l1_expanded").mean()),
        "avg_k2": float(values("l2_expanded").mean()),
        "width_distribution": {},
    }
    payload = {
        "schema": "vigorm-b11-e5-exact-best-first-v1",
        "bundle": str(args.bundle.resolve()),
        "temperature": args.temperature,
        "metrics": metrics,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, default=DATA_ROOT)
    parser.add_argument("--metadata-folder", type=Path, default=METADATA_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-table")
    build.add_argument("--bundle", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--calibrator", type=Path)
    build.add_argument("--device", default="cuda")
    build.add_argument("--batch-size", type=int, default=256)
    build.add_argument("--temperature", type=float, default=TEMPERATURE)
    build.add_argument("--start-query", type=int, default=0)
    build.add_argument("--max-queries", type=int)

    merge = sub.add_parser("merge-tables")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)

    subset = sub.add_parser("subset-table")
    subset.add_argument("--input", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--partition", choices=("calibration", "heldout"), required=True)
    subset.add_argument("--fraction", type=float, default=0.2)
    subset.add_argument("--seed", type=int, default=17)

    calibrator = sub.add_parser("train-calibrator")
    calibrator.add_argument("--table", type=Path, required=True)
    calibrator.add_argument("--output", type=Path, required=True)
    calibrator.add_argument("--device", default="cuda")
    calibrator.add_argument("--hidden", type=int, default=48)
    calibrator.add_argument("--epochs", type=int, default=20)
    calibrator.add_argument("--batch-size", type=int, default=512)
    calibrator.add_argument("--lr", type=float, default=2e-3)
    calibrator.add_argument("--seed", type=int, default=17)

    controller = sub.add_parser("train-controller")
    controller.add_argument("--table", type=Path, required=True)
    controller.add_argument("--output", type=Path, required=True)
    controller.add_argument("--device", default="cuda")
    controller.add_argument("--hidden", type=int, default=128)
    controller.add_argument("--epochs", type=int, default=30)
    controller.add_argument("--batch-size", type=int, default=512)
    controller.add_argument("--lr", type=float, default=1e-3)
    controller.add_argument("--seed", type=int, default=17)

    gate = sub.add_parser("train-gate")
    gate.add_argument("--table", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--device", default="cuda")
    gate.add_argument("--hidden", type=int, default=32)
    gate.add_argument("--epochs", type=int, default=80)
    gate.add_argument("--batch-size", type=int, default=128)
    gate.add_argument("--lr", type=float, default=2e-3)
    gate.add_argument("--seed", type=int, default=17)
    gate.add_argument("--narrow-width", type=int, default=1)
    gate.add_argument("--wide-width", type=int, default=3)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--test-bundle", type=Path, required=True)
    evaluate.add_argument("--train-table", type=Path, required=True)
    evaluate.add_argument("--test-table", type=Path, required=True)
    evaluate.add_argument("--calibrated-test-table", type=Path, required=True)
    evaluate.add_argument("--controller", type=Path, required=True)
    evaluate.add_argument("--joint-controller", type=Path, required=True)
    evaluate.add_argument("--gate", type=Path)
    evaluate.add_argument("--joint-gate", type=Path)
    evaluate.add_argument("--best-first-metrics", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--seed", type=int, default=17)

    best = sub.add_parser("best-first")
    best.add_argument("--bundle", type=Path, required=True)
    best.add_argument("--output", type=Path, required=True)
    best.add_argument("--device", default="cuda")
    best.add_argument("--batch-size", type=int, default=128)
    best.add_argument("--temperature", type=float, default=TEMPERATURE)
    best.add_argument("--max-queries", type=int)

    current = sub.add_parser(
        "evaluate-current",
        help="Report the locked fixed-K4 + PRC method from a calibrated test table.",
    )
    current.add_argument("--calibrated-test-table", type=Path, required=True)
    current.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
        elif isinstance(value, list) and value and all(isinstance(item, Path) for item in value):
            setattr(args, name, [item.expanduser().resolve() for item in value])
    return args


def main() -> None:
    global DATA_ROOT, METADATA_ROOT
    args = parse_args()
    DATA_ROOT = args.data_folder
    METADATA_ROOT = args.metadata_folder
    seed_everything(getattr(args, "seed", 17))
    if args.command == "build-table":
        build_action_table(args)
    elif args.command == "merge-tables":
        merge_action_tables(args)
    elif args.command == "subset-table":
        subset_action_table(args)
    elif args.command == "train-calibrator":
        train_calibrator(args)
    elif args.command == "train-controller":
        train_controller(args)
    elif args.command == "train-gate":
        train_gate(args)
    elif args.command == "evaluate":
        evaluate_tables(args)
    elif args.command == "best-first":
        run_best_first(args)
    elif args.command == "evaluate-current":
        evaluate_current_method(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
