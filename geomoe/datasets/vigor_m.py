"""VIGOR-M dataset adapters, dense-L1 crops, and safe batch construction.

Native L1/L2/L3 tile identifiers encode city, row, and column. Dense L1 uses
overlapping crops from each city's L1 mosaic and maps panorama coordinates to
the nearest crop center. Training samplers exclude duplicate/overlapping
positives so diagonal InfoNCE targets remain valid.
"""

import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
import math
import heapq
from tqdm import tqdm
from collections import defaultdict
import time
from pathlib import Path


VIGOR_M_CITIES = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
VIGOR_M_CITY_TO_ID = {city: idx for idx, city in enumerate(VIGOR_M_CITIES)}
_CITY_MOSAIC_CACHE = {}


def _vigor_m_panorama_root(data_folder):
    """Resolve the canonical or legacy panorama directory."""
    root = Path(data_folder)
    for candidate in (root / "panoramas", root / "Pano", root / "ground"):
        if candidate.is_dir():
            return candidate
    return root / "panoramas"


def _vigor_m_satellite_root(data_folder):
    """Resolve the canonical or legacy hierarchy directory."""
    root = Path(data_folder)
    for candidate in (root / "satellite", root / "level"):
        if candidate.is_dir():
            return candidate
    return root / "satellite"


def _vigor_m_bounds_path(data_folder):
    """Resolve the city-bounds metadata file used for coordinate mapping."""
    root = Path(data_folder)
    candidates = (
        root / "metadata" / "city_bounds.csv",
        root / "figures" / "pano_distribution" / "pano_distribution_summary.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Missing VIGOR-M city bounds. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _vigor_m_split_csv(data_folder, city, split, metadata_folder=None):
    """Resolve one city/split CSV from release or legacy metadata layouts."""
    candidates = []
    if metadata_folder is not None:
        meta_root = Path(metadata_folder)
        candidates.extend([
            meta_root / f"{city}_{split}.csv",
            meta_root / city / f"{city}_{split}.csv",
        ])

    candidates = [
        *candidates,
        Path(data_folder) / "metadata" / f"{city}_{split}.csv",
        Path(data_folder) / "metadata" / city / f"{city}_{split}.csv",
        Path(data_folder) / "meta" / "level_pano" / f"{city}_{split}.csv",
        Path(data_folder) / "meta" / "level_pano" / city / f"{city}_{split}.csv",
        Path(data_folder) / "meta" / "level" / f"{city}_{split}.csv",
        Path(data_folder) / "meta" / "level" / city / f"{city}_{split}.csv",
        Path(data_folder) / f"{city}_{split}.csv",
    ]
    for csv_path in candidates:
        if csv_path.exists():
            return str(csv_path)
    raise FileNotFoundError(candidates[0])


def _vigor_m_city_csv_split(same_area, split):
    """Map public split options to the city-specific metadata suffix."""
    # Cross-area follows original VIGOR: split by cities, using all panos
    # in each selected city rather than an intra-city train/test subset.
    if same_area:
        return split
    return "all"


def _ground_path_from_row(row, data_folder, city):
    """Resolve a portable metadata ground path against the dataset root."""
    panorama_root = _vigor_m_panorama_root(data_folder)
    if "ground_path" in row:
        path = row["ground_path"]
        if pd.notna(path) and str(path).strip():
            path = Path(str(path))
            if path.is_absolute():
                known_roots = {"Pano", "panoramas", "ground"}
                if path.parent.name == city and path.parent.parent.name in known_roots:
                    return str(panorama_root / city / path.name)
                return str(path)
            return str(Path(data_folder) / path)

    return str(panorama_root / city / row["ground"])


def _read_rgb(path):
    """Load an image and normalize OpenCV's channel order to RGB."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _tile_parts(tile_id):
    """Parse a tile identifier into city, level, row, and column."""
    parts = tile_id.split("_")
    return parts[0], parts[1], int(parts[2][1:]), int(parts[3][1:])


def _city_axis_offset(city):
    """Separate city grids in the synthetic coordinate space."""
    return VIGOR_M_CITY_TO_ID.get(city, 0) * 100000.0


def _normal_tile_center(tile_id):
    """Return a regular tile center in a city-separated coordinate space."""
    city, level, row, col = _tile_parts(tile_id)
    axis = 4 ** int(level[1:])
    return (_city_axis_offset(city) + col + 0.5, row + 0.5, axis)


def _load_city_bounds(data_folder):
    """Load release city bounds keyed by city name."""
    summary_path = _vigor_m_bounds_path(data_folder)
    df = pd.read_csv(summary_path)
    bounds = {}
    for _, row in df.iterrows():
        city = row["city"]
        bounds[city] = (
            float(row["l0_min_lon"]),
            float(row["l0_min_lat"]),
            float(row["l0_max_lon"]),
            float(row["l0_max_lat"]),
        )
    return bounds


def _build_city_l1_mosaic(data_folder, city):
    """Stitch the native 4x4 L1 tiles into a crop source for dense L1."""
    cache_key = (str(Path(data_folder).resolve()), city)
    if cache_key in _CITY_MOSAIC_CACHE:
        return _CITY_MOSAIC_CACHE[cache_key]

    tile_dir = _vigor_m_satellite_root(data_folder) / city / "L1"
    first = _read_rgb(tile_dir / f"{city}_L1_r00_c00.png")
    tile_h, tile_w = first.shape[:2]
    axis = 4
    mosaic = np.zeros((axis * tile_h, axis * tile_w, 3), dtype=first.dtype)
    mosaic[0:tile_h, 0:tile_w] = first

    for row in range(axis):
        for col in range(axis):
            if row == 0 and col == 0:
                continue
            tile_id = f"{city}_L1_r{row:02d}_c{col:02d}"
            tile = _read_rgb(tile_dir / f"{tile_id}.png")
            mosaic[row * tile_h:(row + 1) * tile_h,
                   col * tile_w:(col + 1) * tile_w] = tile

    _CITY_MOSAIC_CACHE[cache_key] = mosaic
    return mosaic


def _crop_with_padding(image, center_x, center_y, crop_w, crop_h):
    """Crop around a center, padding out-of-bounds pixels with black."""
    x0 = int(round(center_x - crop_w / 2.0))
    y0 = int(round(center_y - crop_h / 2.0))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    src_h, src_w = image.shape[:2]
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(src_w, x1)
    src_y1 = min(src_h, y1)

    crop = np.zeros((crop_h, crop_w, 3), dtype=image.dtype)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return crop

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    crop[dst_y0:dst_y0 + (src_y1 - src_y0),
         dst_x0:dst_x0 + (src_x1 - src_x0)] = image[src_y0:src_y1, src_x0:src_x1]
    return crop


def build_spatial_neighbor_dict(idx2tile_center, labels=None, top_k=128, block_size=512):
    """Build hard-negative labels from reference-center distances."""
    if labels is None:
        labels = sorted(idx2tile_center)
    else:
        labels = sorted(int(label) for label in labels)

    n = len(labels)
    k = min(int(top_k), max(0, n - 1))
    nearest = {label: [] for label in labels}
    if k == 0:
        return nearest

    coords = np.array([idx2tile_center[label][:2] for label in labels], dtype=np.float32)
    label_array = np.array(labels, dtype=np.int64)
    block_size = max(1, int(block_size))

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = coords[start:end]
        diff = block[:, None, :] - coords[None, :, :]
        dist = np.sum(diff * diff, axis=2)
        rows = np.arange(end - start)
        dist[rows, start + rows] = np.inf

        part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist, part, axis=1)
        order = np.argsort(part_dist, axis=1, kind="stable")
        ordered = np.take_along_axis(part, order, axis=1)

        for row, label in enumerate(labels[start:end]):
            nearest[label] = [int(value) for value in label_array[ordered[row]]]

    return nearest


class VigorMDatasetTrain(Dataset):
    """VIGOR-M training dataset with selectable satellite level (L1-L4).

    For a given level, each street-view image is paired with the satellite
    tile that contains its location. All other tiles serve as negatives.
    """

    def __init__(self,
                 data_folder,
                 same_area=True,
                 data_level="L3",
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 metadata_folder=None,
                 satellite_stride_fraction=None,
                 strict_cell_conflict=True,
                 ):
        super().__init__()

        self.data_folder = data_folder
        self.metadata_folder = metadata_folder
        self.data_level = data_level
        self.satellite_stride_fraction = satellite_stride_fraction
        self.strict_cell_conflict = strict_cell_conflict
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite

        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            self.cities = ['NewYork', 'Seattle']

        ground_list_raw = []

        for city in self.cities:
            csv_split = _vigor_m_city_csv_split(same_area, "train")
            csv_path = _vigor_m_split_csv(data_folder, city, csv_split, metadata_folder)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                tile_id = row[data_level]
                if pd.notna(tile_id) and tile_id:
                    ground_list_raw.append((city, row, tile_id))

        if self._use_dense_l1():
            self._init_dense_l1_tiles()
        else:
            # Build tile↔index mapping
            tile_set = {tile_id for _, _, tile_id in ground_list_raw}
            self.tile_list = sorted(tile_set)
            self.tile2idx = {t: i for i, t in enumerate(self.tile_list)}
            self.idx2tile = dict(enumerate(self.tile_list))
            self.idx2tile_center = {
                idx: _normal_tile_center(tile_id)
                for idx, tile_id in self.idx2tile.items()
            }

        # Build tile index → path (dense L1 tiles are dynamic crops)
        self.idx2tile_path = {}
        if not self._use_dense_l1():
            satellite_root = _vigor_m_satellite_root(data_folder)
            for idx, tile_id in self.idx2tile.items():
                city, level, _, _ = _tile_parts(tile_id)
                self.idx2tile_path[idx] = str(satellite_root / city / level / f"{tile_id}.png")
        self.idx2label_cell = {
            idx: self._label_cell_for_tile(idx)
            for idx in self.idx2tile
        }

        # ----- Build ground→tile pairs -----
        pairs = []
        ground_id_list = []
        self.ground_cell = {}
        self.ground_l1_position = {}

        for city, row, tile_id in ground_list_raw:
            if self._use_dense_l1():
                tile_idx = self._dense_l1_idx_for_latlon(city, float(row["lat"]), float(row["lon"]))
            else:
                tile_idx = self.tile2idx[tile_id]
            pair_idx = len(pairs)  # index into pairs list = ground index
            pairs.append((pair_idx, tile_idx))
            ground_id_list.append((city, row["ground"], _ground_path_from_row(row, data_folder, city)))
            self.ground_cell[pair_idx] = self._fine_cell_for_row(city, row)
            if self._use_dense_l1():
                self.ground_l1_position[pair_idx] = self._l1_position_for_latlon(
                    city, float(row["lat"]), float(row["lon"])
                )

        self.pairs = pairs
        self.idx2ground = dict(enumerate(ground_id_list))
        self.idx2ground_path = {}
        for idx, (_, _, ground_path) in enumerate(ground_id_list):
            self.idx2ground_path[idx] = ground_path

        # For a unique tile_id we can have multiple ground views as gt
        self.idx2pairs = defaultdict(list)
        for pair in self.pairs:
            self.idx2pairs[pair[1]].append(pair)

        # Labels: each ground's positive tile index
        self.label = np.array([p[1] for p in self.pairs], dtype=np.int64).reshape(-1, 1)
        # Pad to 4 columns for compatibility with eval (no near-positives for now)
        self.label = np.pad(self.label, ((0, 0), (0, 3)), mode='edge')

        self.samples = copy.deepcopy(self.pairs)

        # Build L1 parent mapping (for parent-constrained sampling)
        self._build_parent_mapping()

        print(f"VigorMDatasetTrain (level={data_level}):")
        print(f"  Cities: {self.cities}")
        print(f"  Ground images: {len(self.pairs)}")
        print(f"  Satellite reference crops: {len(self.tile_list)}")
        print(f"  Labels with ground images: {len(self.idx2pairs)}")
        if self._use_dense_l1():
            print(
                "  Dense L1 stride: "
                f"fraction={self.satellite_stride_fraction:g}, axis={self.dense_axis}, "
                f"crop={self.dense_crop_px}px, stride={self.dense_stride_px}px"
            )
        if hasattr(self, 'l1_parents'):
            print(f"  L1 parents: {len(self.l1_parents)}")

    def _use_dense_l1(self):
        return self.data_level == "L1" and self.satellite_stride_fraction is not None

    def _init_dense_l1_tiles(self):
        """Create the overlapping dense-L1 gallery and projected tile centers."""
        stride_fraction = float(self.satellite_stride_fraction)
        if not (0.0 < stride_fraction <= 1.0):
            raise ValueError("satellite_stride_fraction must be in (0, 1].")
        dense_axis_float = 4 / stride_fraction
        self.dense_axis = int(round(dense_axis_float))
        if not math.isclose(self.dense_axis, dense_axis_float, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "satellite_stride_fraction must divide the L1 4x4 grid cleanly. "
                f"Got fraction={stride_fraction}."
            )
        self.city_bounds = _load_city_bounds(self.data_folder)
        self.city_mosaics = {}
        self.dense_crop_px = 768
        self.dense_stride_px = int(round(self.dense_crop_px * stride_fraction))

        self.tile_list = []
        self.idx2tile = {}
        self.tile2idx = {}
        self.idx2tile_center = {}
        self.idx2tile_cover_cell = {}
        self.idx2tile_bounds = {}
        idx = 0
        for city in self.cities:
            self.city_mosaics[city] = _build_city_l1_mosaic(self.data_folder, city)
            for row in range(self.dense_axis):
                for col in range(self.dense_axis):
                    tile_id = f"{city}_L1_s{stride_fraction:g}_r{row:02d}_c{col:02d}"
                    self.tile_list.append(tile_id)
                    self.idx2tile[idx] = tile_id
                    self.tile2idx[tile_id] = idx
                    self.idx2tile_center[idx] = (
                        _city_axis_offset(city) + (col + 0.5) * stride_fraction,
                        (row + 0.5) * stride_fraction,
                        4,
                    )
                    center_col = (col + 0.5) * stride_fraction
                    center_row = (row + 0.5) * stride_fraction
                    min_col_cont = max(0.0, center_col - 0.5)
                    max_col_cont = min(4.0, center_col + 0.5)
                    min_row_cont = max(0.0, center_row - 0.5)
                    max_row_cont = min(4.0, center_row + 0.5)
                    min_col = math.floor(min_col_cont)
                    max_col = math.ceil(max_col_cont) - 1
                    min_row = math.floor(min_row_cont)
                    max_row = math.ceil(max_row_cont) - 1
                    self.idx2tile_cover_cell[idx] = (
                        city,
                        max(0, min(3, min_row)),
                        max(0, min(3, max_row)),
                        max(0, min(3, min_col)),
                        max(0, min(3, max_col)),
                    )
                    self.idx2tile_bounds[idx] = (
                        city,
                        min_row_cont,
                        max_row_cont,
                        min_col_cont,
                        max_col_cont,
                    )
                    idx += 1

    def _l1_position_for_latlon(self, city, lat, lon):
        min_lon, min_lat, max_lon, max_lat = self.city_bounds[city]
        col = (lon - min_lon) / (max_lon - min_lon) * 4.0
        row = (max_lat - lat) / (max_lat - min_lat) * 4.0
        eps = 1e-7
        return (
            city,
            max(0.0, min(4.0 - eps, row)),
            max(0.0, min(4.0 - eps, col)),
        )

    def _dense_l1_idx_for_latlon(self, city, lat, lon):
        _, row_pos, col_pos = self._l1_position_for_latlon(city, lat, lon)
        col = int(round(col_pos / float(self.satellite_stride_fraction) - 0.5))
        row = int(round(row_pos / float(self.satellite_stride_fraction) - 0.5))
        col = max(0, min(self.dense_axis - 1, col))
        row = max(0, min(self.dense_axis - 1, row))
        offset = self.cities.index(city) * self.dense_axis * self.dense_axis
        return offset + row * self.dense_axis + col

    def _fine_cell_for_row(self, city, row):
        # L3 is the finest VIGOR-M level currently available.
        tile_id = row["L3"] if "L3" in row and pd.notna(row["L3"]) else row[self.data_level]
        try:
            _, _, r, c = _tile_parts(tile_id)
        except Exception:
            r = c = -1
        return (city, int(r), int(c))

    def _label_cell_for_tile(self, tile_idx):
        tile_id = self.idx2tile[int(tile_idx)]
        city = tile_id.split("_", 1)[0]
        if self._use_dense_l1():
            center_x, center_y, _axis = self.idx2tile_center[int(tile_idx)]
            local_col = center_x - _city_axis_offset(city)
            fine_row = int(center_y * 16)
            fine_col = int(local_col * 16)
            return (
                city,
                max(0, min(63, fine_row)),
                max(0, min(63, fine_col)),
            )

        city, level, row, col = _tile_parts(tile_id)
        axis = 4 ** int(level[1:])
        scale = 64.0 / axis
        fine_row = int((row + 0.5) * scale)
        fine_col = int((col + 0.5) * scale)
        return (
            city,
            max(0, min(63, fine_row)),
            max(0, min(63, fine_col)),
        )

    def _build_parent_mapping(self):
        """Build tile→L1-parent and parent→children mappings for constrained sampling."""
        self.tile_to_parent_l1 = {}     # tile_idx → L1 parent name
        self.parent_to_tiles = defaultdict(list)  # L1 parent → [tile_idx, ...]
        self.parent_to_grounds = defaultdict(list)  # L1 parent → [ground_idx, ...]

        for tile_idx, tile_id in self.idx2tile.items():
            if self._use_dense_l1():
                city = tile_id.split("_", 1)[0]
                parent = self._parent_l1_for_dense_tile(tile_idx)
            else:
                # Parse tile_id like "Chicago_L2_r15_c01" → city, row, col
                city, level, r, c = _tile_parts(tile_id)
                if level == "L1":
                    parent = tile_id
                else:
                    # Compute parent L1: r//div, c//div where each deeper level
                    # subdivides the previous level by 4x4.
                    div = 4 ** (int(level[1:]) - 1)
                    parent = f"{city}_L1_r{r // div:02d}_c{c // div:02d}"
            self.tile_to_parent_l1[tile_idx] = parent
            self.parent_to_tiles[parent].append(tile_idx)

        self.l1_parents = sorted(self.parent_to_tiles.keys())

        # Map ground images to their L1 parent (via the tile they're paired with)
        for ground_idx, tile_idx in self.pairs:
            parent = self.tile_to_parent_l1[tile_idx]
            self.parent_to_grounds[parent].append(ground_idx)

    def _parent_l1_for_dense_tile(self, tile_idx):
        city, min_row, max_row, min_col, max_col = self.idx2tile_cover_cell[tile_idx]
        center_row = (min_row + max_row) // 2
        center_col = (min_col + max_col) // 2
        return f"{city}_L1_r{center_row:02d}_c{center_col:02d}"

    def _dense_l1_crop(self, tile_idx):
        tile_id = self.idx2tile[int(tile_idx)]
        parts = tile_id.split("_")
        city = parts[0]
        row = int(parts[3][1:])
        col = int(parts[4][1:])
        center_x = (col + 0.5) * self.dense_stride_px
        center_y = (row + 0.5) * self.dense_stride_px
        mosaic = self.city_mosaics[city]
        return _crop_with_padding(mosaic, center_x, center_y, self.dense_crop_px, self.dense_crop_px)

    def _is_false_negative(self, ground_idx, tile_idx):
        if not self.strict_cell_conflict:
            return False
        return self.ground_cell.get(int(ground_idx)) == self.idx2label_cell.get(int(tile_idx))

    def _can_add_pair(self, pair, used_ground, used_labels, batch_pairs):
        ground_idx, label = pair
        if ground_idx in used_ground:
            return False
        if label in used_labels:
            return False
        ground_cell = self.ground_cell.get(int(ground_idx))
        label_cell = self.idx2label_cell.get(int(label))
        for other_ground, other_label in batch_pairs:
            other_ground_cell = self.ground_cell.get(int(other_ground))
            other_label_cell = self.idx2label_cell.get(int(other_label))
            if self.strict_cell_conflict:
                if ground_cell is not None and ground_cell == other_ground_cell:
                    return False
                if label_cell is not None and label_cell == other_label_cell:
                    return False
                if ground_cell is not None and ground_cell == other_label_cell:
                    return False
                if label_cell is not None and other_ground_cell == label_cell:
                    return False
            if self._is_false_negative(ground_idx, other_label):
                return False
            if self._is_false_negative(other_ground, label):
                return False
        return True

    def _mark_pair(self, pair, used_ground, used_labels):
        ground_idx, label = pair
        used_ground.add(ground_idx)
        used_labels.add(label)

    def _shuffle_false_negative_free_batches(
        self,
        sim_dict=None,
        neighbour_select=8,
        neighbour_range=16,
    ):
        """Build full batches while rejecting duplicate or overlapping positives."""
        label_to_pairs = {idx: copy.deepcopy(pairs) for idx, pairs in self.idx2pairs.items()}
        for pairs in label_to_pairs.values():
            random.shuffle(pairs)

        batch_size = max(1, int(self.shuffle_batch_size))
        if len(label_to_pairs) < batch_size:
            raise ValueError(
                f"Cannot build false-negative-free batches: "
                f"labels with ground images={len(label_to_pairs)} < batch size={batch_size}. "
                "Use a smaller batch or enable L1 dense stride."
            )

        heap = [
            (-len(pairs), random.random(), label)
            for label, pairs in label_to_pairs.items()
            if pairs
        ]
        heapq.heapify(heap)

        def active_count(label):
            return len(label_to_pairs[label])

        def pop_available_label(exclude):
            skipped = []
            selected = None
            while heap:
                neg_count, tie, label = heapq.heappop(heap)
                count = active_count(label)
                if count <= 0 or -neg_count != count:
                    continue
                if label in exclude:
                    skipped.append((neg_count, tie, label))
                    continue
                selected = label
                break
            for item in skipped:
                heapq.heappush(heap, item)
            return selected

        def push_label(label):
            count = active_count(label)
            if count > 0:
                heapq.heappush(heap, (-count, random.random(), label))

        def hard_candidates(seed, exclude):
            if sim_dict is None:
                return []
            near = list(sim_dict.get(int(seed), []))[:neighbour_range]
            always = near[:neighbour_select // 2]
            random_part = near[neighbour_select // 2:]
            random.shuffle(random_part)
            candidates = always + random_part[:neighbour_select - len(always)]
            candidates = [
                int(label)
                for label in candidates
                if int(label) in label_to_pairs and int(label) not in exclude and active_count(int(label)) > 0
            ]
            candidates.sort(key=lambda label: (-active_count(label), random.random()))
            return candidates

        def take_valid_pair(label, used_ground, used_labels, batch_pairs):
            postponed = []
            chosen = None
            pairs = label_to_pairs[label]
            while pairs:
                pair = pairs.pop()
                if self._can_add_pair(pair, used_ground, used_labels, batch_pairs):
                    chosen = pair
                    break
                postponed.append(pair)
            if postponed:
                pairs[:0] = postponed
            return chosen

        samples = []
        dropped_tail = 0
        while True:
            batch = []
            current_labels = set()
            used_ground = set()
            used_labels = set()
            seed = pop_available_label(current_labels)
            if seed is None:
                break
            candidate_labels = [seed] + hard_candidates(seed, {seed})

            while len(batch) < batch_size:
                if candidate_labels:
                    label = candidate_labels.pop(0)
                else:
                    label = pop_available_label(current_labels)
                    if label is None:
                        break
                if label in current_labels:
                    continue
                pair = take_valid_pair(label, used_ground, used_labels, batch)
                if pair is None:
                    push_label(label)
                    current_labels.add(label)
                    continue
                batch.append(pair)
                current_labels.add(label)
                self._mark_pair(pair, used_ground, used_labels)

            if len(batch) < batch_size:
                for pair in batch:
                    label_to_pairs[pair[1]].append(pair)
                for label in current_labels:
                    push_label(label)
                dropped_tail = sum(len(pairs) for pairs in label_to_pairs.values())
                break

            random.shuffle(batch)
            samples.extend(batch)
            for _, label in batch:
                push_label(label)

        if not samples:
            raise RuntimeError("Could not build any false-negative-free batches.")
        self.samples = samples

        print("\nShuffle Dataset (false-negative-free):")
        print(f"  Original Length: {len(self.pairs)} - Length after Shuffle: {len(self.samples)}")
        print(f"  Dropped tail samples: {dropped_tail}")
        print(f"  Satellite reference crops: {len(self.tile_list)}")
        print(f"  Labels with ground images: {len(self.idx2pairs)}")
        print(f"  Logical batch size: {batch_size}")
        if self._use_dense_l1():
            print(f"  Strict L1-overlap conflict: {self.strict_cell_conflict}")
        if sim_dict is not None:
            print(
                "  Hard sample mining: "
                f"neighbour_select={neighbour_select} neighbour_range={neighbour_range}"
            )

        return

    def shuffle_parent_constrained(self):
        """Build samples where each batch is confined to a single L1 parent's children.

        For each L1 parent, collect all (ground, L2_tile) pairs whose tile falls under
        that parent. Shuffle within each parent, then concatenate into full-batch blocks.
        All negatives within a batch belong to the same L1 region (~16 tiles).
        """
        all_blocks = []  # list of (list of (ground_idx, tile_idx) for one parent)

        for parent in tqdm(self.l1_parents, desc="Constrained shuffle", unit="parent"):
            tiles = self.parent_to_tiles[parent]
            if len(tiles) < 2:
                continue

            # Collect all pairs for tiles under this parent
            parent_pairs = []
            for t in tiles:
                parent_pairs.extend(self.idx2pairs[t])

            if len(parent_pairs) < self.shuffle_batch_size:
                continue

            random.shuffle(parent_pairs)

            # Chunk into full batches
            bs = self.shuffle_batch_size
            for i in range(0, len(parent_pairs) - bs + 1, bs):
                all_blocks.append(parent_pairs[i:i + bs])

        if len(all_blocks) == 0:
            print("  Parent-constrained: no valid blocks, fallback to random")
            random.shuffle(self.samples)
            return

        # Shuffle block order, then flatten
        random.shuffle(all_blocks)
        new_samples = []
        for block in all_blocks:
            new_samples.extend(block)

        # Pad to original length with random global pairs.
        # Keeps len(dataset) stable → DistributedSampler indices stay valid.
        original_len = len(self.pairs)
        if len(new_samples) < original_len:
            backup = copy.deepcopy(self.pairs)
            random.shuffle(backup)
            needed = original_len - len(new_samples)
            new_samples.extend(backup[:needed])

        self.samples = new_samples[:original_len]

        print(f"  Parent-constrained: {len(self.samples)} samples (padded from {sum(len(b) for b in all_blocks)}), "
              f"{len(all_blocks)} constrained batches (per-parent negatives)")

    def __getitem__(self, index):
        idx_ground, idx_tile = self.samples[index]

        # Load query → ground image
        query_img = _read_rgb(self.idx2ground_path[idx_ground])

        # Load reference → satellite tile/crop
        if self._use_dense_l1():
            reference_img = self._dense_l1_crop(idx_tile)
        else:
            ref_path = self.idx2tile_path[idx_tile]
            reference_img = _read_rgb(ref_path)

        # Flip simultaneously
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # Transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # Rotate simultaneously
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            c, h, w = query_img.shape
            shifts = -w // 4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)

        label = torch.tensor(idx_tile, dtype=torch.long)
        return query_img, reference_img, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        """Build full batches with unique positives and no known false negatives."""
        self._shuffle_false_negative_free_batches(
            sim_dict=sim_dict,
            neighbour_select=neighbour_select,
            neighbour_range=neighbour_range,
        )

    def _shuffle_balanced_unique_refs(self, sim_dict=None, neighbour_range=16):
        """Build stable-length batches with unique reference tiles.

        L1 has only 64 reference tiles but tens of thousands of street views.
        A batch cannot contain duplicate reference tiles for diagonal InfoNCE,
        so this sampler balances tile IDs and reuses low-frequency tiles when
        needed while keeping the epoch length equal to the dataset length.
        """
        print("\nShuffle Dataset (balanced unique refs):")

        all_tiles = list(self.idx2pairs.keys())
        batch_size = min(self.shuffle_batch_size, len(all_tiles))
        epoch_len = len(self.pairs)
        num_batches = math.ceil(epoch_len / batch_size)

        tile_queues = {tile: copy.deepcopy(pairs) for tile, pairs in self.idx2pairs.items()}
        for queue in tile_queues.values():
            random.shuffle(queue)

        seed_pool = copy.deepcopy(all_tiles)
        random.shuffle(seed_pool)
        batches = []
        reused = 0

        for _ in tqdm(range(num_batches), desc="Balanced shuffle", unit="batch"):
            if seed_pool:
                seed = seed_pool.pop()
            else:
                seed = random.choice(all_tiles)

            candidates = [seed]
            if sim_dict is not None and seed in sim_dict:
                candidates.extend(sim_dict[seed][:neighbour_range])

            selected = []
            selected_set = set()
            for tile_idx in candidates:
                if tile_idx in self.idx2pairs and tile_idx not in selected_set:
                    selected.append(tile_idx)
                    selected_set.add(tile_idx)
                    if len(selected) >= batch_size:
                        break

            if len(selected) < batch_size:
                remaining = [tile for tile in all_tiles if tile not in selected_set]
                random.shuffle(remaining)
                selected.extend(remaining[:batch_size - len(selected)])

            for tile_idx in selected:
                queue = tile_queues[tile_idx]
                if queue:
                    pair = queue.pop()
                else:
                    pair = random.choice(self.idx2pairs[tile_idx])
                    reused += 1
                batches.append(pair)

        self.samples = batches[:epoch_len]
        unused = sum(len(queue) for queue in tile_queues.values())
        print(f"  Tiles: {len(all_tiles)}, Batch size: {batch_size}")
        print(f"  Original Length: {epoch_len} - Length after Shuffle: {len(self.samples)}")
        print(f"  Reused low-frequency tile samples: {reused}")
        print(f"  Unused high-frequency tile samples: {unused}")


class VigorMDatasetEval(Dataset):
    """VIGOR-M evaluation dataset.

    img_type='reference': returns all unique satellite tiles at the specified level.
    img_type='query':      returns ground images with tile label.
    """

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 same_area=True,
                 data_level="L3",
                 transforms=None,
                 metadata_folder=None,
                 satellite_stride_fraction=None,
                 ):
        super().__init__()

        self.data_folder = data_folder
        self.metadata_folder = metadata_folder
        self.split = split
        self.img_type = img_type
        self.data_level = data_level
        self.satellite_stride_fraction = satellite_stride_fraction
        self.transforms = transforms

        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            if split == "train":
                self.cities = ['NewYork', 'Seattle']
            else:
                self.cities = ['Chicago', 'SanFrancisco']

        # ----- Gather all tile IDs and ground→tile mappings -----
        tile_set = set()
        ground_rows = []

        for city in self.cities:
            csv_split = _vigor_m_city_csv_split(same_area, split)
            csv_path = _vigor_m_split_csv(data_folder, city, csv_split, metadata_folder)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                gname = row["ground"]
                tile_id = row[data_level]
                if pd.notna(tile_id) and tile_id:
                    tile_set.add(tile_id)
                    ground_rows.append((city, row, tile_id))

        if self._use_dense_l1():
            self._init_dense_l1_tiles()
        else:
            self.tile_list = sorted(tile_set)
            self.tile2idx = {t: i for i, t in enumerate(self.tile_list)}
            self.idx2tile = dict(enumerate(self.tile_list))

        self.idx2tile_path = {}
        if not self._use_dense_l1():
            satellite_root = _vigor_m_satellite_root(data_folder)
            for idx, tile_id in self.idx2tile.items():
                parts = tile_id.split("_", 2)
                city = parts[0]
                level = parts[1]
                self.idx2tile_path[idx] = str(satellite_root / city / level / f"{tile_id}.png")

        # Ground images
        self.idx2ground_path = {}
        ground_labels = []
        for i, (city, row, tile_id) in enumerate(ground_rows):
            self.idx2ground_path[i] = _ground_path_from_row(row, data_folder, city)
            if self._use_dense_l1():
                tile_idx = self._dense_l1_idx_for_latlon(city, float(row["lat"]), float(row["lon"]))
            else:
                tile_idx = self.tile2idx[tile_id]
            # [tile_idx, tile_idx, tile_idx, tile_idx] — no near-positives for now
            ground_labels.append([tile_idx, tile_idx, tile_idx, tile_idx])

        self.ground_labels = np.array(ground_labels, dtype=np.int64)

        if self.img_type == "reference":
            if self._use_dense_l1():
                if split == "train":
                    train_labels = sorted({label[0] for label in ground_labels})
                    self.images = train_labels
                    self.label = train_labels
                else:
                    self.images = list(self.idx2tile.keys())
                    self.label = list(self.idx2tile.keys())
            elif split == "train":
                # Only tiles that appear in training ground-truth pairs
                train_tiles = set(t for _, _, t in ground_rows)
                self.images = []
                self.label = []
                for idx, tile_id in self.idx2tile.items():
                    if tile_id in train_tiles:
                        self.images.append(self.idx2tile_path[idx])
                        self.label.append(idx)
            else:
                # All tiles of the cities in this split
                self.images = list(self.idx2tile_path.values())
                self.label = list(self.idx2tile_path.keys())
            self.label = np.array(self.label, dtype=np.int64)

        elif self.img_type == "query":
            self.images = list(self.idx2ground_path.values())
            self.label = self.ground_labels

        else:
            raise ValueError("img_type must be 'query' or 'reference'")

        print(f"VigorMDatasetEval (level={data_level}, split={split}, type={img_type}):")
        print(f"  Images: {len(self.images)}")
        if self._use_dense_l1():
            print(
                "  Dense L1 stride: "
                f"fraction={self.satellite_stride_fraction:g}, axis={self.dense_axis}, "
                f"crop={self.dense_crop_px}px, stride={self.dense_stride_px}px"
            )

    def _use_dense_l1(self):
        return self.data_level == "L1" and self.satellite_stride_fraction is not None

    def _init_dense_l1_tiles(self):
        """Create a deterministic dense-L1 gallery for query/reference evaluation."""
        stride_fraction = float(self.satellite_stride_fraction)
        if not (0.0 < stride_fraction <= 1.0):
            raise ValueError("satellite_stride_fraction must be in (0, 1].")
        dense_axis_float = 4 / stride_fraction
        self.dense_axis = int(round(dense_axis_float))
        if not math.isclose(self.dense_axis, dense_axis_float, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "satellite_stride_fraction must divide the L1 4x4 grid cleanly. "
                f"Got fraction={stride_fraction}."
            )
        self.city_bounds = _load_city_bounds(self.data_folder)
        self.city_mosaics = {}
        self.dense_crop_px = 768
        self.dense_stride_px = int(round(self.dense_crop_px * stride_fraction))

        self.tile_list = []
        self.idx2tile = {}
        self.tile2idx = {}
        self.idx2tile_center = {}
        idx = 0
        for city in self.cities:
            self.city_mosaics[city] = _build_city_l1_mosaic(self.data_folder, city)
            for row in range(self.dense_axis):
                for col in range(self.dense_axis):
                    tile_id = f"{city}_L1_s{stride_fraction:g}_r{row:02d}_c{col:02d}"
                    self.tile_list.append(tile_id)
                    self.idx2tile[idx] = tile_id
                    self.tile2idx[tile_id] = idx
                    self.idx2tile_center[idx] = (
                        _city_axis_offset(city) + (col + 0.5) * stride_fraction,
                        (row + 0.5) * stride_fraction,
                        4,
                    )
                    idx += 1

    def _l1_position_for_latlon(self, city, lat, lon):
        min_lon, min_lat, max_lon, max_lat = self.city_bounds[city]
        col = (lon - min_lon) / (max_lon - min_lon) * 4.0
        row = (max_lat - lat) / (max_lat - min_lat) * 4.0
        eps = 1e-7
        return (
            city,
            max(0.0, min(4.0 - eps, row)),
            max(0.0, min(4.0 - eps, col)),
        )

    def _dense_l1_idx_for_latlon(self, city, lat, lon):
        _, row_pos, col_pos = self._l1_position_for_latlon(city, lat, lon)
        col = int(round(col_pos / float(self.satellite_stride_fraction) - 0.5))
        row = int(round(row_pos / float(self.satellite_stride_fraction) - 0.5))
        col = max(0, min(self.dense_axis - 1, col))
        row = max(0, min(self.dense_axis - 1, row))
        offset = self.cities.index(city) * self.dense_axis * self.dense_axis
        return offset + row * self.dense_axis + col

    def _dense_l1_crop(self, tile_idx):
        tile_id = self.idx2tile[int(tile_idx)]
        parts = tile_id.split("_")
        city = parts[0]
        row = int(parts[3][1:])
        col = int(parts[4][1:])
        center_x = (col + 0.5) * self.dense_stride_px
        center_y = (row + 0.5) * self.dense_stride_px
        mosaic = self.city_mosaics[city]
        return _crop_with_padding(mosaic, center_x, center_y, self.dense_crop_px, self.dense_crop_px)

    def __getitem__(self, index):
        label = self.label[index]

        if self.img_type == "reference" and self._use_dense_l1():
            img = self._dense_l1_crop(int(label))
        else:
            img_path = self.images[index]
            img = _read_rgb(img_path)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.images)
