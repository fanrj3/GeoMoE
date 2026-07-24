import ast
import copy
import heapq
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


GROUND_IMAGE_SUFFIX = "_undistorted.jpg"
DENSE_TILE_ID_PREFIX = "dense"

JUSTZOOMIN_LEVELS = {
    "L1": {"sequence_depth": 1, "satellite_crop_meters": 2500.0},
    "L2": {"sequence_depth": 2, "satellite_crop_meters": 625.0},
    "L3": {"sequence_depth": 3, "satellite_crop_meters": 156.25},
    "L4": {"sequence_depth": 4, "satellite_crop_meters": 39.0625},
}


def resolve_justzoomin_level(data_level):
    level = str(data_level).upper()
    if level not in JUSTZOOMIN_LEVELS:
        valid = ", ".join(JUSTZOOMIN_LEVELS)
        raise ValueError(f"data_level must be one of: {valid}")
    return dict(JUSTZOOMIN_LEVELS[level])


def resolve_justzoomin_levels(data_levels):
    if data_levels is None:
        return list(JUSTZOOMIN_LEVELS.keys())
    return [str(level).upper() for level in data_levels]


def build_spatial_neighbor_dict(idx2tile_center, labels=None, top_k=128, block_size=512):
    """Build a VIGOR-style hard-negative dict from crop center distances."""
    if labels is None:
        labels = sorted(idx2tile_center)
    else:
        labels = sorted(int(label) for label in labels)

    n = len(labels)
    k = min(int(top_k), max(0, n - 1))
    nearest = {label: [] for label in labels}
    if k == 0:
        return nearest

    coords = np.array([idx2tile_center[label] for label in labels], dtype=np.float32)
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


def default_satellite_cache_dir(data_folder):
    return Path(__file__).resolve().parents[2] / "data" / "justzoomin_satellite_cache"


def _format_cache_value(value):
    return str(value).replace("-", "m").replace(".", "p")


def _parse_sequence(value):
    if isinstance(value, str):
        return tuple(ast.literal_eval(value))
    return tuple(value)


def _epsg26985_from_latlon(lat_deg, lon_deg):
    """NAD83 / Maryland (EPSG:26985) forward Lambert Conformal Conic."""
    semi_major = 6378137.0
    inv_flattening = 298.257222101
    flattening = 1.0 / inv_flattening
    eccentricity = math.sqrt(2.0 * flattening - flattening * flattening)

    lat_origin = math.radians(37.0 + 40.0 / 60.0)
    lon_origin = math.radians(-77.0)
    lat_1 = math.radians(38.3)
    lat_2 = math.radians(39.45)
    false_easting = 400000.0
    false_northing = 0.0

    def m(phi):
        return math.cos(phi) / math.sqrt(1.0 - eccentricity**2 * math.sin(phi) ** 2)

    def t(phi):
        sin_phi = math.sin(phi)
        ratio = (1.0 - eccentricity * sin_phi) / (1.0 + eccentricity * sin_phi)
        return math.tan(math.pi / 4.0 - phi / 2.0) / (ratio ** (eccentricity / 2.0))

    n = (math.log(m(lat_1)) - math.log(m(lat_2))) / (math.log(t(lat_1)) - math.log(t(lat_2)))
    f = m(lat_1) / (n * t(lat_1) ** n)
    rho_origin = semi_major * f * t(lat_origin) ** n

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    rho = semi_major * f * t(lat) ** n
    theta = n * (lon - lon_origin)

    easting = false_easting + rho * math.sin(theta)
    northing = false_northing + rho_origin - rho * math.cos(theta)
    return easting, northing


class _JustZoomInBase:
    grid_size = 4
    geographic_center_latlon = (38.8936, -77.0116)
    region_bounds_meters = (-3000.0, 7000.0, -5000.0, 5000.0)
    tile_origin_crs = (600.0, 200.0)
    tile_shape_crs = 20.0
    tile_shape_px = 250

    def _init_common(
        self,
        data_folder,
        split,
        sequence_depth,
        satellite_zoom,
        satellite_crop_meters,
        satellite_stride_fraction=None,
        satellite_cache_dir=None,
        satellite_cache_size=384,
    ):
        self.data_folder = Path(data_folder)
        self.split = split
        self.sequence_depth = sequence_depth
        self.satellite_zoom = satellite_zoom
        self.satellite_stride_fraction = satellite_stride_fraction
        self.satellite_cache_dir = Path(satellite_cache_dir) if satellite_cache_dir else None
        self.satellite_cache_size = int(satellite_cache_size)
        self.satellite_root = self.data_folder / "satellite"
        self.ground_root = self.data_folder / "streetview" / "images"

        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if sequence_depth < 1:
            raise ValueError("sequence_depth must be >= 1")

        metadata_name = f"large_area_{split}_map.csv"
        metadata_path = self.data_folder / "metadata" / metadata_name
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        self.initial_center = self._compute_initial_center()
        self.initial_size = self.region_bounds_meters[1] - self.region_bounds_meters[0]
        self.label_cell_meters = self.initial_size / (self.grid_size ** self.sequence_depth)
        if satellite_crop_meters is None:
            satellite_crop_meters = self.tile_shape_crs * (2 ** (-self.satellite_zoom))
        self.satellite_crop_meters = float(satellite_crop_meters)

        self.df = pd.read_csv(metadata_path)
        self.df["image_id"] = self.df["image_id"].astype(str)
        self.df["sequence_tuple"] = self.df["sequence"].map(_parse_sequence)
        self.df["sequence_key"] = self.df["sequence_tuple"].map(lambda seq: tuple(seq[:self.sequence_depth]))

        self.use_dense_satellite_grid = satellite_stride_fraction is not None
        if self.use_dense_satellite_grid:
            self._init_dense_satellite_grid(float(satellite_stride_fraction))
        else:
            self.tile_list = sorted(self.df["sequence_key"].unique())
            self.tile2idx = {tile: idx for idx, tile in enumerate(self.tile_list)}
            self.idx2tile = dict(enumerate(self.tile_list))
            self.idx2tile_center = {
                idx: self._sequence_center(sequence_key)
                for idx, sequence_key in self.idx2tile.items()
            }
        self._init_satellite_cache_paths()

    def _init_satellite_cache_paths(self):
        if self.satellite_cache_dir is None:
            self.idx2satellite_cache_path = None
            return

        stride = (
            "none"
            if self.satellite_stride_fraction is None
            else _format_cache_value(f"{self.satellite_stride_fraction:g}")
        )
        cache_name = (
            f"depth{self.sequence_depth}_"
            f"crop{_format_cache_value(f'{self.satellite_crop_meters:g}')}_"
            f"z{_format_cache_value(self.satellite_zoom)}_"
            f"stride{stride}_"
            f"size{self.satellite_cache_size}"
        )
        self.satellite_cache_path = self.satellite_cache_dir / cache_name
        self.idx2satellite_cache_path = {
            idx: self.satellite_cache_path / f"{self._tile_cache_id(idx)}.png"
            for idx in self.idx2tile
        }

    def _tile_cache_id(self, tile_idx):
        tile_id = self.idx2tile[int(tile_idx)]
        if isinstance(tile_id, tuple):
            return "seq_" + "_".join(f"{int(value):02d}" for value in tile_id)
        return str(tile_id).replace("/", "_").replace(" ", "_")

    def _init_dense_satellite_grid(self, stride_fraction):
        if not (0.0 < stride_fraction <= 1.0):
            raise ValueError("satellite_stride_fraction must be in (0, 1].")

        cells_per_axis = self.grid_size ** self.sequence_depth
        dense_per_axis_float = cells_per_axis / stride_fraction
        dense_per_axis = int(round(dense_per_axis_float))
        if not math.isclose(dense_per_axis, dense_per_axis_float, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "satellite_stride_fraction must divide the level grid cleanly. "
                f"Got cells_per_axis={cells_per_axis}, fraction={stride_fraction}."
            )

        self.dense_cells_per_axis = dense_per_axis
        self.dense_stride_meters = self.initial_size / dense_per_axis

        left = self.initial_center[0] - self.initial_size / 2.0
        top = self.initial_center[1] + self.initial_size / 2.0

        self.tile_list = []
        self.idx2tile_center = {}
        idx = 0
        for row in range(dense_per_axis):
            center_north = top - (row + 0.5) * self.dense_stride_meters
            for col in range(dense_per_axis):
                center_east = left + (col + 0.5) * self.dense_stride_meters
                tile_id = self._dense_tile_id(row, col)
                self.tile_list.append(tile_id)
                self.idx2tile_center[idx] = (center_east, center_north)
                idx += 1

        self.tile2idx = {tile: idx for idx, tile in enumerate(self.tile_list)}
        self.idx2tile = dict(enumerate(self.tile_list))

    def _dense_tile_id(self, row, col):
        return (
            f"{DENSE_TILE_ID_PREFIX}_L{self.sequence_depth}_"
            f"s{self.satellite_stride_fraction:g}_r{row:04d}_c{col:04d}"
        )

    def _dense_tile_idx_for_latlon(self, lat, lon):
        east, north = _epsg26985_from_latlon(float(lat), float(lon))
        left = self.initial_center[0] - self.initial_size / 2.0
        top = self.initial_center[1] + self.initial_size / 2.0

        col = int(round((east - left) / self.dense_stride_meters - 0.5))
        row = int(round((top - north) / self.dense_stride_meters - 0.5))
        col = max(0, min(self.dense_cells_per_axis - 1, col))
        row = max(0, min(self.dense_cells_per_axis - 1, row))
        return row * self.dense_cells_per_axis + col

    def _tile_idx_for_row(self, row):
        if self.use_dense_satellite_grid:
            return self._dense_tile_idx_for_latlon(row["latitude"], row["longitude"])
        return self.tile2idx[row["sequence_key"]]

    def _compute_initial_center(self):
        center_east = (self.region_bounds_meters[0] + self.region_bounds_meters[1]) / 2.0
        center_north = (self.region_bounds_meters[2] + self.region_bounds_meters[3]) / 2.0
        base_east, base_north = _epsg26985_from_latlon(*self.geographic_center_latlon)
        return base_east + center_east, base_north + center_north

    def _sequence_center(self, sequence_key):
        center_east, center_north = self.initial_center
        current_size = self.initial_size

        for patch_index in sequence_key:
            patch_size = current_size / self.grid_size
            row, col = divmod(int(patch_index), self.grid_size)
            offset_east = (col + 0.5) * patch_size - current_size / 2.0
            offset_north = -((row + 0.5) * patch_size - current_size / 2.0)
            center_east += offset_east
            center_north += offset_north
            current_size = patch_size

        return center_east, center_north

    def _load_ground_image(self, image_id):
        path = self.ground_root / f"{image_id}{GROUND_IMAGE_SUFFIX}"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_satellite_crop(self, tile_idx):
        center_east, center_north = self.idx2tile_center[int(tile_idx)]
        return self._read_satellite_crop(center_east, center_north, self.satellite_crop_meters)

    def _load_satellite_image(self, tile_idx):
        if self.idx2satellite_cache_path is None:
            return self._load_satellite_crop(tile_idx)

        path = self.idx2satellite_cache_path[int(tile_idx)]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(
                f"Missing satellite cache image: {path}. "
                "Run prepare_justzoomin_satellite_cache.py for this level/zoom/stride first."
            )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_satellite_crop(self, center_east, center_north, crop_meters):
        crop = None
        zoom = self.satellite_zoom
        while zoom >= -9:
            crop, missing = self._read_satellite_crop_at_zoom(
                center_east, center_north, crop_meters, zoom
            )
            if not missing:
                return crop
            zoom -= 1

        if crop is not None:
            return crop

        raise RuntimeError(
            f"Could not load satellite crop at center=({center_east:.2f}, {center_north:.2f})"
        )

    def _read_satellite_crop_at_zoom(self, center_east, center_north, crop_meters, zoom):
        tile_meters = self.tile_shape_crs * (2 ** (-zoom))
        scale = self.tile_shape_px / tile_meters
        origin_east, origin_north = self.tile_origin_crs

        west = center_east - crop_meters / 2.0
        east = center_east + crop_meters / 2.0
        south = center_north - crop_meters / 2.0
        north = center_north + crop_meters / 2.0

        x_min = math.floor((west - origin_east) / tile_meters)
        x_max = math.floor((east - 1e-9 - origin_east) / tile_meters)
        y_min = math.floor((south - origin_north) / tile_meters)
        y_max = math.floor((north - 1e-9 - origin_north) / tile_meters)

        cols = x_max - x_min + 1
        rows = y_max - y_min + 1
        mosaic = np.zeros(
            (rows * self.tile_shape_px, cols * self.tile_shape_px, 3),
            dtype=np.uint8,
        )

        zoom_root = self.satellite_root / str(zoom)
        missing = False
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tile_path = zoom_root / str(x) / f"{y}.jpg"
                if not tile_path.exists():
                    missing = True
                    continue
                tile = cv2.imread(str(tile_path))
                if tile is None:
                    missing = True
                    continue
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                if tile.shape[0] != self.tile_shape_px or tile.shape[1] != self.tile_shape_px:
                    tile = cv2.resize(tile, (self.tile_shape_px, self.tile_shape_px))

                row = y_max - y
                col = x - x_min
                r0 = row * self.tile_shape_px
                c0 = col * self.tile_shape_px
                mosaic[r0:r0 + self.tile_shape_px, c0:c0 + self.tile_shape_px] = tile

        mosaic_west = origin_east + x_min * tile_meters
        mosaic_north = origin_north + (y_max + 1) * tile_meters
        left = int(round((west - mosaic_west) * scale))
        right = int(round((east - mosaic_west) * scale))
        top = int(round((mosaic_north - north) * scale))
        bottom = int(round((mosaic_north - south) * scale))

        crop = mosaic[top:bottom, left:right]
        if crop.size == 0:
            raise RuntimeError(
                f"Empty satellite crop at center=({center_east:.2f}, {center_north:.2f}), "
                f"zoom={zoom}"
            )
        return crop, missing


class JustZoomInDatasetTrain(_JustZoomInBase, Dataset):
    """JustZoomIn retrieval training dataset.

    Each ground image is paired with a satellite crop centered at the endpoint
    of its zoom-action sequence. The sequence prefix is used as the retrieval
    class label, keeping the interface compatible with the VIGOR-M trainer.
    """

    def __init__(
        self,
        data_folder,
        split="train",
        sequence_depth=4,
        satellite_zoom=-3,
        satellite_crop_meters=None,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
        satellite_stride_fraction=None,
        label_balanced_oversample=False,
        steps_per_epoch=None,
        satellite_cache_dir=None,
        satellite_cache_size=384,
    ):
        super().__init__()
        self._init_common(
            data_folder,
            split,
            sequence_depth,
            satellite_zoom,
            satellite_crop_meters,
            satellite_stride_fraction=satellite_stride_fraction,
            satellite_cache_dir=satellite_cache_dir,
            satellite_cache_size=satellite_cache_size,
        )

        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.label_balanced_oversample = label_balanced_oversample
        self.steps_per_epoch = steps_per_epoch

        self.idx2ground_path = {}
        self.pairs = []
        for ground_idx, row in self.df.iterrows():
            image_id = row["image_id"]
            tile_idx = self._tile_idx_for_row(row)
            self.idx2ground_path[ground_idx] = self.ground_root / f"{image_id}{GROUND_IMAGE_SUFFIX}"
            self.pairs.append((ground_idx, tile_idx))

        self.idx2pairs = defaultdict(list)
        for pair in self.pairs:
            self.idx2pairs[pair[1]].append(pair)

        self.label = np.array([pair[1] for pair in self.pairs], dtype=np.int64).reshape(-1, 1)
        self.label = np.pad(self.label, ((0, 0), (0, 3)), mode="edge")
        self.samples = copy.deepcopy(self.pairs)

        print(f"JustZoomInDatasetTrain (split={split}, depth={sequence_depth}, zoom={satellite_zoom}):")
        print(f"  Ground images: {len(self.pairs)}")
        print(f"  Satellite reference crops: {len(self.tile_list)}")
        print(f"  Labels with ground images: {len(self.idx2pairs)}")
        print(f"  Satellite crop meters: {self.satellite_crop_meters:.2f}")
        if self.use_dense_satellite_grid:
            print(
                "  Dense satellite grid: "
                f"stride_fraction={self.satellite_stride_fraction:g}, "
                f"axis={self.dense_cells_per_axis}, "
                f"stride_meters={self.dense_stride_meters:.2f}"
            )
        if self.idx2satellite_cache_path is not None:
            print(f"  Satellite cache: {self.satellite_cache_path}")

    def __getitem__(self, index):
        idx_ground, idx_tile = self.samples[index]

        query_img = cv2.imread(str(self.idx2ground_path[idx_ground]))
        if query_img is None:
            raise FileNotFoundError(self.idx2ground_path[idx_ground])
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        reference_img = self._load_satellite_image(idx_tile)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            _, _, w = query_img.shape
            query_img = torch.roll(query_img, shifts=-w // 4 * r, dims=2)

        label = torch.tensor(idx_tile, dtype=torch.long)
        return query_img, reference_img, label

    def __len__(self):
        return len(self.samples)

    def _shuffle_unique_label_batches(
        self,
        label_to_pairs,
        batch_size,
        sim_dict=None,
        neighbour_select=8,
        neighbour_range=16,
    ):
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

        def hard_candidates(seed, exclude):
            if sim_dict is None:
                return []
            near = list(sim_dict.get(seed, []))[:neighbour_range]
            always = near[:neighbour_select // 2]
            random_part = near[neighbour_select // 2:]
            random.shuffle(random_part)
            candidates = always + random_part[:neighbour_select - len(always)]
            candidates = [
                label for label in candidates
                if label in label_to_pairs and label not in exclude and active_count(label) > 0
            ]
            # Keep hard negatives useful, but prefer labels that still have enough samples.
            candidates.sort(key=lambda label: (-active_count(label), random.random()))
            return candidates

        samples = []
        while True:
            batch_labels = []
            current = set()

            seed = pop_available_label(current)
            if seed is None:
                break
            batch_labels.append(seed)
            current.add(seed)

            for label in hard_candidates(seed, current):
                batch_labels.append(label)
                current.add(label)
                if len(batch_labels) >= batch_size:
                    break

            while len(batch_labels) < batch_size:
                label = pop_available_label(current)
                if label is None:
                    break
                batch_labels.append(label)
                current.add(label)

            if len(batch_labels) < batch_size:
                for label in batch_labels:
                    count = active_count(label)
                    if count > 0:
                        heapq.heappush(heap, (-count, random.random(), label))
                break

            batch = [label_to_pairs[label].pop() for label in batch_labels]
            random.shuffle(batch)
            samples.extend(batch)

            for label in batch_labels:
                count = active_count(label)
                if count > 0:
                    heapq.heappush(heap, (-count, random.random(), label))

        return samples

    def _shuffle_label_balanced_oversample(
        self,
        label_to_pairs,
        batch_size,
        sim_dict=None,
        neighbour_select=8,
        neighbour_range=16,
    ):
        labels = list(label_to_pairs.keys())
        if len(labels) < batch_size:
            raise ValueError(
                f"Cannot build false-negative-free batches: "
                f"labels with ground images={len(labels)} < batch size={batch_size}."
            )

        pools = {label: copy.deepcopy(pairs) for label, pairs in label_to_pairs.items()}
        for pairs in pools.values():
            random.shuffle(pairs)
        cursors = {label: 0 for label in labels}
        usage = {label: 0 for label in labels}

        total_pairs = sum(len(pairs) for pairs in pools.values())
        steps = self.steps_per_epoch
        if steps is None:
            steps = math.ceil(total_pairs / batch_size)

        def take_pair(label):
            pairs = pools[label]
            if cursors[label] >= len(pairs):
                random.shuffle(pairs)
                cursors[label] = 0
            pair = pairs[cursors[label]]
            cursors[label] += 1
            usage[label] += 1
            return pair

        def balanced_candidates(exclude):
            candidates = [label for label in labels if label not in exclude]
            random.shuffle(candidates)
            candidates.sort(key=lambda label: (usage[label], random.random()))
            return candidates

        samples = []
        for _ in range(steps):
            current_labels = set()
            batch_labels = []

            seed = balanced_candidates(current_labels)[0]
            batch_labels.append(seed)
            current_labels.add(seed)

            if sim_dict is not None:
                near = list(sim_dict.get(seed, []))[:neighbour_range]
                always = near[:neighbour_select // 2]
                random_part = near[neighbour_select // 2:]
                random.shuffle(random_part)
                hard = always + random_part[:neighbour_select - len(always)]
                hard = [
                    label for label in hard
                    if label in pools and label not in current_labels
                ]
                hard_rank = {label: rank for rank, label in enumerate(hard)}
                hard.sort(key=lambda label: (usage[label], hard_rank[label], random.random()))
                for label in hard:
                    if label in pools and label not in current_labels:
                        batch_labels.append(label)
                        current_labels.add(label)
                        if len(batch_labels) >= batch_size:
                            break

            if len(batch_labels) < batch_size:
                for label in balanced_candidates(current_labels):
                    batch_labels.append(label)
                    current_labels.add(label)
                    if len(batch_labels) >= batch_size:
                        break

            batch = [take_pair(label) for label in batch_labels]
            random.shuffle(batch)
            samples.extend(batch)

        return samples

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        label_to_pairs = {idx: copy.deepcopy(pairs) for idx, pairs in self.idx2pairs.items()}
        for pairs in label_to_pairs.values():
            random.shuffle(pairs)

        batch_size = max(1, int(self.shuffle_batch_size))
        if len(label_to_pairs) < batch_size:
            raise ValueError(
                f"Cannot build false-negative-free batches: "
                f"unique labels={len(label_to_pairs)} < batch size={batch_size}. "
                "Use a smaller batch or enable a denser satellite grid."
            )

        if self.label_balanced_oversample:
            samples = self._shuffle_label_balanced_oversample(
                label_to_pairs,
                batch_size,
                sim_dict=sim_dict,
                neighbour_select=neighbour_select,
                neighbour_range=neighbour_range,
            )
        else:
            samples = self._shuffle_unique_label_batches(
                label_to_pairs,
                batch_size,
                sim_dict=sim_dict,
                neighbour_select=neighbour_select,
                neighbour_range=neighbour_range,
            )
        if len(samples) > 0:
            self.samples = samples
        else:
            raise RuntimeError("Could not build any false-negative-free batches.")

        print("\nShuffle Dataset:")
        print(f"  Original Length: {len(self.pairs)} - Length after Shuffle: {len(self.samples)}")
        if self.label_balanced_oversample:
            sample_delta = len(self.samples) - len(self.pairs)
            if sample_delta >= 0:
                print(f"  Oversampled samples: {sample_delta}")
            else:
                print(f"  Short-epoch samples omitted: {-sample_delta}")
        else:
            print(f"  Dropped tail samples: {len(self.pairs) - len(self.samples)}")
        print(f"  Satellite reference crops: {len(self.tile_list)}")
        print(f"  Labels with ground images: {len(self.idx2pairs)}")
        print(f"  Logical batch size: {batch_size}")
        if self.label_balanced_oversample:
            print(f"  Label-balanced oversample: steps={len(self.samples) // batch_size}")
        if sim_dict is not None:
            print(
                "  Hard sample mining: "
                f"neighbour_select={neighbour_select} neighbour_range={neighbour_range}"
            )


class JustZoomInDatasetEval(_JustZoomInBase, Dataset):
    """JustZoomIn retrieval evaluation dataset."""

    def __init__(
        self,
        data_folder,
        split,
        img_type,
        sequence_depth=4,
        satellite_zoom=-3,
        satellite_crop_meters=None,
        transforms=None,
        satellite_stride_fraction=None,
        satellite_cache_dir=None,
        satellite_cache_size=384,
    ):
        super().__init__()
        self._init_common(
            data_folder,
            split,
            sequence_depth,
            satellite_zoom,
            satellite_crop_meters,
            satellite_stride_fraction=satellite_stride_fraction,
            satellite_cache_dir=satellite_cache_dir,
            satellite_cache_size=satellite_cache_size,
        )

        self.img_type = img_type
        self.transforms = transforms

        if img_type == "reference":
            self.images = list(self.idx2tile.keys())
            self.label = np.array(self.images, dtype=np.int64)
        elif img_type == "query":
            self.images = [
                self.ground_root / f"{image_id}{GROUND_IMAGE_SUFFIX}"
                for image_id in self.df["image_id"].tolist()
            ]
            labels = [self._tile_idx_for_row(row) for _, row in self.df.iterrows()]
            labels = np.array(labels, dtype=np.int64).reshape(-1, 1)
            self.label = np.pad(labels, ((0, 0), (0, 3)), mode="edge")
        else:
            raise ValueError("img_type must be 'query' or 'reference'")

        print(f"JustZoomInDatasetEval (split={split}, type={img_type}, depth={sequence_depth}, zoom={satellite_zoom}):")
        print(f"  Images: {len(self.images)}")
        if img_type == "reference":
            print(f"  Satellite crop meters: {self.satellite_crop_meters:.2f}")
            if self.use_dense_satellite_grid:
                print(
                    "  Dense satellite grid: "
                    f"stride_fraction={self.satellite_stride_fraction:g}, "
                    f"axis={self.dense_cells_per_axis}, "
                    f"stride_meters={self.dense_stride_meters:.2f}"
                )
            if self.idx2satellite_cache_path is not None:
                print(f"  Satellite cache: {self.satellite_cache_path}")

    def __getitem__(self, index):
        label = self.label[index]

        if self.img_type == "reference":
            image = self._load_satellite_image(label)
        else:
            image = cv2.imread(str(self.images[index]))
            if image is None:
                raise FileNotFoundError(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.images)


class _AllInLevelDataset(_JustZoomInBase):
    def __init__(
        self,
        data_folder,
        split,
        level,
        satellite_zoom,
        satellite_stride_fraction,
        satellite_cache_dir,
        satellite_cache_size,
    ):
        cfg = resolve_justzoomin_level(level)
        self.level = level
        self._init_common(
            data_folder,
            split,
            cfg["sequence_depth"],
            satellite_zoom,
            cfg["satellite_crop_meters"],
            satellite_stride_fraction=satellite_stride_fraction,
            satellite_cache_dir=satellite_cache_dir,
            satellite_cache_size=satellite_cache_size,
        )


class _JustZoomInAllInMixin:
    def _init_allin_common(
        self,
        data_folder,
        split,
        data_levels,
        satellite_zoom,
        satellite_stride_fractions,
        satellite_cache_dir,
        satellite_cache_size,
        satellite_cache_levels=None,
    ):
        self.data_folder = Path(data_folder)
        self.split = split
        self.data_levels = resolve_justzoomin_levels(data_levels)
        self.satellite_zoom = satellite_zoom
        self.satellite_cache_dir = Path(satellite_cache_dir) if satellite_cache_dir else None
        self.satellite_cache_size = int(satellite_cache_size)
        self.satellite_cache_levels = set(resolve_justzoomin_levels(satellite_cache_levels))
        if satellite_cache_levels is None:
            self.satellite_cache_levels = set(self.data_levels) if self.satellite_cache_dir else set()
        self.ground_root = self.data_folder / "streetview" / "images"

        metadata_path = self.data_folder / "metadata" / f"large_area_{split}_map.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        self.df = pd.read_csv(metadata_path)
        self.df["image_id"] = self.df["image_id"].astype(str)
        self.df["sequence_tuple"] = self.df["sequence"].map(_parse_sequence)

        if isinstance(satellite_stride_fractions, dict):
            stride_map = {
                str(level).upper(): value
                for level, value in satellite_stride_fractions.items()
            }
        else:
            stride_map = {
                level: satellite_stride_fractions
                for level in self.data_levels
            }

        self.level_datasets = {}
        self.level_offsets = {}
        self.global_to_level_local = {}
        self.idx2tile_center = {}
        self.idx2label_l4_cell = {}
        self.idx2label_level = {}
        self.idx2label_local = {}
        offset = 0
        for level in self.data_levels:
            level_cache_dir = self.satellite_cache_dir if level in self.satellite_cache_levels else None
            level_ds = _AllInLevelDataset(
                data_folder,
                split,
                level,
                satellite_zoom,
                stride_map.get(level),
                level_cache_dir,
                satellite_cache_size,
            )
            self.level_datasets[level] = level_ds
            self.level_offsets[level] = offset
            for local_idx in sorted(level_ds.idx2tile):
                global_label = offset + int(local_idx)
                self.global_to_level_local[global_label] = (level, int(local_idx))
                self.idx2tile_center[global_label] = level_ds.idx2tile_center[int(local_idx)]
                self.idx2label_l4_cell[global_label] = self._l4_cell_for_center(
                    *level_ds.idx2tile_center[int(local_idx)]
                )
                self.idx2label_level[global_label] = level
                self.idx2label_local[global_label] = int(local_idx)
            offset += len(level_ds.idx2tile)
        self.num_reference = offset

    def _l4_cell_for_center(self, center_east, center_north):
        base = _JustZoomInBase()
        initial_center = base._compute_initial_center()
        initial_size = base.region_bounds_meters[1] - base.region_bounds_meters[0]
        axis = base.grid_size ** 4
        stride = initial_size / axis
        left = initial_center[0] - initial_size / 2.0
        top = initial_center[1] + initial_size / 2.0
        col = int(round((center_east - left) / stride - 0.5))
        row = int(round((top - center_north) / stride - 0.5))
        col = max(0, min(axis - 1, col))
        row = max(0, min(axis - 1, row))
        return row * axis + col

    def _global_label_for_row(self, level, row):
        level_ds = self.level_datasets[level]
        if level_ds.use_dense_satellite_grid:
            local_idx = level_ds._dense_tile_idx_for_latlon(row["latitude"], row["longitude"])
        else:
            sequence_key = tuple(row["sequence_tuple"][:level_ds.sequence_depth])
            local_idx = level_ds.tile2idx[sequence_key]
        return self.level_offsets[level] + int(local_idx)

    def _load_satellite_by_global_label(self, global_label):
        level, local_idx = self.global_to_level_local[int(global_label)]
        return self.level_datasets[level]._load_satellite_image(local_idx)


class JustZoomInAllInDatasetTrain(_JustZoomInAllInMixin, Dataset):
    """All-in JustZoomIn training dataset over L1/L2/L3/L4 pairs.

    Each ground image contributes one positive pair per level in each epoch.
    The shuffled sample list is built with level-balanced batches and avoids
    obvious false negatives: duplicated ground images, duplicated satellite
    global labels, and labels mapped to the same finest L4 cell.
    """

    def __init__(
        self,
        data_folder,
        split="train",
        data_levels=None,
        satellite_zoom=-3,
        satellite_stride_fractions=None,
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
        satellite_cache_dir=None,
        satellite_cache_size=384,
        satellite_cache_levels=None,
        strict_l4_conflict=True,
        steps_per_epoch=None,
        return_level_id=False,
    ):
        super().__init__()
        self._init_allin_common(
            data_folder,
            split,
            data_levels,
            satellite_zoom,
            satellite_stride_fractions,
            satellite_cache_dir,
            satellite_cache_size,
            satellite_cache_levels=satellite_cache_levels,
        )
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.strict_l4_conflict = strict_l4_conflict
        self.steps_per_epoch = steps_per_epoch
        self.return_level_id = return_level_id
        self.level_to_id = {level: idx for idx, level in enumerate(self.data_levels)}

        self.idx2ground_path = {}
        self.ground_l4_cell = {}
        self.pairs = []
        self.level_to_pairs = {level: [] for level in self.data_levels}
        self.idx2pairs = defaultdict(list)
        self.label_to_pairs = self.idx2pairs

        for ground_idx, row in self.df.iterrows():
            image_id = row["image_id"]
            self.idx2ground_path[ground_idx] = self.ground_root / f"{image_id}{GROUND_IMAGE_SUFFIX}"
            self.ground_l4_cell[ground_idx] = tuple(row["sequence_tuple"][:4])
            for level in self.data_levels:
                label = self._global_label_for_row(level, row)
                sample = (int(ground_idx), int(label), level)
                self.pairs.append(sample)
                self.level_to_pairs[level].append(sample)
                self.idx2pairs[int(label)].append(sample)

        self.label = np.array([sample[1] for sample in self.pairs], dtype=np.int64).reshape(-1, 1)
        self.label = np.pad(self.label, ((0, 0), (0, 3)), mode="edge")
        self.samples = copy.deepcopy(self.pairs)

        print(f"JustZoomInAllInDatasetTrain (split={split}, levels={','.join(self.data_levels)}, zoom={satellite_zoom}):")
        print(f"  Ground images: {len(self.df)}")
        print(f"  Training pairs per epoch target: {len(self.pairs)}")
        print(f"  Satellite reference crops: {self.num_reference}")
        for level in self.data_levels:
            ds = self.level_datasets[level]
            stride = ds.satellite_stride_fraction
            stride_text = "none" if stride is None else f"{stride:g}"
            print(
                f"  {level}: refs={len(ds.idx2tile)} pairs={len(self.level_to_pairs[level])} "
                f"crop={ds.satellite_crop_meters:.2f} stride={stride_text}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        idx_ground, global_label, level = self.samples[index]

        query_img = cv2.imread(str(self.idx2ground_path[idx_ground]))
        if query_img is None:
            raise FileNotFoundError(self.idx2ground_path[idx_ground])
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        reference_img = self._load_satellite_by_global_label(global_label)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            _, _, w = query_img.shape
            query_img = torch.roll(query_img, shifts=-w // 4 * r, dims=2)

        label = torch.tensor(global_label, dtype=torch.long)
        if self.return_level_id:
            level_id = torch.tensor(self.level_to_id[level], dtype=torch.long)
            return query_img, reference_img, label, level_id
        return query_img, reference_img, label

    def _batch_quota(self, batch_size):
        levels = list(self.data_levels)
        base = batch_size // len(levels)
        rem = batch_size % len(levels)
        return {level: base + (idx < rem) for idx, level in enumerate(levels)}

    def _can_add_sample(self, sample, used_ground, used_labels, used_l4_cells):
        ground_idx, label, _level = sample
        if ground_idx in used_ground:
            return False
        if label in used_labels:
            return False
        if self.strict_l4_conflict:
            cell = self.idx2label_l4_cell[int(label)]
            if cell in used_l4_cells:
                return False
        return True

    def _mark_sample(self, sample, used_ground, used_labels, used_l4_cells):
        ground_idx, label, _level = sample
        used_ground.add(ground_idx)
        used_labels.add(label)
        if self.strict_l4_conflict:
            used_l4_cells.add(self.idx2label_l4_cell[int(label)])

    def _hard_ordered_labels(self, seed_label, sim_dict, neighbour_select, neighbour_range):
        if sim_dict is None:
            return []
        hard = list(sim_dict.get(int(seed_label), []))[:neighbour_range]
        always = hard[:neighbour_select // 2]
        random_part = hard[neighbour_select // 2:]
        random.shuffle(random_part)
        hard = always + random_part[:neighbour_select - len(always)]
        return [
            int(label)
            for label in hard
            if int(label) in self.idx2pairs
        ]

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        batch_size = max(1, int(self.shuffle_batch_size))
        quota = self._batch_quota(batch_size)

        pools = {
            level: copy.deepcopy(pairs)
            for level, pairs in self.level_to_pairs.items()
        }
        for pairs in pools.values():
            random.shuffle(pairs)
        cursors = {level: 0 for level in self.data_levels}
        deferred = {level: [] for level in self.data_levels}
        consumed = {level: set() for level in self.data_levels}

        natural_steps = min(
            len(pools[level]) // max(1, quota[level])
            for level in self.data_levels
            if quota[level] > 0
        )
        fixed_steps = self.steps_per_epoch is not None
        steps = self.steps_per_epoch or natural_steps
        replacement_mode = fixed_steps or steps > 0
        reuse_cycles = Counter()

        def next_candidate(level):
            if deferred[level]:
                while deferred[level]:
                    sample = deferred[level].pop()
                    if (sample[0], sample[1]) not in consumed[level]:
                        return sample
            while True:
                while cursors[level] < len(pools[level]):
                    sample = pools[level][cursors[level]]
                    cursors[level] += 1
                    if (sample[0], sample[1]) not in consumed[level]:
                        return sample
                if not replacement_mode or not pools[level] or not consumed[level]:
                    return None
                consumed[level].clear()
                random.shuffle(pools[level])
                cursors[level] = 0
                reuse_cycles[level] += 1
            return None

        def find_sample_for_label(level, label, used_ground, used_labels, used_l4_cells):
            candidates = self.idx2pairs.get(int(label), [])
            candidates = [sample for sample in candidates if sample[2] == level]
            random.shuffle(candidates)
            for sample in candidates:
                if (sample[0], sample[1]) in consumed[level]:
                    continue
                if self._can_add_sample(sample, used_ground, used_labels, used_l4_cells):
                    return sample
            return None

        def add_to_batch(batch, sample, batch_level_counts, used_ground, used_labels, used_l4_cells):
            batch.append(sample)
            consumed[sample[2]].add((sample[0], sample[1]))
            batch_level_counts[sample[2]] += 1
            self._mark_sample(sample, used_ground, used_labels, used_l4_cells)

        def rescue_sample(level, used_ground, used_labels, used_l4_cells):
            pool = pools[level]
            if not pool:
                return None
            passes = 3 if replacement_mode else 1
            for pass_idx in range(passes):
                scanned = 0
                while scanned < len(pool):
                    if cursors[level] >= len(pool):
                        if not replacement_mode:
                            return None
                        consumed[level].clear()
                        random.shuffle(pool)
                        cursors[level] = 0
                        reuse_cycles[level] += 1
                    sample = pool[cursors[level]]
                    cursors[level] += 1
                    scanned += 1
                    if (sample[0], sample[1]) in consumed[level]:
                        continue
                    if self._can_add_sample(sample, used_ground, used_labels, used_l4_cells):
                        return sample
                if replacement_mode and pass_idx + 1 < passes:
                    consumed[level].clear()
                    random.shuffle(pool)
                    cursors[level] = 0
                    reuse_cycles[level] += 1
            return None

        samples = []
        short_batches = 0
        level_counts = {level: 0 for level in self.data_levels}
        for _ in range(steps):
            batch = []
            used_ground = set()
            used_labels = set()
            used_l4_cells = set()
            postponed = {level: [] for level in self.data_levels}
            batch_level_counts = Counter()

            for level in self.data_levels:
                target = quota[level]
                seed = None
                tries = 0
                max_tries = max(1000, len(pools[level]) * 2, batch_size * 20)
                while batch_level_counts[level] < target and seed is None:
                    tries += 1
                    if tries > max_tries:
                        break

                    sample = next_candidate(level)
                    if sample is None:
                        break
                    chosen = None
                    if self._can_add_sample(sample, used_ground, used_labels, used_l4_cells):
                        chosen = sample
                    else:
                        postponed[level].append(sample)

                    if chosen is None:
                        continue

                    seed = chosen
                    add_to_batch(batch, chosen, batch_level_counts, used_ground, used_labels, used_l4_cells)

                if seed is not None and sim_dict is not None:
                    for hard_label in self._hard_ordered_labels(
                        seed[1], sim_dict, neighbour_select, neighbour_range
                    ):
                        hard_level = self.idx2label_level[int(hard_label)]
                        if hard_level != level:
                            continue
                        if batch_level_counts[level] >= target:
                            break
                        chosen = find_sample_for_label(
                            level, hard_label, used_ground, used_labels, used_l4_cells
                        )
                        if chosen is None:
                            continue
                        add_to_batch(batch, chosen, batch_level_counts, used_ground, used_labels, used_l4_cells)

                while batch_level_counts[level] < target:
                    tries += 1
                    if tries > max_tries:
                        break

                    sample = next_candidate(level)
                    if sample is None:
                        break
                    if self._can_add_sample(sample, used_ground, used_labels, used_l4_cells):
                        add_to_batch(batch, sample, batch_level_counts, used_ground, used_labels, used_l4_cells)
                    else:
                        postponed[level].append(sample)

            if len(batch) < batch_size:
                for level in self.data_levels:
                    tries = 0
                    max_tries = max(1000, len(pools[level]) * 2, batch_size * 20)
                    while len(batch) < batch_size and batch_level_counts[level] < quota[level]:
                        tries += 1
                        if tries > max_tries:
                            break
                        sample = next_candidate(level)
                        if sample is None:
                            break
                        if self._can_add_sample(sample, used_ground, used_labels, used_l4_cells):
                            batch.append(sample)
                            consumed[sample[2]].add((sample[0], sample[1]))
                            batch_level_counts[sample[2]] += 1
                            self._mark_sample(sample, used_ground, used_labels, used_l4_cells)
                        else:
                            postponed[level].append(sample)

            if len(batch) < batch_size and replacement_mode:
                for level in self.data_levels:
                    while batch_level_counts[level] < quota[level]:
                        sample = rescue_sample(level, used_ground, used_labels, used_l4_cells)
                        if sample is None:
                            break
                        add_to_batch(batch, sample, batch_level_counts, used_ground, used_labels, used_l4_cells)

            if len(batch) < batch_size:
                for level in self.data_levels:
                    deferred[level].extend(reversed(postponed[level]))
                short_batches += 1
                if not batch:
                    break
                break

            random.shuffle(batch)
            samples.extend(batch)
            for sample in batch:
                level_counts[sample[2]] += 1
            for level in self.data_levels:
                deferred[level].extend(reversed(postponed[level]))

        if not samples:
            raise RuntimeError("Could not build any all-in batches.")
        self.samples = samples

        print("\nShuffle All-In Dataset:")
        print(f"  Original target pairs: {len(self.pairs)} - Length after Shuffle: {len(self.samples)}")
        print(f"  Logical batch size: {batch_size}  Steps: {len(self.samples) // batch_size}")
        print(f"  Level quota per batch: {quota}")
        print(f"  Level sampled counts: {level_counts}")
        print(f"  Strict L4-cell conflict: {self.strict_l4_conflict}")
        print(f"  Short batches stopped: {short_batches}")
        if replacement_mode and reuse_cycles:
            print(f"  Fixed-step sample reuse cycles: {dict(reuse_cycles)}")
        if sim_dict is not None:
            print(
                "  Hard sample mining: "
                f"per-level neighbour_select={neighbour_select} neighbour_range={neighbour_range}"
            )


class JustZoomInAllInDatasetEval(_JustZoomInAllInMixin, Dataset):
    """All-in evaluation dataset with global labels across levels."""

    def __init__(
        self,
        data_folder,
        split,
        img_type,
        data_levels=None,
        satellite_zoom=-3,
        satellite_stride_fractions=None,
        transforms=None,
        satellite_cache_dir=None,
        satellite_cache_size=384,
        satellite_cache_levels=None,
        query_level="L4",
    ):
        super().__init__()
        self._init_allin_common(
            data_folder,
            split,
            data_levels,
            satellite_zoom,
            satellite_stride_fractions,
            satellite_cache_dir,
            satellite_cache_size,
            satellite_cache_levels=satellite_cache_levels,
        )
        self.img_type = img_type
        self.transforms = transforms
        self.query_level = str(query_level).upper()

        if img_type == "reference":
            self.images = list(range(self.num_reference))
            self.label = np.array(self.images, dtype=np.int64)
        elif img_type == "query":
            self.images = [
                self.ground_root / f"{image_id}{GROUND_IMAGE_SUFFIX}"
                for image_id in self.df["image_id"].tolist()
            ]
            labels = [
                self._global_label_for_row(self.query_level, row)
                for _, row in self.df.iterrows()
            ]
            labels = np.array(labels, dtype=np.int64).reshape(-1, 1)
            self.label = np.pad(labels, ((0, 0), (0, 3)), mode="edge")
        else:
            raise ValueError("img_type must be 'query' or 'reference'")

        print(f"JustZoomInAllInDatasetEval (split={split}, type={img_type}, levels={','.join(self.data_levels)}, zoom={satellite_zoom}):")
        print(f"  Images: {len(self.images)}")
        print(f"  Satellite reference crops: {self.num_reference}")
        if img_type == "query":
            print(f"  Query positive level: {self.query_level}")

    def __getitem__(self, index):
        label = self.label[index]

        if self.img_type == "reference":
            image = self._load_satellite_by_global_label(int(label))
        else:
            image = cv2.imread(str(self.images[index]))
            if image is None:
                raise FileNotFoundError(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.images)
