#!/usr/bin/env python3
"""Extract VIGOR-M B11/E5 features for fixed-beam PRC evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "geomoe").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geomoe.datasets.vigor_m import VigorMDatasetEval  # noqa: E402
from geomoe.model import LevelMoEFFNTimmModel  # noqa: E402
from geomoe.transforms import get_transforms_val  # noqa: E402
from geomoe.utils import setup_system  # noqa: E402


CHECKPOINT = PROJECT_ROOT / "weights" / "vigor_m" / "geomoe_b11_e5_e60.pth"
CHECKPOINT_SHA256 = "13d75e2a456e346138e3aac62b707739f800b77fc3f7648492862d72191a3463"
CHECKPOINT_SIZE = 420_006_033
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
LEVELS = ("L1", "L2", "L3")
EXPECTED_TEST_QUERIES = 37_789
EXPECTED_TEST_REFS = {"L1": 1_024, "L2": 998, "L3": 10_818}
BUNDLE_SCHEMA = "vigorm-b11-e5-adaptive-search-features-v1"


class IndexedDataset(Dataset):
    """Attach stable dataset indices so feature batches can be reordered."""

    def __init__(self, dataset: Dataset):
        """Wrap a query or reference dataset."""
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped dataset size."""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Return the image together with its original dataset index."""
        image, _ = self.dataset[index]
        return image, int(index)


def parse_args() -> argparse.Namespace:
    """Parse feature-preparation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-folder", type=Path, default=Path("data/VIGOR-M"))
    parser.add_argument(
        "--metadata-folder",
        type=Path,
        default=Path("data/VIGOR-M/metadata"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def digest_strings(values) -> str:
    """Create an order-sensitive digest for an identifier sequence."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def digest_tensor(value: torch.Tensor) -> str:
    """Hash tensor contents together with shape and dtype metadata."""
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def checkpoint_meta(checkpoint: Path) -> dict[str, Any]:
    """Record checkpoint identity used to invalidate stale feature caches."""
    stat = checkpoint.stat()
    if stat.st_size != CHECKPOINT_SIZE:
        raise RuntimeError(f"Checkpoint size mismatch: {stat.st_size}")
    sha256 = file_sha256(checkpoint)
    if sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {sha256}")
    return {
        "path": str(checkpoint.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256,
    }


def load_cache(path: Path, expected_items: int | None = None) -> tuple[torch.Tensor, dict]:
    """Load and validate a feature cache produced by this script."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"meta", "features"}:
        raise RuntimeError(f"Malformed feature cache: {path}")
    features = payload["features"]
    if features.dtype != torch.float32 or features.ndim != 2 or features.shape[1] != 768:
        raise RuntimeError(f"Invalid feature shape in {path}: {features.shape}")
    if expected_items is not None and features.shape[0] != expected_items:
        raise RuntimeError(f"Unexpected items in {path}: {features.shape[0]}")
    if not torch.isfinite(features).all():
        raise RuntimeError(f"Non-finite features in {path}")
    norm_error = float((torch.linalg.vector_norm(features, dim=1) - 1.0).abs().max())
    if norm_error > 1e-3:
        raise RuntimeError(f"Non-normalized cache {path}: {norm_error}")
    if payload["meta"]["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError(f"Cache checkpoint mismatch: {path}")
    return features.contiguous(), payload["meta"]


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Publish a Torch payload only after its temporary file is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_model(device: torch.device, checkpoint: Path):
    """Construct the released GeoMoE encoder and restore its weights."""
    model = LevelMoEFFNTimmModel(
        MODEL_NAME,
        pretrained=False,
        img_size=384,
        levels=LEVELS,
        moe_start_block=11,
        num_experts=5,
        top_k=2,
        router_jitter=0.0,
        router_condition="none",
        expert_layout="routed",
        default_level="L3",
    )
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")
    return model.to(device).eval()


def build_transforms(model):
    """Build deterministic satellite and panorama evaluation transforms."""
    config = model.get_config()
    satellite, pano = get_transforms_val(
        (384, 384),
        (432, 768),
        mean=config["mean"],
        std=config["std"],
        ground_cutting=0,
    )
    meta = {
        "satellite_resize_hw": [384, 384],
        "pano_resize_hw": [432, 768],
        "mean": [float(value) for value in config["mean"]],
        "std": [float(value) for value in config["std"]],
        "ground_cutting": 0,
    }
    return satellite, pano, meta


def make_dataset(
    args: argparse.Namespace,
    split: str,
    level: str,
    img_type: str,
    transform,
    *,
    dense_l1: bool,
):
    """Instantiate one VIGOR-M query/reference dataset view."""
    return VigorMDatasetEval(
        data_folder=str(args.data_folder),
        split=split,
        img_type=img_type,
        same_area=True,
        data_level=level,
        transforms=transform,
        metadata_folder=str(args.metadata_folder),
        satellite_stride_fraction=0.25 if dense_l1 and level == "L1" else None,
    )


def make_loader(dataset: Dataset, args: argparse.Namespace, device: torch.device):
    """Create a deterministic, index-preserving feature loader."""
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(IndexedDataset(dataset), **kwargs)


@torch.inference_mode()
def extract_features(model, dataset: Dataset, args, device: torch.device, description: str, head: str):
    """Encode a dataset and restore canonical order after batching."""
    features = []
    indices = []
    for images, batch_indices in tqdm(
        make_loader(dataset, args, device), desc=description
    ):
        images = images.to(device, non_blocking=device.type == "cuda")
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = F.normalize(model(images, levels=head), dim=-1)
        features.append(output.detach().cpu().float())
        indices.append(torch.as_tensor(batch_indices).long().cpu())
    actual = torch.cat(indices)
    if not torch.equal(actual, torch.arange(len(dataset), dtype=torch.long)):
        raise RuntimeError(f"Feature order changed for {description}")
    result = torch.cat(features).contiguous()
    if not torch.isfinite(result).all():
        raise RuntimeError(f"Non-finite features for {description}")
    return result


def build_protocol_datasets(args, split: str, satellite_transform, pano_transform):
    """Build the three hierarchy levels for a split with aligned queries."""
    query = {
        level: make_dataset(
            args,
            split,
            level,
            "query",
            pano_transform,
            dense_l1=True,
        )
        for level in LEVELS
    }
    query_images = list(query["L3"].images)
    for level in LEVELS:
        if list(query[level].images) != query_images:
            raise RuntimeError(f"Query order differs at {level}")

    reference = {
        "L1": make_dataset(args, split, "L1", "reference", satellite_transform, dense_l1=True),
        "L2": make_dataset(args, split, "L2", "reference", satellite_transform, dense_l1=False),
        "L3": make_dataset(args, split, "L3", "reference", satellite_transform, dense_l1=False),
    }
    return query, reference, query_images


def main() -> None:
    """Prepare validated VIGOR-M feature bundles for downstream evaluation."""
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data_folder = args.data_folder.expanduser().resolve()
    args.metadata_folder = args.metadata_folder.expanduser().resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Invalid batch/worker settings")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    setup_system(seed=1, cudnn_benchmark=True, cudnn_deterministic=False)
    ckpt = checkpoint_meta(args.checkpoint)
    model = build_model(device, args.checkpoint)
    satellite_transform, pano_transform, preprocess = build_transforms(model)

    query, reference, query_images = build_protocol_datasets(
        args, args.split, satellite_transform, pano_transform
    )
    query_count = len(query_images)
    if args.split == "test" and query_count != EXPECTED_TEST_QUERIES:
        raise RuntimeError(f"Unexpected test query count: {query_count}")

    full_dense_tiles = list(query["L1"].tile_list)
    if len(full_dense_tiles) != EXPECTED_TEST_REFS["L1"]:
        raise RuntimeError(f"Unexpected full dense-L1 gallery: {len(full_dense_tiles)}")
    if list(reference["L1"].tile_list) != full_dense_tiles:
        raise RuntimeError("Dense-L1 query/reference gallery order differs")
    query_features = extract_features(
        model, query["L3"], args, device, f"B11/E5 {args.split} queries", "L3"
    )
    ref_features = {
        level: extract_features(
            model,
            reference[level],
            args,
            device,
            f"B11/E5 {args.split} {level} refs",
            level,
        )
        for level in LEVELS
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tile_lists = {
        "L1": full_dense_tiles,
        "L2": list(reference["L2"].tile_list),
        "L3": list(reference["L3"].tile_list),
    }
    labels = {
        level: torch.from_numpy(query[level].ground_labels.copy()).long()
        for level in LEVELS
    }
    for level in LEVELS:
        if labels[level].shape != (query_count, 4):
            raise RuntimeError(f"{level} label shape mismatch: {labels[level].shape}")
        if ref_features[level].shape != (len(tile_lists[level]), 768):
            raise RuntimeError(
                f"{level} reference feature mismatch: {ref_features[level].shape}"
            )
        if int(labels[level][:, 0].min()) < 0 or int(labels[level][:, 0].max()) >= len(tile_lists[level]):
            raise RuntimeError(f"{level} labels outside gallery")

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "split": args.split,
        "checkpoint": ckpt,
        "model": {
            "name": MODEL_NAME,
            "moe_start_block": 11,
            "num_experts": 5,
            "top_k": 2,
            "router_condition": "none",
            "expert_layout": "routed",
            "feature_dim": 768,
        },
        "preprocess": preprocess,
        "query_images": query_images,
        "query_images_sha256": digest_strings(query_images),
        "query_features": query_features,
        "query_features_sha256": digest_tensor(query_features),
        "levels": {
            level: {
                "tile_list": tile_lists[level],
                "tile_list_sha256": digest_strings(tile_lists[level]),
                "ref_features": ref_features[level],
                "ref_features_sha256": digest_tensor(ref_features[level]),
                "query_labels": labels[level],
                "query_labels_sha256": digest_tensor(labels[level]),
            }
            for level in LEVELS
        },
        "data_folder": str(args.data_folder),
        "metadata_folder": str(args.metadata_folder),
    }
    manifest = {
        "schema": bundle["schema"],
        "split": bundle["split"],
        "checkpoint_sha256": ckpt["sha256"],
        "query_count": query_count,
        "query_images_sha256": bundle["query_images_sha256"],
        "query_features_sha256": bundle["query_features_sha256"],
        "gallery_counts": {level: len(tile_lists[level]) for level in LEVELS},
        "gallery_sha256": {
            level: bundle["levels"][level]["tile_list_sha256"] for level in LEVELS
        },
        "label_sha256": {
            level: bundle["levels"][level]["query_labels_sha256"] for level in LEVELS
        },
    }
    bundle["manifest"] = manifest
    bundle["signature"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    atomic_torch_save(args.output, bundle)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "signature": bundle["signature"],
                "split": args.split,
                "queries": query_count,
                "galleries": manifest["gallery_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
