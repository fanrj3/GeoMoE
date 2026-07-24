#!/usr/bin/env python3
"""Train a MoE-FFN unified VIGOR-M model for L1/L2/L3.

Attention stays shared through the full ViT. From ``moe_start_block`` onward,
each block replaces its FFN with top-k routed experts. The released setup uses
DINOv2 initialization for all experts. InfoNCE is computed per level so samples
from different physical scales are never used as negatives.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "geomoe").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import copy
import csv
import heapq
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.distributed.nn.functional import all_gather as all_gather_with_grad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from geomoe.datasets.vigor_m import (  # noqa: E402
    VigorMDatasetEval,
    VigorMDatasetTrain,
    _read_rgb,
    build_spatial_neighbor_dict,
)
from geomoe.loss import LevelWiseInfoNCE  # noqa: E402
from geomoe.model import LevelMoEFFNTimmModel, TimmModel  # noqa: E402
from geomoe.transforms import get_transforms_train, get_transforms_val  # noqa: E402
from geomoe.utils import Logger, setup_system  # noqa: E402


LEVELS = ("L1", "L2", "L3")


def _level_quota(batch_size, levels=LEVELS):
    base = int(batch_size) // len(levels)
    remainder = int(batch_size) % len(levels)
    return {
        level: base + int(idx < remainder)
        for idx, level in enumerate(levels)
    }

DEFAULT_CKPTS = {
    level: str(_PROJECT_ROOT / "weights" / "initialization" / "vigor_m" / f"{level}.pth")
    for level in LEVELS
}


@dataclass
class Configuration:
    model: str = "vit_base_patch14_dinov2.lvd142m"
    img_size: int = 384
    moe_start_block: int = 11
    num_experts: int = 5
    top_k: int = 2
    router_jitter: float = 0.01
    router_condition: str = "none"  # none | scale
    expert_layout: str = "routed"  # routed | shared_level_private
    moe_aux_weight: float = 0.01
    init_mode: str = "pretrained"  # pretrained | level_ckpts | checkpoint
    shared_init: str = "avg"        # avg | L1 | L2 | L3
    checkpoint_start: str = None
    level_ckpts: dict = None
    start_epoch: int = 1
    freeze_logit_scale: bool = True

    mixed_precision: bool = True
    seed: int = 1
    epochs: int = 60
    batch_size: int = 128
    batch_size_eval: int = 192
    verbose: bool = True
    gpu_ids: tuple = (0,)
    master_port: int = 12362

    custom_sampling: bool = True
    gps_sample: bool = True
    sim_sample: bool = True
    neighbour_select: int = 64
    neighbour_range: int = 128
    gps_neighbor_block_size: int = 512
    shuffle_retry: int = 8

    eval_every_n_epoch: int = 4
    normalize_features: bool = True
    skip_final_eval: bool = False
    zero_shot: bool = False
    smoke_shuffle: bool = False
    smoke_shuffle_rounds: int = 6

    clip_grad: float = 100.0
    grad_checkpointing: bool = False
    label_smoothing: float = 0.1
    level_loss_weights: tuple = (1.0, 1.0, 1.0)
    loss_mode: str = "levelwise"  # levelwise | plain

    lr: float = 0.0001
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    lr_end: float = 0.00001

    data_folder: str = "./data/VIGOR-M"
    metadata_folder: str = "./data/VIGOR-M/metadata"
    same_area: bool = True
    data_levels: tuple = LEVELS
    eval_levels: tuple = LEVELS
    dense_levels: tuple = ("L1",)
    l1_stride_fraction: float = 0.25
    steps_per_epoch: int = None
    strict_cell_conflict: bool = True

    ground_cutting: int = 0
    prob_rotate: float = 0.0
    prob_flip: float = 0.5

    model_path: str = "./outputs/checkpoints/vigor_m"
    run_name: str = None
    num_workers: int = 16
    num_workers_eval: int = 4
    persistent_workers: bool = True
    persistent_workers_eval: bool = False
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    def __post_init__(self):
        if self.level_ckpts is None:
            self.level_ckpts = dict(DEFAULT_CKPTS)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributedInfoNCE(torch.nn.Module):
    """Plain InfoNCE over the full DDP batch, without level grouping."""

    def __init__(self, loss_function, distributed=False):
        super().__init__()
        self.loss_function = loss_function
        self.distributed = distributed

    def _gather_features(self, tensor):
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return tensor
        return torch.cat(all_gather_with_grad(tensor), dim=0)

    def forward(self, image_features1, image_features2, logit_scale, _level_ids=None):
        image_features1 = torch.nn.functional.normalize(image_features1, dim=-1)
        image_features2 = torch.nn.functional.normalize(image_features2, dim=-1)

        all_features1 = self._gather_features(image_features1)
        all_features2 = self._gather_features(image_features2)

        local_batch = image_features1.size(0)
        if self.distributed and dist.is_available() and dist.is_initialized():
            targets = (
                torch.arange(local_batch, device=image_features1.device, dtype=torch.long)
                + dist.get_rank() * local_batch
            )
        else:
            targets = torch.arange(local_batch, device=image_features1.device, dtype=torch.long)

        logits_qr = logit_scale * image_features1 @ all_features2.T
        logits_rq = logit_scale * image_features2 @ all_features1.T
        return (self.loss_function(logits_qr, targets) + self.loss_function(logits_rq, targets)) / 2


def parse_levels(value):
    if value is None:
        return None
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def parse_gpu_ids(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def stride_for_level(config, level):
    return config.l1_stride_fraction if level == "L1" and level in set(config.dense_levels) else None


def get_logit_scale(model):
    return model.module.logit_scale.exp() if hasattr(model, "module") else model.logit_scale.exp()


def strip_module(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {key.replace("module.", ""): value for key, value in state.items()}


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_model_state(payload):
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def load_level_ckpt(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    return strip_module(load_torch_checkpoint(path))


def mean_tensors(tensors):
    return torch.stack([tensor.to(torch.float32) for tensor in tensors], dim=0).mean(dim=0).to(tensors[0].dtype)


def init_moe_ffn_model_from_level_ckpts(model, config, rank=0):
    level_states = {level: load_level_ckpt(config.level_ckpts[level]) for level in config.data_levels}
    target = model.state_dict()
    new_state = {}
    expert_levels = list(config.data_levels)
    if config.num_experts < len(expert_levels) + 1:
        raise ValueError(
            f"num_experts={config.num_experts} is too small for {expert_levels} plus base DINOv2 expert."
        )

    for key, value in target.items():
        if key == "logit_scale":
            source_values = [state[key] for state in level_states.values() if key in state]
            new_state[key] = mean_tensors(source_values) if source_values else value
            continue

        if ".moe.router." in key:
            new_state[key] = value
            continue

        if ".moe.experts." in key:
            parts = key.split(".")
            block_idx = int(parts[2])
            expert_idx = int(parts[5])
            rest = ".".join(parts[6:])
            old_key = f"model.blocks.{block_idx}.mlp.{rest}"
            if config.expert_layout == "shared_level_private":
                if expert_idx == 0:
                    source_values = [
                        state[old_key]
                        for state in level_states.values()
                        if old_key in state
                    ]
                    new_state[key] = mean_tensors(source_values) if source_values else value
                elif expert_idx <= len(expert_levels):
                    level = expert_levels[expert_idx - 1]
                    new_state[key] = level_states[level].get(old_key, value)
                else:
                    new_state[key] = value
            elif expert_idx < len(expert_levels):
                level = expert_levels[expert_idx]
                new_state[key] = level_states[level].get(old_key, value)
            else:
                # The extra expert is the original pretrained DINOv2 FFN that
                # was deep-copied during model construction.
                new_state[key] = value
            continue

        if key.startswith("model.blocks."):
            source_values = [state[key] for state in level_states.values() if key in state]
            selected = config.shared_init.upper()
            if selected in level_states and key in level_states[selected]:
                new_state[key] = level_states[selected][key]
            elif source_values:
                new_state[key] = mean_tensors(source_values)
            else:
                new_state[key] = value
            continue

        if key.startswith("model."):
            source_values = [state[key] for state in level_states.values() if key in state]
            selected = config.shared_init.upper()
            if selected in level_states and key in level_states[selected]:
                new_state[key] = level_states[selected][key]
            elif source_values:
                new_state[key] = mean_tensors(source_values)
            else:
                new_state[key] = value
        else:
            new_state[key] = value

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if rank == 0:
        print(
            "Initialized VIGOR-M MoE-FFN model from level ckpts: "
            f"shared_init={config.shared_init} experts={config.num_experts} "
            f"top_k={config.top_k} missing={len(missing)} unexpected={len(unexpected)}"
        )
        if config.expert_layout == "shared_level_private":
            print("  expert0: shared FFN <- mean(L1, L2, L3)")
            for expert_idx, level in enumerate(expert_levels, start=1):
                print(f"  expert{expert_idx}: {level} private FFN <- {config.level_ckpts[level]}")
        else:
            for expert_idx, level in enumerate(expert_levels):
                print(f"  expert{expert_idx}: {level} FFN <- {config.level_ckpts[level]}")
            print(f"  expert{len(expert_levels)}: base DINOv2 FFN")


class VigorMAllInDatasetTrain(Dataset):
    """Balanced L1/L2/L3 VIGOR-M train wrapper with strict batch guards."""

    def __init__(
        self,
        config,
        transforms_query=None,
        transforms_reference=None,
        shuffle_batch_size=252,
    ):
        super().__init__()
        self.config = config
        self.data_levels = tuple(config.data_levels)
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.prob_flip = config.prob_flip
        self.prob_rotate = config.prob_rotate
        self.shuffle_batch_size = int(shuffle_batch_size)
        self.steps_per_epoch = config.steps_per_epoch
        self.strict_cell_conflict = config.strict_cell_conflict

        self.level_datasets = {}
        self.level_offsets = {}
        self.level_label_counts = {}
        self.idx2pairs = defaultdict(list)
        self.idx2tile_center = {}
        self.idx2label_l3_cell = {}
        self.idx2label_level = {}
        self.global_to_local_label = {}
        self.global_ground_to_local = {}
        self.ground_l3_cell = {}
        self.ground_identity = {}
        self._shared_samples = None

        ground_key_to_idx = {}
        label_offset = 0

        for level in self.data_levels:
            ds = VigorMDatasetTrain(
                data_folder=config.data_folder,
                same_area=config.same_area,
                data_level=level,
                transforms_query=None,
                transforms_reference=None,
                prob_flip=0.0,
                prob_rotate=0.0,
                shuffle_batch_size=self.shuffle_batch_size,
                metadata_folder=config.metadata_folder,
                satellite_stride_fraction=stride_for_level(config, level),
                strict_cell_conflict=config.strict_cell_conflict,
            )
            self.level_datasets[level] = ds
            self.level_offsets[level] = label_offset
            self.level_label_counts[level] = len(ds.idx2tile)

            for local_label, center in ds.idx2tile_center.items():
                global_label = label_offset + int(local_label)
                self.idx2tile_center[global_label] = center
                self.idx2label_l3_cell[global_label] = ds.idx2label_cell[int(local_label)]
                self.idx2label_level[global_label] = level
                self.global_to_local_label[global_label] = (level, int(local_label))

            for local_ground, local_label in ds.pairs:
                city, ground_name, _ground_path = ds.idx2ground[int(local_ground)]
                ground_key = (city, ground_name)
                if ground_key not in ground_key_to_idx:
                    ground_key_to_idx[ground_key] = len(ground_key_to_idx)
                global_ground = ground_key_to_idx[ground_key]
                global_label = label_offset + int(local_label)
                sample = (global_ground, global_label, level)
                self.idx2pairs[global_label].append(sample)
                self.global_ground_to_local[(global_ground, level)] = int(local_ground)
                self.ground_l3_cell[global_ground] = ds.ground_cell[int(local_ground)]
                self.ground_identity[global_ground] = ground_key

            label_offset += len(ds.idx2tile)

        self.samples = [sample for pairs in self.idx2pairs.values() for sample in pairs]

        print("VigorMAllInDatasetTrain:")
        print(f"  Same area: {config.same_area}")
        print(f"  Levels: {self.data_levels}")
        print(f"  Ground identities: {len(self.ground_identity)}")
        print(f"  Total pairs: {len(self.samples)}")
        print(f"  Total reference labels: {len(self.idx2tile_center)}")
        print(f"  Labels with ground images: {len(self.idx2pairs)}")
        print(f"  Level offsets: {self.level_offsets}")
        print(f"  Level label counts: {self.level_label_counts}")

    def publish_worker_samples(self):
        """Publish the current epoch layout to persistent DataLoader workers."""
        encoded = torch.tensor(
            [
                (int(ground), int(label), self.data_levels.index(level))
                for ground, label, level in self.samples
            ],
            dtype=torch.int64,
        )
        if self._shared_samples is None:
            self._shared_samples = encoded.share_memory_()
            return
        if self._shared_samples.shape != encoded.shape:
            raise RuntimeError(
                "Persistent worker sample layout changed shape: "
                f"{tuple(self._shared_samples.shape)} -> {tuple(encoded.shape)}"
            )
        self._shared_samples.copy_(encoded)

    def _batch_quota(self, batch_size):
        return _level_quota(batch_size, self.data_levels)

    def _is_false_negative(self, ground_idx, label):
        if not self.strict_cell_conflict:
            return False
        return self.ground_l3_cell.get(int(ground_idx)) == self.idx2label_l3_cell.get(int(label))

    def _can_add_sample(self, sample, used_ground, used_labels, used_ground_cells, used_label_cells, batch):
        ground_idx, label, _level = sample
        if ground_idx in used_ground or label in used_labels:
            return False

        ground_cell = self.ground_l3_cell.get(int(ground_idx))
        label_cell = self.idx2label_l3_cell.get(int(label))
        if self.strict_cell_conflict:
            if ground_cell is not None and ground_cell in used_ground_cells:
                return False
            if label_cell is not None and label_cell in used_label_cells:
                return False
            if ground_cell is not None and ground_cell in used_label_cells:
                return False
            if label_cell is not None and label_cell in used_ground_cells:
                return False

        for other_ground, other_label, _other_level in batch:
            if self._is_false_negative(ground_idx, other_label):
                return False
            if self._is_false_negative(other_ground, label):
                return False
        return True

    def _mark_sample(self, sample, used_ground, used_labels, used_ground_cells, used_label_cells):
        ground_idx, label, _level = sample
        used_ground.add(int(ground_idx))
        used_labels.add(int(label))
        ground_cell = self.ground_l3_cell.get(int(ground_idx))
        label_cell = self.idx2label_l3_cell.get(int(label))
        if ground_cell is not None:
            used_ground_cells.add(ground_cell)
        if label_cell is not None:
            used_label_cells.add(label_cell)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128, target_steps=None, max_retries=1):
        best_samples = None
        best_dropped_tail = None
        best_steps = -1
        attempts = max(1, int(max_retries))
        for attempt in range(1, attempts + 1):
            samples_out, dropped_tail, global_quota = self._build_shuffle_once(
                sim_dict=sim_dict,
                neighbour_select=neighbour_select,
                neighbour_range=neighbour_range,
                target_steps=target_steps,
            )
            steps = len(samples_out) // self.shuffle_batch_size
            if steps > best_steps:
                best_samples = samples_out
                best_dropped_tail = dropped_tail
                best_steps = steps
            if target_steps is None or steps >= target_steps:
                if target_steps is not None:
                    samples_out = samples_out[:target_steps * self.shuffle_batch_size]
                    steps = target_steps
                self.samples = samples_out
                self._print_shuffle_summary(
                    global_quota,
                    dropped_tail,
                    sim_dict,
                    neighbour_select,
                    neighbour_range,
                    attempt,
                    attempts,
                    target_steps,
                )
                return

        if target_steps is not None:
            raise RuntimeError(
                "Could not build fixed-length VIGOR-M all-in batches without false negatives: "
                f"target_steps={target_steps}, best_steps={best_steps}, retries={attempts}."
            )
        if best_samples is None:
            raise RuntimeError("Could not build any VIGOR-M all-in false-negative-free batches.")
        self.samples = best_samples
        self._print_shuffle_summary(
            global_quota,
            best_dropped_tail,
            sim_dict,
            neighbour_select,
            neighbour_range,
            attempts,
            attempts,
            target_steps,
        )

    def _print_shuffle_summary(
        self,
        global_quota,
        dropped_tail,
        sim_dict,
        neighbour_select,
        neighbour_range,
        attempt,
        attempts,
        target_steps,
    ):
        print("\nShuffle VigorMAllInDatasetTrain:")
        print(f"  Length after shuffle: {len(self.samples)}")
        print(f"  Batches: {len(self.samples) // self.shuffle_batch_size}")
        print(f"  Dropped/unused tail samples: {dropped_tail}")
        print(f"  Global quota: {global_quota}")
        if target_steps is not None:
            print(f"  Fixed target steps: {target_steps} attempt={attempt}/{attempts}")
        if sim_dict is not None:
            print(f"  Hard sample mining: neighbour_select={neighbour_select} neighbour_range={neighbour_range}")

    def _build_shuffle_once(self, sim_dict=None, neighbour_select=64, neighbour_range=128, target_steps=None):
        by_level_labels = {
            level: sorted(
                int(label)
                for label in self.idx2pairs
                if self.idx2label_level[int(label)] == level
            )
            for level in self.data_levels
        }
        label_to_samples = {label: copy.deepcopy(samples) for label, samples in self.idx2pairs.items()}
        for samples in label_to_samples.values():
            random.shuffle(samples)

        global_quota = self._batch_quota(self.shuffle_batch_size)
        if any(len(by_level_labels[level]) < global_quota[level] for level in self.data_levels):
            raise ValueError(
                f"Not enough labels for quota {global_quota}; "
                f"labels per level={ {level: len(by_level_labels[level]) for level in self.data_levels} }"
            )

        heaps = {}
        for level in self.data_levels:
            heap = [
                (-len(label_to_samples[label]), random.random(), label)
                for label in by_level_labels[level]
                if label_to_samples[label]
            ]
            heapq.heapify(heap)
            heaps[level] = heap

        def active_count(label):
            return len(label_to_samples[label])

        def push_label(level, label):
            count = active_count(label)
            if count > 0:
                heapq.heappush(heaps[level], (-count, random.random(), label))

        def pop_available_label(level, exclude):
            skipped = []
            selected = None
            heap = heaps[level]
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

        def hard_candidates(seed, level, exclude):
            if sim_dict is None:
                return []
            near = list(sim_dict.get(int(seed), []))[:neighbour_range]
            always = near[:neighbour_select // 2]
            random_part = near[neighbour_select // 2:]
            random.shuffle(random_part)
            candidates = always + random_part[:neighbour_select - len(always)]
            out = []
            for label in candidates:
                label = int(label)
                if (
                    label in label_to_samples
                    and label not in exclude
                    and self.idx2label_level[label] == level
                    and active_count(label) > 0
                ):
                    out.append(label)
            out.sort(key=lambda item: (-active_count(item), random.random()))
            return out

        def take_valid_sample(label, used_ground, used_labels, used_ground_cells, used_label_cells, batch):
            postponed = []
            chosen = None
            samples = label_to_samples[label]
            while samples:
                sample = samples.pop()
                if self._can_add_sample(
                    sample,
                    used_ground,
                    used_labels,
                    used_ground_cells,
                    used_label_cells,
                    batch,
                ):
                    chosen = sample
                    break
                postponed.append(sample)
            if postponed:
                samples[:0] = postponed
            return chosen

        samples_out = []
        dropped_tail = 0
        while True:
            batch = []
            used_ground = set()
            used_labels = set()
            used_ground_cells = set()
            used_label_cells = set()
            batch_failed = False

            for level in self.data_levels:
                current_labels = set()
                seed = pop_available_label(level, current_labels)
                if seed is None:
                    batch_failed = True
                    break
                candidate_labels = [seed] + hard_candidates(seed, level, {seed})

                while sum(1 for sample in batch if sample[2] == level) < global_quota[level]:
                    if candidate_labels:
                        label = candidate_labels.pop(0)
                    else:
                        label = pop_available_label(level, current_labels)
                        if label is None:
                            batch_failed = True
                            break
                    if label in current_labels:
                        continue
                    sample = take_valid_sample(
                        label,
                        used_ground,
                        used_labels,
                        used_ground_cells,
                        used_label_cells,
                        batch,
                    )
                    current_labels.add(label)
                    if sample is None:
                        push_label(level, label)
                        continue
                    batch.append(sample)
                    self._mark_sample(
                        sample,
                        used_ground,
                        used_labels,
                        used_ground_cells,
                        used_label_cells,
                    )
                    push_label(level, label)

                for label in current_labels:
                    push_label(level, label)
                if batch_failed:
                    break

            if batch_failed or len(batch) < self.shuffle_batch_size:
                for sample in batch:
                    label_to_samples[sample[1]].append(sample)
                dropped_tail = sum(len(items) for items in label_to_samples.values())
                break

            random.shuffle(batch)
            samples_out.extend(batch)

            max_steps = target_steps if target_steps is not None else self.steps_per_epoch
            if max_steps is not None and len(samples_out) >= max_steps * self.shuffle_batch_size:
                break

        if not samples_out:
            raise RuntimeError("Could not build any VIGOR-M all-in false-negative-free batches.")
        return samples_out, dropped_tail, global_quota

    def __getitem__(self, index):
        if self._shared_samples is None:
            global_ground, global_label, level = self.samples[index]
        else:
            global_ground, global_label, level_idx = self._shared_samples[index].tolist()
            level = self.data_levels[level_idx]
        ds = self.level_datasets[level]
        local_ground = self.global_ground_to_local[(int(global_ground), level)]
        local_level, local_label = self.global_to_local_label[int(global_label)]
        if local_level != level:
            raise RuntimeError(f"Sample level mismatch: {level} vs {local_level}")

        query_img = _read_rgb(ds.idx2ground_path[local_ground])
        if ds._use_dense_l1():
            reference_img = ds._dense_l1_crop(local_label)
        else:
            reference_img = _read_rgb(ds.idx2tile_path[local_label])

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
            _c, _h, w = query_img.shape
            query_img = torch.roll(query_img, shifts=-w // 4 * r, dims=2)

        level_id = torch.tensor(self.data_levels.index(level), dtype=torch.long)
        label = torch.tensor(global_label, dtype=torch.long)
        return query_img, reference_img, label, level_id

    def __len__(self):
        return len(self.samples)


def summarize_level_batch(samples, global_batch_size, levels=LEVELS):
    rows = []
    for start in range(0, len(samples), global_batch_size):
        batch = samples[start:start + global_batch_size]
        if len(batch) < global_batch_size:
            continue
        counts = {level: 0 for level in levels}
        labels = set()
        grounds = set()
        dup_labels = 0
        dup_grounds = 0
        for ground, label, level in batch:
            counts[level] = counts.get(level, 0) + 1
            dup_labels += int(label in labels)
            dup_grounds += int(ground in grounds)
            labels.add(label)
            grounds.add(ground)
        rows.append((counts, dup_labels, dup_grounds))
    return rows


def validate_ddp_shuffle_samples(
    train_ds,
    local_batch_size,
    world_size,
    levels=LEVELS,
    expected_steps=None,
    context="shuffle",
):
    global_batch_size = local_batch_size * world_size
    if len(train_ds.samples) % global_batch_size != 0:
        raise RuntimeError(
            f"{context}: sample count {len(train_ds.samples)} is not divisible by "
            f"global batch size {global_batch_size}."
        )

    steps = len(train_ds.samples) // global_batch_size
    if expected_steps is not None and steps != expected_steps:
        raise RuntimeError(
            f"{context}: custom sampler produced {steps} DDP batches, expected {expected_steps}."
        )

    global_quota = train_ds._batch_quota(global_batch_size)
    local_quota = train_ds._batch_quota(local_batch_size)
    max_dup_labels = 0
    max_dup_grounds = 0
    max_dup_cells = 0

    for batch_idx in range(steps):
        batch = train_ds.samples[batch_idx * global_batch_size:(batch_idx + 1) * global_batch_size]
        counts = {level: 0 for level in levels}
        used_ground = set()
        used_labels = set()
        used_ground_cells = set()
        used_label_cells = set()
        dup_ground = 0
        dup_label = 0
        dup_cell = 0

        for ground_idx, label, level in batch:
            if level not in counts:
                raise RuntimeError(f"{context}: unknown level {level!r} in batch {batch_idx}.")
            counts[level] += 1
            dup_ground += int(ground_idx in used_ground)
            dup_label += int(label in used_labels)
            used_ground.add(int(ground_idx))
            used_labels.add(int(label))
            ground_cell = train_ds.ground_l3_cell.get(int(ground_idx))
            label_cell = train_ds.idx2label_l3_cell.get(int(label))
            if train_ds.strict_cell_conflict:
                dup_cell += int(ground_cell is not None and ground_cell in used_ground_cells)
                dup_cell += int(label_cell is not None and label_cell in used_label_cells)
                dup_cell += int(ground_cell is not None and ground_cell in used_label_cells)
                dup_cell += int(label_cell is not None and label_cell in used_ground_cells)
            if ground_cell is not None:
                used_ground_cells.add(ground_cell)
            if label_cell is not None:
                used_label_cells.add(label_cell)

        if counts != global_quota:
            raise RuntimeError(f"{context}: global batch {batch_idx} quota mismatch {counts} vs {global_quota}.")
        if dup_ground or dup_label or dup_cell:
            raise RuntimeError(
                f"{context}: false-negative guard failed in global batch {batch_idx}: "
                f"dup_ground={dup_ground} dup_label={dup_label} dup_cell={dup_cell}."
            )

        max_dup_grounds = max(max_dup_grounds, dup_ground)
        max_dup_labels = max(max_dup_labels, dup_label)
        max_dup_cells = max(max_dup_cells, dup_cell)

        for rank in range(world_size):
            rank_batch = batch[rank::world_size]
            rank_counts = {level: 0 for level in levels}
            for _ground_idx, _label, level in rank_batch:
                rank_counts[level] += 1
            if rank_counts != local_quota:
                raise RuntimeError(
                    f"{context}: rank {rank} local batch {batch_idx} quota mismatch "
                    f"{rank_counts} vs {local_quota}."
                )

    return {
        "steps": steps,
        "samples": len(train_ds.samples),
        "global_batch_size": global_batch_size,
        "global_quota": global_quota,
        "local_quota": local_quota,
        "max_dup_labels": max_dup_labels,
        "max_dup_grounds": max_dup_grounds,
        "max_dup_cells": max_dup_cells,
    }


def layout_samples_for_ddp(samples, local_batch_size, world_size, levels=LEVELS):
    global_batch_size = local_batch_size * world_size
    local_quota = _level_quota(local_batch_size, levels)

    reordered = []
    for start in range(0, len(samples), global_batch_size):
        batch = list(samples[start:start + global_batch_size])
        if len(batch) < global_batch_size:
            break

        by_level = {level: [] for level in levels}
        for sample in batch:
            by_level[sample[2]].append(sample)

        per_rank = [[] for _ in range(world_size)]
        for level in levels:
            bucket = by_level[level]
            cursor = 0
            for rank in range(world_size):
                quota = local_quota[level]
                per_rank[rank].extend(bucket[cursor:cursor + quota])
                cursor += quota
            if cursor != len(bucket):
                raise ValueError(
                    f"Cannot split {len(bucket)} {level} samples across {world_size} ranks "
                    f"with local quota {local_quota[level]}."
                )

        for rank_samples in per_rank:
            if len(rank_samples) != local_batch_size:
                raise ValueError(f"Rank local batch has {len(rank_samples)}, expected {local_batch_size}.")

        for i in range(local_batch_size):
            for rank in range(world_size):
                reordered.append(per_rank[rank][i])
    return reordered


def shuffle_layout_and_validate(config, train_ds, sim_dict, world_size, expected_steps=None, context="shuffle"):
    target_steps = expected_steps if expected_steps is not None else train_ds.steps_per_epoch
    train_ds.shuffle(
        sim_dict,
        neighbour_select=config.neighbour_select,
        neighbour_range=config.neighbour_range,
        target_steps=target_steps,
        max_retries=config.shuffle_retry,
    )
    train_ds.samples = layout_samples_for_ddp(
        train_ds.samples,
        config.batch_size,
        world_size,
        levels=config.data_levels,
    )
    stats = validate_ddp_shuffle_samples(
        train_ds,
        config.batch_size,
        world_size,
        levels=config.data_levels,
        expected_steps=expected_steps,
        context=context,
    )
    print(
        f"Shuffle validation [{context}]: "
        f"steps={stats['steps']} samples={stats['samples']} "
        f"global_quota={stats['global_quota']} local_quota={stats['local_quota']}"
    )
    return stats


def sync_samples(train_ds, rank):
    payload = [train_ds.samples if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    if rank != 0:
        train_ds.samples = payload[0]
    if train_ds._shared_samples is not None:
        train_ds.publish_worker_samples()


def sync_sim_dict(sim_dict, rank):
    payload = [sim_dict if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score,
    train_ds,
    sim_dict,
    config,
):
    state = {
        "format_version": 1,
        "epoch": int(epoch),
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "best_score": float(best_score),
        "samples": list(train_ds.samples),
        "sim_dict": sim_dict,
        "rng_state": capture_rng_state(),
        "config": dict(vars(config)),
    }
    temp_path = f"{path}.tmp-{os.getpid()}"
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def dataloader_kwargs(config, batch_size, sampler=None, shuffle=False, train=False):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers if train else config.num_workers_eval,
        pin_memory=True,
        drop_last=train,
    )
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = (
            config.persistent_workers if train else config.persistent_workers_eval
        )
        kwargs["multiprocessing_context"] = "spawn"
    return kwargs


def dist_predict(config, model, dataloader, level, rank, world_size):
    model.eval()
    local_feats, local_labels = [], []

    if rank == 0 and config.verbose:
        from tqdm import tqdm
        bar = tqdm(dataloader, total=len(dataloader), desc=f"Feat {level}")
    else:
        bar = dataloader

    level_id = config.data_levels.index(level)
    with torch.no_grad():
        for img, ids in bar:
            img = img.cuda(non_blocking=True)
            levels = torch.full((img.size(0),), level_id, dtype=torch.long, device=img.device)
            with autocast(device_type="cuda", enabled=config.mixed_precision):
                feat = model(img, levels=levels)
                if config.normalize_features:
                    feat = torch.nn.functional.normalize(feat, dim=-1)
            local_feats.append(feat.to(torch.float32))
            local_labels.append(ids)

    if rank == 0 and config.verbose:
        bar.close()

    local_feats = torch.cat(local_feats, dim=0)
    local_labels = torch.cat(local_labels, dim=0).cuda()

    local_size = torch.tensor([local_feats.size(0)], dtype=torch.long, device="cuda")
    sizes = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)

    feat_dim = local_feats.size(1)
    pad_f = torch.zeros(max_size, feat_dim, device="cuda", dtype=local_feats.dtype)
    pad_f[: local_feats.size(0)] = local_feats

    if local_labels.ndim == 1:
        pad_l = torch.zeros(max_size, device="cuda", dtype=local_labels.dtype)
        pad_l[: local_labels.size(0)] = local_labels
    else:
        pad_l = torch.zeros(max_size, local_labels.size(1), device="cuda", dtype=local_labels.dtype)
        pad_l[: local_labels.size(0)] = local_labels

    all_feats = [torch.zeros_like(pad_f) for _ in range(world_size)]
    all_labels = [torch.zeros_like(pad_l) for _ in range(world_size)]
    dist.all_gather(all_feats, pad_f)
    dist.all_gather(all_labels, pad_l)

    if rank != 0:
        return None, None

    return (
        torch.cat([all_feats[i][: sizes[i]] for i in range(world_size)], dim=0),
        torch.cat([all_labels[i][: sizes[i]] for i in range(world_size)], dim=0),
    )


def dist_evaluate(config, model, ref_loader, qry_loader, level, rank, world_size, ranks=(1, 5, 10)):
    ref_feats, ref_labels = dist_predict(config, model, ref_loader, level, rank, world_size)
    qry_feats, qry_labels = dist_predict(config, model, qry_loader, level, rank, world_size)
    if rank != 0:
        return None

    from geomoe.evaluate.vigor_m import calculate_scores
    return calculate_scores(qry_feats, ref_feats, qry_labels, ref_labels, step_size=1000, ranks=list(ranks))


def evaluate_all_levels(config, model, eval_loaders, rank, world_size):
    scores = {}
    for level, (ref_loader, qry_loader) in eval_loaders.items():
        if rank == 0:
            print(f"\n{'=' * 30}[Evaluate {level}]{'=' * 30}")
        score = dist_evaluate(config, model, ref_loader, qry_loader, level, rank, world_size)
        if rank == 0:
            scores[level] = score
    if rank == 0:
        mean_score = sum(scores.values()) / max(1, len(scores))
        print(
            "VIGOR-M MoE-FFN Eval Summary: "
            + " ".join(f"{level} R@1={score:.4f}" for level, score in scores.items())
            + f" Mean={mean_score:.4f}"
        )
        return mean_score, scores
    return None, None


def calc_ref_sim(config, model, ref_loaders, level_offsets, rank, world_size):
    if not ref_loaders:
        return None
    sim_dict = {}
    for level, ref_loader in ref_loaders.items():
        ref_feats, ref_labels = dist_predict(config, model, ref_loader, level, rank, world_size)
        if rank != 0:
            continue
        ref_labels_np = ref_labels.cpu().numpy().astype(int) + int(level_offsets[level])
        for start in range(0, len(ref_feats), 1000):
            end = min(start + 1000, len(ref_feats))
            sim = ref_feats[start:end] @ ref_feats.T
            rows = torch.arange(end - start, device=sim.device)
            sim[rows, torch.arange(start, end, device=sim.device)] = -float("inf")
            k = min(config.neighbour_range, max(0, len(ref_feats) - 1))
            if k == 0:
                continue
            _, idx = torch.topk(sim, k=k, dim=1)
            idx = idx.cpu().numpy()
            for row, label in enumerate(ref_labels_np[start:end]):
                sim_dict[int(label)] = [int(ref_labels_np[col]) for col in idx[row]]
    return sim_dict if rank == 0 else None


def train_epoch(config, model, loader, loss_fn, optimizer, scheduler, scaler, epoch, rank, gpu):
    model.train()
    losses = AverageMeter()
    info_losses = AverageMeter()
    aux_losses = AverageMeter()
    optimizer.zero_grad(set_to_none=True)

    if rank == 0 and config.verbose:
        from tqdm import tqdm
        bar = tqdm(loader, total=len(loader), desc=f"E{epoch}")
    else:
        bar = loader

    for step, batch in enumerate(bar):
        query, reference, _ids, level_ids = batch
        query = query.cuda(gpu, non_blocking=True)
        reference = reference.cuda(gpu, non_blocking=True)
        level_ids = level_ids.cuda(gpu, non_blocking=True)

        try:
            if scaler:
                with autocast(device_type="cuda", enabled=config.mixed_precision):
                    f1, f2, moe_aux = model(query, reference, levels=level_ids, return_moe_loss=True)
                    info_loss = loss_fn(f1, f2, get_logit_scale(model), level_ids)
                    loss = info_loss + config.moe_aux_weight * moe_aux
                losses.update(loss.item(), query.size(0))
                info_losses.update(info_loss.item(), query.size(0))
                aux_losses.update(moe_aux.item(), query.size(0))
                scaler.scale(loss).backward()
                if config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                f1, f2, moe_aux = model(query, reference, levels=level_ids, return_moe_loss=True)
                info_loss = loss_fn(f1, f2, get_logit_scale(model), level_ids)
                loss = info_loss + config.moe_aux_weight * moe_aux
                losses.update(loss.item(), query.size(0))
                info_losses.update(info_loss.item(), query.size(0))
                aux_losses.update(moe_aux.item(), query.size(0))
                loss.backward()
                if config.clip_grad:
                    torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            print(f"[Rank {rank}] Error at step {step}: {e}", flush=True)
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise
            optimizer.zero_grad(set_to_none=True)
            continue

        if scheduler is not None:
            scheduler.step()

        if rank == 0 and config.verbose:
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                info=f"{info_losses.val:.4f}",
                aux=f"{aux_losses.val:.4f}",
                avg=f"{losses.avg:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

    if rank == 0 and config.verbose:
        bar.close()
    return losses.avg


def build_model(config, rank):
    resume_state = None
    model = LevelMoEFFNTimmModel(
        config.model,
        pretrained=(config.init_mode != "checkpoint"),
        img_size=config.img_size,
        levels=config.data_levels,
        moe_start_block=config.moe_start_block,
        num_experts=config.num_experts,
        top_k=config.top_k,
        router_jitter=config.router_jitter,
        router_condition=config.router_condition,
        expert_layout=config.expert_layout,
        default_level="L3",
    )
    data_cfg = model.get_config()
    if config.init_mode == "level_ckpts":
        init_moe_ffn_model_from_level_ckpts(model, config, rank=rank)
    elif config.init_mode == "checkpoint":
        payload = load_torch_checkpoint(config.checkpoint_start)
        state = strip_module(checkpoint_model_state(payload))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if isinstance(payload, dict) and "model" in payload:
            resume_state = payload
        if rank == 0:
            print(f"Loaded MoE-FFN checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    elif config.checkpoint_start:
        payload = load_torch_checkpoint(config.checkpoint_start)
        state = strip_module(checkpoint_model_state(payload))
        model.load_state_dict(state, strict=False)
        if isinstance(payload, dict) and "model" in payload:
            resume_state = payload
    return model, data_cfg, resume_state


def build_train_dataset_for_shuffle(config, transforms_query=None, transforms_reference=None):
    return VigorMAllInDatasetTrain(
        config,
        transforms_query=transforms_query,
        transforms_reference=transforms_reference,
        shuffle_batch_size=config.batch_size * len(config.gpu_ids),
    )


def build_label_cycle_sim_dict(train_ds, reverse=False):
    sim_dict = {}
    for level in train_ds.data_levels:
        labels = sorted(
            int(label)
            for label in train_ds.idx2pairs
            if train_ds.idx2label_level[int(label)] == level
        )
        if reverse:
            labels = list(reversed(labels))
        if len(labels) <= 1:
            for label in labels:
                sim_dict[int(label)] = []
            continue
        limit = min(len(labels) - 1, 256)
        for idx, label in enumerate(labels):
            neighbours = []
            for offset in range(1, min(len(labels), 257)):
                neighbours.append(int(labels[(idx + offset) % len(labels)]))
                if len(neighbours) >= limit:
                    break
            sim_dict[int(label)] = neighbours
    return sim_dict


def build_levelwise_gps_sim_dict(config, train_ds):
    sim_dict = {}
    for level in train_ds.data_levels:
        labels = [
            int(label)
            for label in train_ds.idx2pairs.keys()
            if train_ds.idx2label_level[int(label)] == level
        ]
        level_sim = build_spatial_neighbor_dict(
            train_ds.idx2tile_center,
            labels=labels,
            top_k=config.neighbour_range,
            block_size=config.gps_neighbor_block_size,
        )
        sim_dict.update(level_sim)
    return sim_dict


def smoke_shuffle(config):
    world_size = len(config.gpu_ids)
    logical_batch_size = config.batch_size * world_size
    print(
        f"Smoke shuffle VIGOR-M: world_size={world_size} batch/GPU={config.batch_size} "
        f"global_batch={logical_batch_size} rounds={config.smoke_shuffle_rounds}"
    )
    train_ds = build_train_dataset_for_shuffle(config)

    sim_cases = []
    if config.gps_sample:
        gps_sim = build_levelwise_gps_sim_dict(config, train_ds)
        sim_cases.append(("gps", gps_sim))
    sim_cases.append(("none", None))
    sim_cases.append(("cycle", build_label_cycle_sim_dict(train_ds, reverse=False)))
    sim_cases.append(("cycle_reverse", build_label_cycle_sim_dict(train_ds, reverse=True)))

    fixed_steps = None
    for case_name, sim_dict in sim_cases:
        print(f"\n{'=' * 30}[Smoke Case: {case_name}]{'=' * 30}")
        train_ds.steps_per_epoch = fixed_steps
        for round_idx in range(config.smoke_shuffle_rounds):
            stats = shuffle_layout_and_validate(
                config,
                train_ds,
                sim_dict,
                world_size,
                expected_steps=fixed_steps,
                context=f"smoke_{case_name}_round_{round_idx + 1}",
            )
            if fixed_steps is None:
                fixed_steps = stats["steps"]
                train_ds.steps_per_epoch = fixed_steps
                print(f"Smoke fixed steps per epoch: {fixed_steps}")
            if stats["steps"] != fixed_steps:
                raise RuntimeError(
                    f"Smoke case {case_name} round {round_idx + 1}: "
                    f"steps changed to {stats['steps']} from {fixed_steps}."
                )
    print("\nSmoke shuffle passed.")


def run(rank, world_size, config, gpu_ids):
    gpu = gpu_ids[rank]
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(config.master_port)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu)

    setup_system(
        seed=config.seed + rank,
        cudnn_benchmark=config.cudnn_benchmark,
        cudnn_deterministic=config.cudnn_deterministic,
    )

    level_tag = "".join(config.data_levels)
    area_tag = "same" if config.same_area else "cross"
    dense_tag = f"denseL1_stride{config.l1_stride_fraction:g}" if "L1" in config.dense_levels else "nodense"
    model_path = "{}/{}_moeffnB{}_E{}_top{}_{}_{}_{}/aux{}/{}/{}".format(
        config.model_path,
        area_tag,
        config.moe_start_block,
        config.num_experts,
        config.top_k,
        level_tag,
        dense_tag,
        "lr" + f"{config.lr:g}",
        f"{config.moe_aux_weight:g}",
        config.model,
        config.run_name or time.strftime("%H%M%S"),
    )
    if rank == 0:
        os.makedirs(model_path, exist_ok=True)
        resume_run = config.start_epoch > 1 or config.init_mode == "checkpoint"
        resume_tag = time.strftime("%Y%m%d_%H%M%S") if resume_run else None
        train_name = f"train_resume_{resume_tag}.py" if resume_run else "train.py"
        model_name = f"model_resume_{resume_tag}.py" if resume_run else "model.py"
        log_name = f"log_resume_{resume_tag}.txt" if resume_run else "log.txt"
        shutil.copyfile(Path(__file__).resolve(), f"{model_path}/{train_name}")
        shutil.copyfile(_PROJECT_ROOT / "geomoe" / "model.py", f"{model_path}/{model_name}")
        sys.stdout = Logger(os.path.join(model_path, log_name))
        print(f"Output path: {model_path}")

    model, data_cfg, resume_state = build_model(config, rank)
    mean, std = data_cfg["mean"], data_cfg["std"]
    image_size_sat = (config.img_size, config.img_size)
    new_w = config.img_size * 2
    new_h = int(((288 - 2 * config.ground_cutting) / 512) * new_w)
    img_size_ground = (new_h, new_w)

    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    if config.freeze_logit_scale:
        model.logit_scale.requires_grad_(False)

    model = model.cuda(gpu)
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True, gradient_as_bucket_view=False)

    if rank == 0:
        print(
            f"\nModel: {config.model} VIGOR-M MoE-FFN levels={config.data_levels} "
            f"same_area={config.same_area} shared_attention=all_blocks "
            f"moe_ffn_blocks={config.moe_start_block}..11 experts={config.num_experts} "
            f"top_k={config.top_k} expert_layout={config.expert_layout} "
            f"router_condition={config.router_condition} "
            f"GPUs={world_size}"
        )
        print(
            f"Batch/GPU: {config.batch_size} Effective batch: {config.batch_size * world_size} "
            + (
                "Level-wise InfoNCE: no cross-level negatives "
                if config.loss_mode == "levelwise"
                else "Plain InfoNCE: cross-level negatives enabled "
            )
            + f"MoE aux={config.moe_aux_weight}"
        )
        print(f"Init mode: {config.init_mode} shared_init={config.shared_init}")
        print(f"Train Epochs: {config.epochs}")

    sat_tf_tr, gnd_tf_tr = get_transforms_train(
        image_size_sat, img_size_ground, mean=mean, std=std, ground_cutting=config.ground_cutting
    )
    sat_tf_val, gnd_tf_val = get_transforms_val(
        image_size_sat, img_size_ground, mean=mean, std=std, ground_cutting=config.ground_cutting
    )

    logical_batch_size = config.batch_size * world_size
    train_ds = build_train_dataset_for_shuffle(
        config,
        transforms_query=gnd_tf_tr,
        transforms_reference=sat_tf_tr,
    )

    eval_loaders = {}
    for level in config.eval_levels:
        ref_ds = VigorMDatasetEval(
            data_folder=config.data_folder,
            split="test",
            img_type="reference",
            same_area=config.same_area,
            data_level=level,
            transforms=sat_tf_val,
            metadata_folder=config.metadata_folder,
            satellite_stride_fraction=stride_for_level(config, level),
        )
        qry_ds = VigorMDatasetEval(
            data_folder=config.data_folder,
            split="test",
            img_type="query",
            same_area=config.same_area,
            data_level=level,
            transforms=gnd_tf_val,
            metadata_folder=config.metadata_folder,
            satellite_stride_fraction=stride_for_level(config, level),
        )
        ref_sampler = DistributedSampler(ref_ds, num_replicas=world_size, rank=rank, shuffle=False)
        qry_sampler = DistributedSampler(qry_ds, num_replicas=world_size, rank=rank, shuffle=False)
        eval_loaders[level] = (
            DataLoader(ref_ds, **dataloader_kwargs(config, config.batch_size_eval, sampler=ref_sampler)),
            DataLoader(qry_ds, **dataloader_kwargs(config, config.batch_size_eval, sampler=qry_sampler)),
        )
        if rank == 0:
            print(f"Eval {level}: {len(qry_ds)} queries, {len(ref_ds)} refs")

    ref_loaders_train = {}
    if config.sim_sample:
        for level in config.data_levels:
            ref_ds = VigorMDatasetEval(
                data_folder=config.data_folder,
                split="train",
                img_type="reference",
                same_area=config.same_area,
                data_level=level,
                transforms=sat_tf_val,
                metadata_folder=config.metadata_folder,
                satellite_stride_fraction=stride_for_level(config, level),
            )
            ref_sampler = DistributedSampler(ref_ds, num_replicas=world_size, rank=rank, shuffle=False)
            ref_loaders_train[level] = DataLoader(
                ref_ds, **dataloader_kwargs(config, config.batch_size_eval, sampler=ref_sampler)
            )
            if rank == 0:
                print(f"Train ref {level}: {len(ref_ds)} refs")

    sim_dict = None
    if config.gps_sample:
        sim_dict = build_levelwise_gps_sim_dict(config, train_ds)
        if rank == 0:
            print(f"Spatial GPS Sample: labels={len(sim_dict)} topk={config.neighbour_range}")
    sim_dict = sync_sim_dict(sim_dict, rank)
    if resume_state is not None and resume_state.get("sim_dict") is not None:
        sim_dict = resume_state["sim_dict"]
        sim_dict = sync_sim_dict(sim_dict, rank)
        if rank == 0:
            print(f"Restored hard-sampling graph: labels={len(sim_dict)}")

    if config.zero_shot:
        if rank == 0:
            print(f"\n{'=' * 30}[Zero Shot]{'=' * 30}")
        evaluate_all_levels(config, model.module, eval_loaders, rank, world_size)
        if config.sim_sample:
            sim_dict = calc_ref_sim(config, model.module, ref_loaders_train, train_ds.level_offsets, rank, world_size)
            sim_dict = sync_sim_dict(sim_dict, rank)

    if config.custom_sampling:
        resume_samples = None if resume_state is None else resume_state.get("samples")
        if resume_samples:
            train_ds.samples = resume_samples
            train_ds.steps_per_epoch = len(train_ds.samples) // logical_batch_size
            if rank == 0:
                print(
                    "Restored next-epoch sample layout: "
                    f"samples={len(train_ds.samples)} steps={train_ds.steps_per_epoch}"
                )
        elif rank == 0:
            stats = shuffle_layout_and_validate(
                config,
                train_ds,
                sim_dict,
                world_size,
                expected_steps=None,
                context="initial",
            )
            fixed_steps_per_epoch = stats["steps"]
            train_ds.steps_per_epoch = fixed_steps_per_epoch
            print(f"Fixed DDP steps per epoch: {fixed_steps_per_epoch}")
        sync_samples(train_ds, rank)
        if rank != 0:
            train_ds.steps_per_epoch = len(train_ds.samples) // logical_batch_size
        train_ds.publish_worker_samples()

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        **dataloader_kwargs(config, config.batch_size, sampler=train_sampler, train=True),
    )

    loss_fn_ce = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    if config.loss_mode == "levelwise":
        level_weights = torch.tensor(config.level_loss_weights, dtype=torch.float32)
        loss_function = LevelWiseInfoNCE(
            loss_function=loss_fn_ce,
            level_weights=level_weights,
            distributed=True,
            device=f"cuda:{gpu}",
        )
    elif config.loss_mode == "plain":
        loss_function = DistributedInfoNCE(loss_function=loss_fn_ce, distributed=True)
    else:
        raise ValueError(f"Unsupported loss_mode: {config.loss_mode}")
    scaler = GradScaler(device="cuda", init_scale=2.0**10) if config.mixed_precision else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = len(train_loader) * config.warmup_epochs
    if config.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_training_steps=total_steps, lr_end=config.lr_end,
            power=1.5, num_warmup_steps=warmup_steps,
        )
    elif config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    full_resume = resume_state is not None and resume_state.get("optimizer") is not None
    best_score = 0.0
    if full_resume:
        optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        if scaler is not None and resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        completed_epoch = int(resume_state.get("epoch", 0))
        config.start_epoch = max(config.start_epoch, completed_epoch + 1)
        best_score = float(resume_state.get("best_score", 0.0))
        restore_rng_state(resume_state.get("rng_state"))
        if rank == 0:
            print(
                f"Restored optimizer/scheduler/scaler from epoch {completed_epoch}; "
                f"continuing at epoch {config.start_epoch}"
            )
    elif scheduler is not None and config.start_epoch > 1:
        for _ in range((config.start_epoch - 1) * len(train_loader)):
            scheduler.step()
    if rank == 0:
        print(f"Scheduler: {config.scheduler} Warmup: {warmup_steps} Total: {total_steps}")

    loss_csv_path = os.path.join(model_path, "loss.csv") if rank == 0 else None
    if rank == 0:
        csv_mode = "a" if config.start_epoch > 1 and os.path.exists(loss_csv_path) else "w"
        with open(loss_csv_path, csv_mode, newline="") as f:
            writer = csv.writer(f)
            if csv_mode == "w":
                writer.writerow(["epoch", "train_loss", "lr", "mean_recall@1"] +
                                [f"{level}_recall@1" for level in config.eval_levels])

    for epoch in range(config.start_epoch, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        dist.barrier()
        if rank == 0:
            print(f"\n{'=' * 30}[Epoch: {epoch}]{'=' * 30}")

        train_loss = train_epoch(config, model, train_loader, loss_function,
                                 optimizer, scheduler, scaler, epoch, rank, gpu)
        if rank == 0:
            print(f"Train Loss: {train_loss:.3f} LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Epoch: {epoch}, Train Loss = {train_loss:.4f}, Lr = {optimizer.param_groups[0]['lr']:.6e}")
            torch.save(model.module.state_dict(), f"{model_path}/weights_latest.pth")

        mean_score = None
        level_scores = {}
        should_eval = (epoch % config.eval_every_n_epoch == 0 and epoch != 0)
        should_eval = should_eval or (epoch == config.epochs and not config.skip_final_eval)
        if should_eval:
            mean_score, level_scores = evaluate_all_levels(config, model.module, eval_loaders, rank, world_size)
            if config.sim_sample:
                sim_dict = calc_ref_sim(
                    config, model.module, ref_loaders_train, train_ds.level_offsets, rank, world_size
                )
                sim_dict = sync_sim_dict(sim_dict, rank)
            if rank == 0 and mean_score is not None and mean_score > best_score:
                best_score = mean_score
                torch.save(model.module.state_dict(), f"{model_path}/weights_e{epoch}_{mean_score:.4f}.pth")

        if rank == 0:
            with open(loss_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [epoch, f"{train_loss:.4f}", f"{optimizer.param_groups[0]['lr']:.6f}",
                     "" if mean_score is None else mean_score]
                    + [level_scores.get(level, "") for level in config.eval_levels]
                )

        if config.custom_sampling:
            if rank == 0:
                shuffle_layout_and_validate(
                    config,
                    train_ds,
                    sim_dict,
                    world_size,
                    expected_steps=len(train_loader),
                    context=f"post_epoch_{epoch}",
                )
            sync_samples(train_ds, rank)

        if rank == 0:
            state_path = f"{model_path}/training_state_latest.pth"
            save_training_state(
                state_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_score,
                train_ds,
                sim_dict,
                config,
            )
            print(f"Saved full training state: {state_path}")

    if rank == 0:
        torch.save(model.module.state_dict(), f"{model_path}/weights_end.pth")
        print(f"\nBest mean Recall@1: {best_score:.4f}\nDone.")

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train VIGOR-M MoE-FFN L1/L2/L3 model with DDP.")
    parser.add_argument("--download-pretrained-only", action="store_true")
    parser.add_argument("--data-levels", default=None)
    parser.add_argument("--dense-levels", default=None)
    parser.add_argument("--l1-stride-fraction", type=float, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--sim-sample", action="store_true")
    parser.add_argument("--no-sim-sample", action="store_true")
    parser.add_argument("--gps-sample", action="store_true")
    parser.add_argument("--no-gps-sample", action="store_true")
    parser.add_argument("--strict-cell-conflict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--same-area", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-eval", type=int, default=None)
    parser.add_argument("--eval-every-n-epoch", type=int, default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--smoke-shuffle", action="store_true")
    parser.add_argument("--smoke-shuffle-rounds", type=int, default=None)
    parser.add_argument("--shuffle-retry", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-workers-eval", type=int, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--moe-start-block", type=int, default=None)
    parser.add_argument("--expert-start-block", type=int, default=None,
                        help="Deprecated alias for --moe-start-block.")
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--router-jitter", type=float, default=None)
    parser.add_argument("--router-condition", choices=("none", "scale"), default=None)
    parser.add_argument(
        "--expert-layout",
        choices=("routed", "shared_level_private"),
        default=None,
    )
    parser.add_argument("--moe-aux-weight", type=float, default=None)
    parser.add_argument("--init-mode", choices=("pretrained", "level_ckpts", "checkpoint"), default=None)
    parser.add_argument("--shared-init", choices=("avg", "L1", "L2", "L3"), default=None)
    parser.add_argument("--checkpoint-start", default=None)
    parser.add_argument("--l1-checkpoint", default=None)
    parser.add_argument("--l2-checkpoint", default=None)
    parser.add_argument("--l3-checkpoint", default=None)
    parser.add_argument("--level-loss-weights", default=None)
    parser.add_argument("--loss-mode", choices=("levelwise", "plain"), default=None)
    parser.add_argument("--train-logit-scale", action="store_true")
    parser.add_argument("--data-folder", default=None)
    parser.add_argument("--metadata-folder", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-epoch", type=int, default=None)
    return parser.parse_args()


def apply_args(config, args):
    if args.data_levels is not None:
        config.data_levels = parse_levels(args.data_levels)
        config.eval_levels = config.data_levels
    if args.dense_levels is not None:
        config.dense_levels = parse_levels(args.dense_levels)
    if args.l1_stride_fraction is not None:
        config.l1_stride_fraction = args.l1_stride_fraction
    if args.steps_per_epoch is not None:
        config.steps_per_epoch = args.steps_per_epoch
    if args.sim_sample:
        config.sim_sample = True
    if args.no_sim_sample:
        config.sim_sample = False
    if args.gps_sample:
        config.gps_sample = True
    if args.no_gps_sample:
        config.gps_sample = False
    if args.strict_cell_conflict is not None:
        config.strict_cell_conflict = args.strict_cell_conflict
    if args.same_area is not None:
        config.same_area = args.same_area
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.batch_size_eval is not None:
        config.batch_size_eval = args.batch_size_eval
    if args.eval_every_n_epoch is not None:
        config.eval_every_n_epoch = args.eval_every_n_epoch
    if args.skip_final_eval:
        config.skip_final_eval = True
    if args.zero_shot:
        config.zero_shot = True
    if args.smoke_shuffle:
        config.smoke_shuffle = True
    if args.smoke_shuffle_rounds is not None:
        config.smoke_shuffle_rounds = args.smoke_shuffle_rounds
    if args.shuffle_retry is not None:
        config.shuffle_retry = args.shuffle_retry
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.num_workers_eval is not None:
        config.num_workers_eval = args.num_workers_eval
    if args.persistent_workers is not None:
        config.persistent_workers = args.persistent_workers
    if args.lr is not None:
        config.lr = args.lr
    if args.gpu_ids is not None:
        config.gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.master_port is not None:
        config.master_port = args.master_port
    if args.moe_start_block is not None:
        config.moe_start_block = args.moe_start_block
    if args.expert_start_block is not None:
        config.moe_start_block = args.expert_start_block
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.router_jitter is not None:
        config.router_jitter = args.router_jitter
    if args.router_condition is not None:
        config.router_condition = args.router_condition
    if args.expert_layout is not None:
        config.expert_layout = args.expert_layout
    if args.moe_aux_weight is not None:
        config.moe_aux_weight = args.moe_aux_weight
    if args.init_mode is not None:
        config.init_mode = args.init_mode
    if args.shared_init is not None:
        config.shared_init = args.shared_init
    if args.checkpoint_start is not None:
        config.checkpoint_start = args.checkpoint_start
    if args.l1_checkpoint is not None:
        config.level_ckpts["L1"] = args.l1_checkpoint
    if args.l2_checkpoint is not None:
        config.level_ckpts["L2"] = args.l2_checkpoint
    if args.l3_checkpoint is not None:
        config.level_ckpts["L3"] = args.l3_checkpoint
    if args.level_loss_weights is not None:
        config.level_loss_weights = tuple(float(x) for x in args.level_loss_weights.split(","))
    if args.loss_mode is not None:
        config.loss_mode = args.loss_mode
    if args.train_logit_scale:
        config.freeze_logit_scale = False
    if args.data_folder is not None:
        config.data_folder = args.data_folder
    if args.metadata_folder is not None:
        config.metadata_folder = args.metadata_folder
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.start_epoch is not None:
        config.start_epoch = args.start_epoch


def main():
    args = parse_args()
    config = Configuration()
    apply_args(config, args)

    if len(config.level_loss_weights) != len(config.data_levels):
        raise ValueError(
            f"level-loss-weights length {len(config.level_loss_weights)} must match "
            f"levels {config.data_levels}"
        )

    if config.smoke_shuffle:
        smoke_shuffle(config)
        return

    if args.download_pretrained_only:
        model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
        data_cfg = model.get_config()
        print(f"Cached pretrained model. mean={data_cfg['mean']} std={data_cfg['std']}")
        return

    gpu_ids = list(config.gpu_ids)
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("No CUDA GPUs visible.")
    gpu_ids = [gpu for gpu in gpu_ids if gpu < available]
    if len(gpu_ids) < 1:
        raise RuntimeError(f"No requested CUDA GPUs visible. Available={available}, requested={config.gpu_ids}")

    print(
        f"GPUs: {gpu_ids} World size: {len(gpu_ids)} "
        f"Batch/GPU: {config.batch_size} Effective batch: {config.batch_size * len(gpu_ids)} "
        f"Master port: {config.master_port}"
    )
    if len(gpu_ids) == 1:
        run(0, 1, config, gpu_ids)
    else:
        mp.spawn(run, args=(len(gpu_ids), config, gpu_ids), nprocs=len(gpu_ids), join=True)


if __name__ == "__main__":
    main()
