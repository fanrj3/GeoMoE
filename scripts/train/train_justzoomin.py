#!/usr/bin/env python3
"""Train a unified JustZoomIn model with level-routed MoE FFNs.

Released MoE split:
    shared attention/backbone: all blocks
    MoE FFN routing: block_11

The batch can contain L1/L2/L3/L4 samples for throughput, but InfoNCE is
computed separately per level. Samples from other levels never become negatives.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "geomoe").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from geomoe.datasets.justzoomin import (
    JustZoomInAllInDatasetEval,
    JustZoomInAllInDatasetTrain,
    JustZoomInDatasetEval,
    build_spatial_neighbor_dict,
    default_satellite_cache_dir,
    resolve_justzoomin_level,
)
from geomoe.loss import LevelWiseInfoNCE
from geomoe.model import LevelMoEFFNTimmModel, TimmModel
from geomoe.transforms import get_transforms_train, get_transforms_val
from geomoe.utils import Logger, setup_system


LEVELS = ("L1", "L2", "L3", "L4")

DEFAULT_CKPTS = {
    level: str(_PROJECT_ROOT / "weights" / "initialization" / "justzoomin" / f"{level}.pth")
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
    moe_aux_weight: float = 0.01
    init_mode: str = "pretrained"  # pretrained | level_ckpts | checkpoint
    shared_init: str = "avg"        # avg | L1 | L2 | L3 | L4
    checkpoint_start: str = None
    level_ckpts: dict = None
    start_epoch: int = 1
    freeze_logit_scale: bool = True

    mixed_precision: bool = True
    seed: int = 1
    epochs: int = 60
    batch_size: int = 128          # per-GPU batch, effective batch = batch_size * GPUs
    batch_size_eval: int = 192
    verbose: bool = True
    gpu_ids: tuple = (0,)
    master_port: int = 12358

    custom_sampling: bool = True
    gps_sample: bool = True
    sim_sample: bool = True
    neighbour_select: int = 64
    neighbour_range: int = 128
    gps_neighbor_block_size: int = 512

    eval_every_n_epoch: int = 4
    normalize_features: bool = True
    skip_final_eval: bool = False
    zero_shot: bool = False
    smoke_shuffle: bool = False
    smoke_shuffle_rounds: int = 12
    smoke_shuffle_checkpoint_sim: bool = False

    clip_grad: float = 100.0
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False
    label_smoothing: float = 0.1
    level_loss_weights: tuple = (1.0, 1.0, 1.0, 2.0)

    lr: float = 0.0001
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    lr_end: float = 0.00001

    data_folder: str = "./data/justzoomin"
    ground_cutting: int = 0
    data_levels: tuple = LEVELS
    eval_levels: tuple = LEVELS
    satellite_zoom: int = -3
    satellite_stride_fraction: float = 0.25
    dense_levels: tuple = ("L1", "L2")
    satellite_cache_levels: tuple = ("L1", "L2")
    use_satellite_cache: bool = True
    satellite_cache_dir: str = None
    satellite_cache_size: int = 384
    steps_per_epoch: int = None
    strict_l4_conflict: bool = True

    prob_rotate: float = 0.0
    prob_flip: float = 0.5

    model_path: str = "./outputs/checkpoints/justzoomin"
    run_name: str = None
    device: str = "cuda"
    num_workers: int = 8
    num_workers_eval: int = 2
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


def parse_levels(value):
    if value is None:
        return None
    return tuple(part.strip().upper() for part in value.split(",") if part.strip())


def parse_gpu_ids(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_stride_map(config):
    dense_levels = set(config.dense_levels)
    return {
        level: config.satellite_stride_fraction if level in dense_levels else None
        for level in config.data_levels
    }


def get_logit_scale(model):
    return model.module.logit_scale.exp() if hasattr(model, "module") else model.logit_scale.exp()


def level_ids_for(level, count, device, levels=LEVELS):
    return torch.full((count,), levels.index(level), dtype=torch.long, device=device)


def strip_module(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {key.replace("module.", ""): value for key, value in state.items()}


def load_level_ckpt(path):
    return strip_module(torch.load(path, map_location="cpu"))


def mean_tensors(tensors):
    return torch.stack([tensor.to(torch.float32) for tensor in tensors], dim=0).mean(dim=0).to(tensors[0].dtype)


def init_moe_ffn_model_from_level_ckpts(model, config, rank=0):
    level_states = {level: load_level_ckpt(config.level_ckpts[level]) for level in config.data_levels}
    expert_levels = list(config.data_levels)
    if config.num_experts < len(expert_levels) + 1:
        raise ValueError(
            f"num_experts={config.num_experts} is too small for {len(expert_levels)} level experts "
            "plus one base DINOv2 expert."
        )
    target = model.state_dict()
    new_state = {}

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
            if expert_idx < len(expert_levels):
                level = expert_levels[expert_idx]
                old_key = f"model.blocks.{block_idx}.mlp.{rest}"
                new_state[key] = level_states[level].get(old_key, value)
            else:
                new_state[key] = value
            continue

        if key.startswith("model."):
            source_values = [state[key] for state in level_states.values() if key in state]
            if config.shared_init.upper() in level_states and key in level_states[config.shared_init.upper()]:
                new_state[key] = level_states[config.shared_init.upper()][key]
            elif source_values:
                new_state[key] = mean_tensors(source_values)
            else:
                new_state[key] = value
        else:
            new_state[key] = value

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if rank == 0:
        print(
            "Initialized MoE-FFN model from level ckpts: "
            f"level_experts={expert_levels} base_experts={config.num_experts - len(expert_levels)} "
            f"shared_init={config.shared_init} missing={len(missing)} unexpected={len(unexpected)}"
        )


def summarize_level_batch(samples, global_batch_size):
    rows = []
    for start in range(0, len(samples), global_batch_size):
        batch = samples[start:start + global_batch_size]
        if len(batch) < global_batch_size:
            continue
        counts = {level: 0 for level in LEVELS}
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


def print_ddp_batch_stats(train_ds, local_batch_size, world_size, rank):
    if rank != 0:
        return
    global_batch_size = local_batch_size * world_size
    rows = summarize_level_batch(train_ds.samples, global_batch_size)
    if not rows:
        return
    avg_counts = {
        level: sum(row[0].get(level, 0) for row in rows) / len(rows)
        for level in LEVELS
    }
    max_dup_labels = max(row[1] for row in rows)
    max_dup_grounds = max(row[2] for row in rows)
    print(
        "DDP logical batch stats: "
        f"global_batch={global_batch_size} batches={len(rows)} "
        f"avg_level_counts={avg_counts} "
        f"max_dup_labels={max_dup_labels} max_dup_grounds={max_dup_grounds}"
    )


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
            f"{context}: custom sampler produced {steps} DDP batches, "
            f"expected {expected_steps}."
        )

    global_quota = train_ds._batch_quota(global_batch_size)
    local_quota = train_ds._batch_quota(local_batch_size)
    strict_l4 = bool(getattr(train_ds, "strict_l4_conflict", False))
    max_dup_labels = 0
    max_dup_grounds = 0
    max_dup_l4_cells = 0

    for batch_idx in range(steps):
        start = batch_idx * global_batch_size
        batch = train_ds.samples[start:start + global_batch_size]
        counts = {level: 0 for level in levels}
        used_ground = set()
        used_labels = set()
        used_l4_cells = set()
        dup_ground = 0
        dup_label = 0
        dup_l4 = 0

        for ground_idx, label, level in batch:
            if level not in counts:
                raise RuntimeError(f"{context}: unknown level {level!r} in batch {batch_idx}.")
            counts[level] += 1
            dup_ground += int(ground_idx in used_ground)
            dup_label += int(label in used_labels)
            used_ground.add(ground_idx)
            used_labels.add(label)
            if strict_l4:
                l4_cell = train_ds.idx2label_l4_cell[int(label)]
                dup_l4 += int(l4_cell in used_l4_cells)
                used_l4_cells.add(l4_cell)

        if counts != global_quota:
            raise RuntimeError(
                f"{context}: global batch {batch_idx} level quota mismatch: "
                f"{counts} vs {global_quota}."
            )
        if dup_ground or dup_label or dup_l4:
            raise RuntimeError(
                f"{context}: false-negative guard failed in global batch {batch_idx}: "
                f"dup_ground={dup_ground} dup_label={dup_label} dup_l4_cell={dup_l4}."
            )

        max_dup_grounds = max(max_dup_grounds, dup_ground)
        max_dup_labels = max(max_dup_labels, dup_label)
        max_dup_l4_cells = max(max_dup_l4_cells, dup_l4)

        for rank in range(world_size):
            rank_batch = batch[rank::world_size]
            rank_counts = {level: 0 for level in levels}
            for _ground_idx, _label, level in rank_batch:
                rank_counts[level] += 1
            if rank_counts != local_quota:
                raise RuntimeError(
                    f"{context}: rank {rank} local batch {batch_idx} level quota mismatch: "
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
        "max_dup_l4_cells": max_dup_l4_cells,
    }


def layout_samples_for_ddp(samples, local_batch_size, world_size, levels=LEVELS):
    """Reorder logical batches so each DDP rank gets a level-balanced local batch.

    DistributedSampler with ``shuffle=False`` assigns rank r the strided indices
    r, r + world_size, r + 2 * world_size, ...
    This layout fills those strided slots explicitly.
    """
    global_batch_size = local_batch_size * world_size
    if local_batch_size < len(levels):
        raise ValueError(
            f"local batch size {local_batch_size} must be >= number of levels {len(levels)}"
        )

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
            if len(bucket) % world_size != 0:
                raise ValueError(
                    f"Cannot split {len(bucket)} {level} samples across {world_size} ranks. "
                    "Use a global batch size divisible by number of levels and ranks."
                )
            cursor = 0
            quota = len(bucket) // world_size
            for rank in range(world_size):
                per_rank[rank].extend(bucket[cursor:cursor + quota])
                cursor += quota

        for rank_samples in per_rank:
            if len(rank_samples) != local_batch_size:
                raise ValueError(
                    f"Rank local batch has {len(rank_samples)} samples, expected {local_batch_size}."
                )

        for i in range(local_batch_size):
            for rank in range(world_size):
                reordered.append(per_rank[rank][i])

    return reordered


def shuffle_layout_and_validate(config, train_ds, sim_dict, world_size, expected_steps=None, context="shuffle"):
    train_ds.shuffle(
        sim_dict,
        neighbour_select=config.neighbour_select,
        neighbour_range=config.neighbour_range,
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


def sync_sim_dict(sim_dict, rank):
    payload = [sim_dict if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


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
        kwargs["persistent_workers"] = train and not config.custom_sampling
    return kwargs


def dist_predict(config, model, dataloader, level, rank, world_size):
    model.eval()
    local_feats, local_labels = [], []

    if rank == 0 and config.verbose:
        from tqdm import tqdm
        bar = tqdm(dataloader, total=len(dataloader), desc=f"Feat {level}")
    else:
        bar = dataloader

    level_id = LEVELS.index(level)
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
            print(f"\n{'='*30}[Evaluate {level}]{'='*30}")
        score = dist_evaluate(config, model, ref_loader, qry_loader, level, rank, world_size)
        if rank == 0:
            scores[level] = score
    if rank == 0:
        mean_score = sum(scores.values()) / max(1, len(scores))
        print(
            "MoE-FFN Eval Summary: "
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
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            continue

        if scheduler is not None:
            scheduler.step()

        if rank == 0 and config.verbose:
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                info=f"{info_losses.avg:.4f}",
                aux=f"{aux_losses.avg:.4f}",
                avg=f"{losses.avg:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

    if rank == 0 and config.verbose:
        bar.close()
    return losses.avg, info_losses.avg, aux_losses.avg


def build_model(config, rank):
    model = LevelMoEFFNTimmModel(
        config.model,
        pretrained=(config.init_mode != "checkpoint"),
        img_size=config.img_size,
        levels=config.data_levels,
        moe_start_block=config.moe_start_block,
        num_experts=config.num_experts,
        top_k=config.top_k,
        router_jitter=config.router_jitter,
        default_level="L4",
    )
    data_cfg = model.get_config()
    if config.init_mode == "level_ckpts":
        init_moe_ffn_model_from_level_ckpts(model, config, rank=rank)
    elif config.init_mode == "checkpoint":
        state = strip_module(torch.load(config.checkpoint_start, map_location="cpu"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if rank == 0:
            print(f"Loaded MoE-FFN checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    elif config.checkpoint_start:
        state = strip_module(torch.load(config.checkpoint_start, map_location="cpu"))
        model.load_state_dict(state, strict=False)
    return model, data_cfg


def build_train_dataset_for_shuffle(config, transforms_query=None, transforms_reference=None):
    if config.use_satellite_cache and config.satellite_cache_dir is None:
        config.satellite_cache_dir = str(default_satellite_cache_dir(config.data_folder))
    stride_map = build_stride_map(config)
    return JustZoomInAllInDatasetTrain(
        data_folder=config.data_folder,
        split="train",
        data_levels=config.data_levels,
        satellite_zoom=config.satellite_zoom,
        satellite_stride_fractions=stride_map,
        satellite_cache_dir=config.satellite_cache_dir if config.use_satellite_cache else None,
        satellite_cache_size=config.satellite_cache_size,
        satellite_cache_levels=config.satellite_cache_levels if config.use_satellite_cache else None,
        steps_per_epoch=config.steps_per_epoch,
        strict_l4_conflict=config.strict_l4_conflict,
        return_level_id=True,
        transforms_query=transforms_query,
        transforms_reference=transforms_reference,
        prob_flip=config.prob_flip,
        prob_rotate=config.prob_rotate,
        shuffle_batch_size=config.batch_size * len(config.gpu_ids),
    )


def build_label_cycle_sim_dict(train_ds, reverse=False):
    sim_dict = {}
    for level in train_ds.data_levels:
        labels = sorted(
            label for label in train_ds.idx2pairs
            if train_ds.idx2label_level[int(label)] == level
        )
        if reverse:
            labels = list(reversed(labels))
        if len(labels) <= 1:
            for label in labels:
                sim_dict[int(label)] = []
            continue
        limit = min(len(labels) - 1, train_ds.steps_per_epoch or len(labels))
        for idx, label in enumerate(labels):
            neighbours = []
            for offset in range(1, min(len(labels), 257)):
                neighbours.append(int(labels[(idx + offset) % len(labels)]))
                if len(neighbours) >= limit:
                    break
            sim_dict[int(label)] = neighbours
    return sim_dict


def smoke_shuffle(config):
    world_size = len(config.gpu_ids)
    logical_batch_size = config.batch_size * world_size
    print(
        f"Smoke shuffle: world_size={world_size} batch/GPU={config.batch_size} "
        f"global_batch={logical_batch_size} rounds={config.smoke_shuffle_rounds}"
    )
    train_ds = build_train_dataset_for_shuffle(config)

    sim_cases = [("none", None)]
    if config.gps_sample:
        gps_sim = build_spatial_neighbor_dict(
            train_ds.idx2tile_center,
            labels=train_ds.idx2pairs.keys(),
            top_k=config.neighbour_range,
            block_size=config.gps_neighbor_block_size,
        )
        sim_cases.append(("gps", gps_sim))
    sim_cases.append(("cycle", build_label_cycle_sim_dict(train_ds, reverse=False)))
    sim_cases.append(("cycle_reverse", build_label_cycle_sim_dict(train_ds, reverse=True)))

    fixed_steps = None
    for case_name, sim_dict in sim_cases:
        print(f"\n{'='*30}[Smoke Case: {case_name}]{'='*30}")
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


def smoke_checkpoint_sim_worker(rank, world_size, config, gpu_ids):
    gpu = gpu_ids[rank]
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu)
    setup_system(
        seed=config.seed + rank,
        cudnn_benchmark=config.cudnn_benchmark,
        cudnn_deterministic=config.cudnn_deterministic,
    )

    if config.use_satellite_cache and config.satellite_cache_dir is None:
        config.satellite_cache_dir = str(default_satellite_cache_dir(config.data_folder))

    train_ds = build_train_dataset_for_shuffle(config)
    if rank == 0:
        print(
            f"Smoke checkpoint-sim: checkpoint={config.checkpoint_start} "
            f"world_size={world_size} batch/GPU={config.batch_size} "
            f"eval_batch={config.batch_size_eval} rounds={config.smoke_shuffle_rounds}"
        )

    model, data_cfg = build_model(config, rank)
    model = model.cuda(gpu)
    model.eval()

    mean, std = data_cfg["mean"], data_cfg["std"]
    image_size_sat = (config.img_size, config.img_size)
    sat_tf_val, _gnd_tf_val = get_transforms_val(
        image_size_sat,
        image_size_sat,
        mean=mean,
        std=std,
        ground_cutting=config.ground_cutting,
    )

    stride_map = build_stride_map(config)
    ref_loaders_train = {}
    for level in config.data_levels:
        cfg = resolve_justzoomin_level(level)
        ref_ds = JustZoomInDatasetEval(
            data_folder=config.data_folder,
            split="train",
            img_type="reference",
            sequence_depth=cfg["sequence_depth"],
            satellite_zoom=config.satellite_zoom,
            satellite_crop_meters=cfg["satellite_crop_meters"],
            satellite_stride_fraction=stride_map.get(level),
            satellite_cache_dir=(
                config.satellite_cache_dir
                if config.use_satellite_cache and level in config.satellite_cache_levels
                else None
            ),
            satellite_cache_size=config.satellite_cache_size,
            transforms=sat_tf_val,
        )
        ref_sampler = DistributedSampler(ref_ds, num_replicas=world_size, rank=rank, shuffle=False)
        ref_loaders_train[level] = DataLoader(
            ref_ds,
            **dataloader_kwargs(config, config.batch_size_eval, sampler=ref_sampler),
        )
        if rank == 0:
            print(f"Smoke train ref {level}: {len(ref_ds)} refs")

    initial_sim_dict = None
    if config.gps_sample:
        initial_sim_dict = build_spatial_neighbor_dict(
            train_ds.idx2tile_center,
            labels=train_ds.idx2pairs.keys(),
            top_k=config.neighbour_range,
            block_size=config.gps_neighbor_block_size,
        )
        if rank == 0:
            print(f"Smoke checkpoint-sim initial GPS Sample: labels={len(initial_sim_dict)}")
    initial_sim_dict = sync_sim_dict(initial_sim_dict, rank)

    fixed_steps = None
    if rank == 0:
        stats = shuffle_layout_and_validate(
            config,
            train_ds,
            initial_sim_dict,
            world_size,
            expected_steps=None,
            context="smoke_checkpoint_sim_initial",
        )
        fixed_steps = stats["steps"]
        train_ds.steps_per_epoch = fixed_steps
        print(f"Smoke checkpoint-sim fixed steps per epoch: {fixed_steps}")
    sync_samples(train_ds, rank)
    if rank != 0:
        train_ds.steps_per_epoch = len(train_ds.samples) // (config.batch_size * world_size)
    dist.barrier()

    sim_dict = calc_ref_sim(
        config,
        model,
        ref_loaders_train,
        train_ds.level_offsets,
        rank,
        world_size,
    )
    sim_dict = sync_sim_dict(sim_dict, rank)

    if rank == 0:
        for round_idx in range(config.smoke_shuffle_rounds):
            shuffle_layout_and_validate(
                config,
                train_ds,
                sim_dict,
                world_size,
                expected_steps=fixed_steps,
                context=f"smoke_checkpoint_sim_round_{round_idx + 1}",
            )

    sync_samples(train_ds, rank)
    dist.barrier()
    if rank == 0:
        print("\nSmoke checkpoint-sim passed.")
    dist.destroy_process_group()


def smoke_checkpoint_sim(config):
    if not config.checkpoint_start:
        raise ValueError("--smoke-shuffle-checkpoint-sim requires --checkpoint-start.")
    config.init_mode = "checkpoint"
    config.sim_sample = True
    gpu_ids = list(config.gpu_ids)
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("No CUDA GPUs visible.")
    for gpu in gpu_ids:
        if gpu < 0 or gpu >= available:
            raise ValueError(f"Requested GPU {gpu}, but only {available} CUDA device(s) are visible.")
    mp.spawn(
        smoke_checkpoint_sim_worker,
        args=(len(gpu_ids), config, gpu_ids),
        nprocs=len(gpu_ids),
        join=True,
    )


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

    if config.use_satellite_cache and config.satellite_cache_dir is None:
        config.satellite_cache_dir = str(default_satellite_cache_dir(config.data_folder))
    stride_map = build_stride_map(config)

    level_tag = "".join(config.data_levels)
    dense_tag = "dense" + "".join(config.dense_levels) + f"_stride{config.satellite_stride_fraction:g}"
    model_path = "{}/moeffnB{}_E{}_top{}_{}_{}_z{}/aux{:.4g}/{}/{}".format(
        config.model_path,
        config.moe_start_block,
        config.num_experts,
        config.top_k,
        level_tag,
        dense_tag,
        config.satellite_zoom,
        config.moe_aux_weight,
        config.model,
        config.run_name or time.strftime("%H%M%S"),
    )
    if rank == 0:
        os.makedirs(model_path, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve(), f"{model_path}/train.py")
        shutil.copyfile(_PROJECT_ROOT / "geomoe" / "model.py", f"{model_path}/model.py")
        sys.stdout = Logger(os.path.join(model_path, "log.txt"))
        print(f"Output path: {model_path}")

    model, data_cfg = build_model(config, rank)
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

    import signal
    def emergency_save(_sig, _frame):
        path = f"{model_path}/emergency_rank{rank}.pth"
        torch.save(model.module.state_dict(), path)
        print(f"\n[Rank {rank}] EMERGENCY SAVED: {path}", flush=True)
    signal.signal(signal.SIGUSR1, emergency_save)

    if rank == 0:
        print(
            f"\nModel: {config.model} MoE-FFN JustZoomIn levels={config.data_levels} "
            f"shared_attention_blocks=0..11 moe_ffn_blocks={config.moe_start_block}..11 "
            f"experts={config.num_experts} top_k={config.top_k} GPUs={world_size}"
        )
        print(
            f"Batch/GPU: {config.batch_size} Effective batch: {config.batch_size * world_size} "
            f"Level-wise InfoNCE: no cross-level negatives"
        )
        print(
            f"Init mode: {config.init_mode} shared_init={config.shared_init} "
            f"moe_aux_weight={config.moe_aux_weight} router_jitter={config.router_jitter}"
        )

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
        cfg = resolve_justzoomin_level(level)
        ref_ds = JustZoomInDatasetEval(
            data_folder=config.data_folder,
            split="val",
            img_type="reference",
            sequence_depth=cfg["sequence_depth"],
            satellite_zoom=config.satellite_zoom,
            satellite_crop_meters=cfg["satellite_crop_meters"],
            satellite_stride_fraction=stride_map.get(level),
            satellite_cache_dir=(
                config.satellite_cache_dir
                if config.use_satellite_cache and level in config.satellite_cache_levels
                else None
            ),
            satellite_cache_size=config.satellite_cache_size,
            transforms=sat_tf_val,
        )
        qry_ds = JustZoomInDatasetEval(
            data_folder=config.data_folder,
            split="val",
            img_type="query",
            sequence_depth=cfg["sequence_depth"],
            satellite_zoom=config.satellite_zoom,
            satellite_crop_meters=cfg["satellite_crop_meters"],
            satellite_stride_fraction=stride_map.get(level),
            satellite_cache_dir=(
                config.satellite_cache_dir
                if config.use_satellite_cache and level in config.satellite_cache_levels
                else None
            ),
            satellite_cache_size=config.satellite_cache_size,
            transforms=gnd_tf_val,
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
            cfg = resolve_justzoomin_level(level)
            ref_ds = JustZoomInDatasetEval(
                data_folder=config.data_folder,
                split="train",
                img_type="reference",
                sequence_depth=cfg["sequence_depth"],
                satellite_zoom=config.satellite_zoom,
                satellite_crop_meters=cfg["satellite_crop_meters"],
                satellite_stride_fraction=stride_map.get(level),
                satellite_cache_dir=(
                    config.satellite_cache_dir
                    if config.use_satellite_cache and level in config.satellite_cache_levels
                    else None
                ),
                satellite_cache_size=config.satellite_cache_size,
                transforms=sat_tf_val,
            )
            ref_sampler = DistributedSampler(ref_ds, num_replicas=world_size, rank=rank, shuffle=False)
            ref_loaders_train[level] = DataLoader(
                ref_ds, **dataloader_kwargs(config, config.batch_size_eval, sampler=ref_sampler)
            )
            if rank == 0:
                print(f"Train ref {level}: {len(ref_ds)} refs")

    sim_dict = None
    if config.gps_sample:
        sim_dict = build_spatial_neighbor_dict(
            train_ds.idx2tile_center,
            labels=train_ds.idx2pairs.keys(),
            top_k=config.neighbour_range,
            block_size=config.gps_neighbor_block_size,
        )
        if rank == 0:
            print(f"Spatial GPS Sample: labels={len(sim_dict)} topk={config.neighbour_range}")
    sim_dict = sync_sim_dict(sim_dict, rank)

    if config.zero_shot:
        if rank == 0:
            print(f"\n{'='*30}[Zero Shot]{'='*30}")
        evaluate_all_levels(config, model.module, eval_loaders, rank, world_size)
        if config.sim_sample:
            sim_dict = calc_ref_sim(
                config, model.module, ref_loaders_train, train_ds.level_offsets, rank, world_size
            )
            sim_dict = sync_sim_dict(sim_dict, rank)

    if config.custom_sampling:
        if rank == 0:
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
        print_ddp_batch_stats(train_ds, config.batch_size, world_size, rank)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        **dataloader_kwargs(config, config.batch_size, sampler=train_sampler, train=True),
    )

    loss_fn_ce = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    level_weights = torch.tensor(config.level_loss_weights, dtype=torch.float32)
    loss_function = LevelWiseInfoNCE(
        loss_function=loss_fn_ce,
        level_weights=level_weights,
        distributed=True,
        device=f"cuda:{gpu}",
    )
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
    if scheduler is not None and config.start_epoch > 1:
        resume_steps = (config.start_epoch - 1) * len(train_loader)
        for _ in range(resume_steps):
            scheduler.step()
        if rank == 0:
            print(f"Advanced scheduler by {resume_steps} steps for start_epoch={config.start_epoch}")
    if rank == 0:
        print(f"Scheduler: {config.scheduler} Warmup: {warmup_steps} Total: {total_steps}")

    best_score = 0.0
    loss_csv_path = os.path.join(model_path, "loss.csv") if rank == 0 else None
    if rank == 0:
        csv_mode = "a" if config.start_epoch > 1 and os.path.exists(loss_csv_path) else "w"
        with open(loss_csv_path, csv_mode, newline="") as f:
            writer = csv.writer(f)
            if csv_mode == "w":
                writer.writerow(["epoch", "train_loss", "info_loss", "moe_aux", "lr", "mean_recall@1"] +
                                [f"{level}_recall@1" for level in config.eval_levels])

    for epoch in range(config.start_epoch, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        dist.barrier()
        if rank == 0:
            print(f"\n{'='*30}[Epoch: {epoch}]{'='*30}")

        train_loss, info_loss, moe_aux = train_epoch(
            config, model, train_loader, loss_function, optimizer, scheduler, scaler, epoch, rank, gpu
        )
        if rank == 0:
            print(
                f"Train Loss: {train_loss:.3f} Info: {info_loss:.3f} "
                f"MoE Aux: {moe_aux:.3f} LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
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
                    [epoch, f"{train_loss:.4f}", f"{info_loss:.4f}", f"{moe_aux:.4f}",
                     f"{optimizer.param_groups[0]['lr']:.6f}",
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
            print_ddp_batch_stats(train_ds, config.batch_size, world_size, rank)

    if rank == 0:
        torch.save(model.module.state_dict(), f"{model_path}/weights_end.pth")
        print(f"\nBest mean Recall@1: {best_score:.4f}\nDone.")

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train JustZoomIn MoE-FFN all-level model with DDP.")
    parser.add_argument("--download-pretrained-only", action="store_true")
    parser.add_argument("--data-levels", default=None)
    parser.add_argument("--dense-levels", default=None)
    parser.add_argument("--satellite-cache-levels", default=None)
    parser.add_argument("--satellite-stride-fraction", type=float, default=None)
    parser.add_argument("--satellite-zoom", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--sim-sample", action="store_true")
    parser.add_argument("--no-sim-sample", action="store_true")
    parser.add_argument("--gps-sample", action="store_true")
    parser.add_argument("--no-gps-sample", action="store_true")
    parser.add_argument("--use-satellite-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--satellite-cache-dir", default=None)
    parser.add_argument("--strict-l4-conflict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-eval", type=int, default=None)
    parser.add_argument("--eval-every-n-epoch", type=int, default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--smoke-shuffle", action="store_true")
    parser.add_argument("--smoke-shuffle-rounds", type=int, default=None)
    parser.add_argument("--smoke-shuffle-checkpoint-sim", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-workers-eval", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--moe-start-block", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--router-jitter", type=float, default=None)
    parser.add_argument("--moe-aux-weight", type=float, default=None)
    parser.add_argument("--init-mode", choices=("pretrained", "level_ckpts", "checkpoint"), default=None)
    parser.add_argument("--shared-init", choices=("avg", "L1", "L2", "L3", "L4"), default=None)
    parser.add_argument("--checkpoint-start", default=None)
    parser.add_argument("--l1-checkpoint", default=None)
    parser.add_argument("--l2-checkpoint", default=None)
    parser.add_argument("--l3-checkpoint", default=None)
    parser.add_argument("--l4-checkpoint", default=None)
    parser.add_argument("--level-loss-weights", default=None)
    parser.add_argument("--train-logit-scale", action="store_true")
    parser.add_argument("--data-folder", default=None)
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
    if args.satellite_cache_levels is not None:
        config.satellite_cache_levels = parse_levels(args.satellite_cache_levels)
    if args.satellite_stride_fraction is not None:
        config.satellite_stride_fraction = args.satellite_stride_fraction
    if args.satellite_zoom is not None:
        config.satellite_zoom = args.satellite_zoom
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
    if args.use_satellite_cache is not None:
        config.use_satellite_cache = args.use_satellite_cache
    if args.satellite_cache_dir is not None:
        config.satellite_cache_dir = args.satellite_cache_dir
    if args.strict_l4_conflict is not None:
        config.strict_l4_conflict = args.strict_l4_conflict
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
    if args.smoke_shuffle_checkpoint_sim:
        config.smoke_shuffle_checkpoint_sim = True
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.num_workers_eval is not None:
        config.num_workers_eval = args.num_workers_eval
    if args.lr is not None:
        config.lr = args.lr
    if args.gpu_ids is not None:
        config.gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.master_port is not None:
        config.master_port = args.master_port
    if args.moe_start_block is not None:
        config.moe_start_block = args.moe_start_block
    if args.num_experts is not None:
        config.num_experts = args.num_experts
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.router_jitter is not None:
        config.router_jitter = args.router_jitter
    if args.moe_aux_weight is not None:
        config.moe_aux_weight = args.moe_aux_weight
    if args.init_mode is not None:
        config.init_mode = args.init_mode
    if args.shared_init is not None:
        config.shared_init = args.shared_init
    if args.checkpoint_start is not None:
        config.checkpoint_start = args.checkpoint_start
    for level in LEVELS:
        value = getattr(args, f"{level.lower()}_checkpoint")
        if value is not None:
            config.level_ckpts[level] = value
    if args.level_loss_weights is not None:
        config.level_loss_weights = tuple(float(x) for x in args.level_loss_weights.split(","))
    if args.train_logit_scale:
        config.freeze_logit_scale = False
    if args.data_folder is not None:
        config.data_folder = args.data_folder
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

    if config.smoke_shuffle_checkpoint_sim:
        smoke_checkpoint_sim(config)
        return

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
        raise RuntimeError(
            f"No requested CUDA GPU is visible. Available={available}, requested={config.gpu_ids}"
        )

    print(
        f"GPUs: {gpu_ids} World size: {len(gpu_ids)} "
        f"Batch/GPU: {config.batch_size} Effective batch: {config.batch_size * len(gpu_ids)}"
    )
    if len(gpu_ids) == 1:
        run(0, 1, config, gpu_ids)
    else:
        mp.spawn(run, args=(len(gpu_ids), config, gpu_ids), nprocs=len(gpu_ids), join=True)


if __name__ == "__main__":
    main()
