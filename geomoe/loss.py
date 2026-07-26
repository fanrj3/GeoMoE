"""Contrastive objectives used by GeoMoE training pipelines."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn
from torch.distributed.nn.functional import all_gather as all_gather_with_grad


class InfoNCE(nn.Module):
    """Symmetric in-batch InfoNCE for aligned query/reference pairs."""

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        """Return the mean query-to-reference and reference-to-query loss."""
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        logits_per_image2 = logits_per_image1.T

        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss = (self.loss_function(logits_per_image1, labels) +
                self.loss_function(logits_per_image2, labels)) / 2
        return loss


class LevelWiseInfoNCE(nn.Module):
    """InfoNCE grouped by level, optionally using DDP all-gather negatives.

    Samples from different levels never appear in the same similarity matrix.
    This is important for hierarchical all-in training where L1/L2/L3/L4 heads
    represent different retrieval spaces.
    """

    def __init__(
        self,
        loss_function,
        level_weights=None,
        distributed=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super().__init__()
        self.loss_function = loss_function
        self.distributed = distributed
        self.device = device
        self.level_weights = level_weights

    def _gather_features(self, tensor):
        """Gather features across ranks while preserving autograd edges."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return tensor
        return torch.cat(all_gather_with_grad(tensor), dim=0)

    def _gather_ids(self, tensor):
        """Gather non-differentiable level identifiers across DDP ranks."""
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _weight(self, level_id, device):
        if self.level_weights is None:
            return torch.tensor(1.0, device=device)
        if isinstance(self.level_weights, torch.Tensor):
            return self.level_weights.to(device=device, dtype=torch.float32)[int(level_id)]
        return torch.tensor(float(self.level_weights[int(level_id)]), device=device)

    def forward(self, image_features1, image_features2, logit_scale, level_ids, return_stats=False):
        """Compute one contrastive matrix per represented resolution level."""
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        level_ids = level_ids.to(image_features1.device, dtype=torch.long)

        all_features1 = self._gather_features(image_features1)
        all_features2 = self._gather_features(image_features2)
        all_level_ids = self._gather_ids(level_ids)

        local_batch = image_features1.size(0)
        if self.distributed and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            local_global_indices = torch.arange(
                local_batch, device=image_features1.device, dtype=torch.long
            ) + rank * local_batch
        else:
            local_global_indices = torch.arange(
                local_batch, device=image_features1.device, dtype=torch.long
            )

        total_loss = image_features1.new_tensor(0.0)
        total_weight = image_features1.new_tensor(0.0)
        stats = {}

        for level_id in torch.unique(level_ids).tolist():
            local_mask = level_ids == int(level_id)
            global_mask = all_level_ids == int(level_id)
            if int(local_mask.sum()) == 0 or int(global_mask.sum()) < 2:
                continue

            global_positions = torch.nonzero(global_mask, as_tuple=False).flatten()
            local_positive_indices = local_global_indices[local_mask]
            targets = (
                global_positions.unsqueeze(0) == local_positive_indices.unsqueeze(1)
            ).nonzero(as_tuple=False)[:, 1]

            q_local = image_features1[local_mask]
            r_local = image_features2[local_mask]
            q_global = all_features1[global_mask]
            r_global = all_features2[global_mask]

            logits_qr = logit_scale * q_local @ r_global.T
            logits_rq = logit_scale * r_local @ q_global.T
            level_loss = (
                self.loss_function(logits_qr, targets)
                + self.loss_function(logits_rq, targets)
            ) / 2

            weight = self._weight(level_id, image_features1.device)
            total_loss = total_loss + weight * level_loss
            total_weight = total_weight + weight
            stats[int(level_id)] = float(level_loss.detach().cpu())

        if total_weight.item() == 0:
            raise RuntimeError("No level had enough samples for LevelWiseInfoNCE.")

        loss = total_loss / total_weight
        if return_stats:
            return loss, stats
        return loss


class SpatialPriorInfoNCE(nn.Module):
    """InfoNCE where in-batch negatives are weighted by a spatial prior.

    For hierarchical retrieval: L2 training gets L1 similarities as prior weights.
    P(tile_j | query_i) ∝ exp(logit_{i,j}) * prior_{i,j}^λ

    prior_{i,j} = s1(query_i, parent_L1_of(tile_j))

    This lets the L2 model focus its capacity on distinguishing tiles within
    the same L1 parent, rather than wasting effort on spatially irrelevant tiles.
    """

    def __init__(self, prior_strength=1.0, label_smoothing=0.0, eps=1e-8,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.prior_strength = prior_strength
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale, prior_weights):
        """
        Args:
            image_features1: (B, D)  query features (street view)
            image_features2: (B, D)  reference features (satellite tiles)
            logit_scale:      scalar temperature
            prior_weights:    (B, B) prior weights.
                              prior_weights[i,j] = L1_similarity(query_i, parent_L1(tile_j))
        """
        B = image_features1.size(0)
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # Visual logits
        logits = logit_scale * image_features1 @ image_features2.T  # (B, B)

        # Add log-prior: log(P) ∝ visual_logit + λ * log(prior)
        log_prior = torch.log(prior_weights.clamp(min=self.eps))
        logits = logits + self.prior_strength * log_prior

        labels = torch.arange(B, dtype=torch.long, device=self.device)

        loss_12 = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_21 = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
        return (loss_12 + loss_21) / 2


class SoftInfoNCE(nn.Module):
    """InfoNCE with soft labels based on spatial distance.

    y_j = exp(-dist_j^2 / (2*sigma^2))  for dist_j <= max_dist
    y_j = 0                              for dist_j > max_dist

    dist is measured in tile units: pixel_distance / tile_unit
    where tile_unit = min(tile_w, tile_h) of the satellite crop.
    """

    def __init__(self, tile_unit, sigma=0.5, max_dist=2.0,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.tile_unit = tile_unit
        self.sigma = sigma
        self.max_dist = max_dist
        self.two_sigma2 = 2.0 * sigma * sigma
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale, coords):
        """
        Args:
            image_features1: (B, D) query features
            image_features2: (B, D) reference features
            logit_scale: scalar temperature
            coords: (B, 3) tensor [city_id, cx_px, cy_px]
        """
        B = image_features1.size(0)
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        # ── Spatial distance matrix ──
        city = coords[:, 0:1]             # (B, 1)
        cx = coords[:, 1:2]               # (B, 1)
        cy = coords[:, 2:3]               # (B, 1)

        same_city = (city == city.T).float()             # (B, B)
        dx = cx - cx.T                                    # (B, B)
        dy = cy - cy.T                                    # (B, B)
        pixel_dist = torch.sqrt(dx * dx + dy * dy)        # (B, B)
        tile_dist = pixel_dist / self.tile_unit

        # ── Soft labels ──
        y = torch.exp(-tile_dist * tile_dist / self.two_sigma2)
        y = y * same_city                                 # diff city → 0
        y = y * (tile_dist <= self.max_dist).float()      # dist > 2 → 0
        y.fill_diagonal_(1.0)                            # self-match = 1

        # Normalize each row to sum to 1
        y = y / (y.sum(dim=1, keepdim=True) + 1e-8)

        # ── Bidirectional soft-label cross-entropy ──
        logits_12 = logit_scale * image_features1 @ image_features2.T
        logits_21 = logits_12.T

        loss_12 = -(y * F.log_softmax(logits_12, dim=1)).sum(dim=1)
        loss_21 = -(y.T * F.log_softmax(logits_21, dim=1)).sum(dim=1)

        return (loss_12.mean() + loss_21.mean()) / 2
