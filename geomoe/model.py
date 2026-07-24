import copy

import torch
import timm
import numpy as np
import torch.nn as nn


def _level_value_from_name(level):
    text = str(level).strip().upper().replace("P", ".")
    if text.startswith("L"):
        text = text[1:]
    if "." not in text and len(text) == 2 and text[1] == "5":
        return float(text[0]) + 0.5
    return float(text)


class TimmModel(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 img_size=383):

        super(TimmModel, self).__init__()

        self.img_size = img_size

        if "vit" in model_name:
            # Allow the shared ViT backbone to process square satellite crops
            # and wider ground panoramas in the same batch flow.
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=img_size,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config


    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)


    def forward(self, img1, img2=None):

        if img2 is not None:

            image_features1 = self.model(img1)
            image_features2 = self.model(img2)

            return image_features1, image_features2

        else:
            image_features = self.model(img1)

            return image_features


class LevelSplitTimmModel(nn.Module):
    """ViT model with shared early blocks and level-specific late blocks.

    With ``expert_start_block=7``, blocks 0-6 are shared and blocks 7-11,
    final norm, and head are copied once per level.
    """

    def __init__(
        self,
        model_name,
        pretrained=True,
        img_size=383,
        levels=("L1", "L2", "L3", "L4"),
        expert_start_block=7,
        default_level="L4",
    ):
        super().__init__()

        self.img_size = img_size
        self.levels = tuple(str(level).upper() for level in levels)
        self.level_to_id = {level: idx for idx, level in enumerate(self.levels)}
        self.id_to_level = {idx: level for level, idx in self.level_to_id.items()}
        self.default_level = str(default_level).upper()
        self.expert_start_block = int(expert_start_block)

        if "vit" not in model_name:
            raise ValueError("LevelSplitTimmModel currently supports ViT/timm backbones only.")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

        total_blocks = len(self.model.blocks)
        if not (0 <= self.expert_start_block <= total_blocks):
            raise ValueError(
                f"expert_start_block must be in [0, {total_blocks}], "
                f"got {self.expert_start_block}."
            )

        tail_blocks = self.model.blocks[self.expert_start_block:]
        self.tail_blocks = nn.ModuleDict(
            {
                level: nn.ModuleList([copy.deepcopy(block) for block in tail_blocks])
                for level in self.levels
            }
        )
        self.tail_norms = nn.ModuleDict(
            {level: copy.deepcopy(self.model.norm) for level in self.levels}
        )
        self.tail_fc_norms = nn.ModuleDict(
            {level: copy.deepcopy(self.model.fc_norm) for level in self.levels}
        )
        self.tail_heads = nn.ModuleDict(
            {level: copy.deepcopy(self.model.head) for level in self.levels}
        )

        self.model.blocks = nn.Sequential(*list(self.model.blocks[:self.expert_start_block]))
        self.model.norm = nn.Identity()
        self.model.fc_norm = nn.Identity()
        self.model.head = nn.Identity()

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        return timm.data.resolve_model_data_config(self.model)

    def set_grad_checkpointing(self, enable=True):
        # The split tail is executed manually, so timm's checkpoint_seq path is
        # not used here. Keep this method for interface compatibility.
        if hasattr(self.model, "set_grad_checkpointing"):
            self.model.set_grad_checkpointing(enable)

    def _level_ids(self, levels, batch_size, device):
        if levels is None:
            level_id = self.level_to_id[self.default_level]
            return torch.full((batch_size,), level_id, dtype=torch.long, device=device)

        if isinstance(levels, str):
            level_id = self.level_to_id[levels.upper()]
            return torch.full((batch_size,), level_id, dtype=torch.long, device=device)

        if isinstance(levels, int):
            return torch.full((batch_size,), int(levels), dtype=torch.long, device=device)

        if isinstance(levels, torch.Tensor):
            return levels.to(device=device, dtype=torch.long)

        mapped = [
            self.level_to_id[item.upper()] if isinstance(item, str) else int(item)
            for item in levels
        ]
        return torch.tensor(mapped, dtype=torch.long, device=device)

    def encode(self, img, levels=None):
        level_ids = self._level_ids(levels, img.shape[0], img.device)

        x = self.model.patch_embed(img)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)

        out = None
        for level_id, level in self.id_to_level.items():
            mask = level_ids == level_id
            if not torch.any(mask):
                continue
            z = x[mask]
            for block in self.tail_blocks[level]:
                z = block(z)
            z = self.tail_norms[level](z)
            z = self.model.pool(z)
            z = self.tail_fc_norms[level](z)
            z = self.model.head_drop(z)
            z = self.tail_heads[level](z)
            if out is None:
                out = z.new_empty((img.shape[0], z.shape[-1]))
            out[mask] = z

        return out

    def forward(self, img1, img2=None, levels=None):
        if img2 is not None:
            image_features1 = self.encode(img1, levels=levels)
            image_features2 = self.encode(img2, levels=levels)
            return image_features1, image_features2

        return self.encode(img1, levels=levels)


class RoutedMoEFFN(nn.Module):
    """Top-k routed FFN experts using one router decision per image.

    The router reads the current CLS token and applies the same expert mixture
    to every token of that image. This keeps image-level routing stable while
    still allowing patch tokens to pass through level-specialized FFNs.
    """

    def __init__(
        self,
        experts,
        dim,
        top_k=2,
        router_jitter=0.0,
        router_condition_dim=0,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.top_k = int(top_k)
        self.router = nn.Linear(dim, self.num_experts)
        self.router_jitter = float(router_jitter)
        self.router_condition_dim = int(router_condition_dim)
        if self.router_condition_dim > 0:
            self.router_condition = nn.Sequential(
                nn.Linear(self.router_condition_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.router_condition[-1].weight)
            nn.init.zeros_(self.router_condition[-1].bias)
        else:
            self.router_condition = None
        nn.init.normal_(self.router.weight, std=1e-3)
        nn.init.zeros_(self.router.bias)

    def forward(self, x, router_condition=None):
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1, {self.num_experts}], got {self.top_k}")

        router_input = x[:, 0]
        if self.router_condition is not None:
            if router_condition is None:
                raise ValueError("router_condition is required when router_condition_dim > 0")
            condition = router_condition.to(device=x.device, dtype=x.dtype)
            router_input = router_input + self.router_condition(condition)

        router_logits = self.router(router_input)
        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(router_logits).uniform_(
                1.0 - self.router_jitter,
                1.0 + self.router_jitter,
            )
            router_logits = router_logits + torch.log(noise.clamp_min(1e-6))

        router_probs = torch.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        out = x.new_zeros(x.shape)
        top_mask = torch.zeros_like(router_probs)
        top_mask.scatter_(1, top_indices, 1.0)

        for expert_id, expert in enumerate(self.experts):
            sample_weights = torch.where(
                top_indices == expert_id,
                top_probs,
                torch.zeros_like(top_probs),
            ).sum(dim=1)
            selected = sample_weights > 0
            if torch.any(selected):
                expert_out = expert(x[selected])
                out[selected] = out[selected] + expert_out * sample_weights[selected].view(-1, 1, 1)

        # Switch-style load balancing. It is near 1.0 when experts are balanced.
        density_proxy = router_probs.mean(dim=0)
        density = top_mask.mean(dim=0) / float(self.top_k)
        aux_loss = self.num_experts * torch.sum(density_proxy * density)
        stats = {
            "prob": density_proxy.detach(),
            "load": density.detach(),
            "top1": torch.nn.functional.one_hot(
                top_indices[:, 0], num_classes=self.num_experts
            ).float().mean(dim=0).detach(),
            "aux": aux_loss.detach(),
        }
        return out, aux_loss, stats


class SharedLevelPrivateFFN(nn.Module):
    """One always-on shared expert plus one level-conditioned private branch.

    Expert 0 is shared. Experts 1..N correspond to the configured level
    anchors. Native levels activate one private expert, while intermediate
    levels linearly interpolate the two adjacent private experts.
    """

    def __init__(
        self,
        experts,
        dim,
        num_private_experts,
        router_jitter=0.0,
        router_condition_dim=0,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(self.experts)
        self.num_private_experts = int(num_private_experts)
        if self.num_experts != self.num_private_experts + 1:
            raise ValueError(
                "shared_level_private requires one shared expert plus one private "
                f"expert per level, got {self.num_experts} experts for "
                f"{self.num_private_experts} levels."
            )

        self.router = nn.Linear(dim, 2)
        self.router_jitter = float(router_jitter)
        self.router_condition_dim = int(router_condition_dim)
        if self.router_condition_dim > 0:
            self.router_condition = nn.Sequential(
                nn.Linear(self.router_condition_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.router_condition[-1].weight)
            nn.init.zeros_(self.router_condition[-1].bias)
        else:
            self.router_condition = None
        nn.init.normal_(self.router.weight, std=1e-3)
        nn.init.zeros_(self.router.bias)

    def forward(self, x, private_weights, router_condition=None):
        if private_weights is None:
            raise ValueError("private_weights is required for shared_level_private routing")
        if private_weights.shape != (x.shape[0], self.num_private_experts):
            raise ValueError(
                "private_weights must have shape "
                f"({x.shape[0]}, {self.num_private_experts}), got "
                f"{tuple(private_weights.shape)}."
            )

        router_input = x[:, 0]
        if self.router_condition is not None:
            if router_condition is None:
                raise ValueError("router_condition is required when router_condition_dim > 0")
            condition = router_condition.to(device=x.device, dtype=x.dtype)
            router_input = router_input + self.router_condition(condition)

        router_logits = self.router(router_input)
        if self.training and self.router_jitter > 0:
            noise = torch.empty_like(router_logits).uniform_(
                1.0 - self.router_jitter,
                1.0 + self.router_jitter,
            )
            router_logits = router_logits + torch.log(noise.clamp_min(1e-6))
        gate_probs = torch.softmax(router_logits, dim=-1)

        shared_out = self.experts[0](x)
        private_out = x.new_zeros(x.shape)
        private_weights = private_weights.to(device=x.device, dtype=x.dtype)
        for private_id, expert in enumerate(self.experts[1:]):
            sample_weights = private_weights[:, private_id]
            selected = sample_weights > 0
            if torch.any(selected):
                expert_out = expert(x[selected])
                private_out[selected] = (
                    private_out[selected]
                    + expert_out * sample_weights[selected].view(-1, 1, 1)
                )

        out = (
            shared_out * gate_probs[:, 0].view(-1, 1, 1)
            + private_out * gate_probs[:, 1].view(-1, 1, 1)
        )

        # Keep shared/private capacity balanced without forcing private levels
        # to balance against each other; the level-balanced sampler does that.
        gate_density = gate_probs.mean(dim=0)
        aux_loss = 2.0 * torch.sum(gate_density * gate_density)

        effective_probs = torch.cat(
            (gate_probs[:, :1], gate_probs[:, 1:2] * private_weights),
            dim=1,
        )
        active_private = (private_weights > 0).to(dtype=x.dtype)
        active = torch.cat((torch.ones_like(gate_probs[:, :1]), active_private), dim=1)
        load = (active / active.sum(dim=1, keepdim=True)).mean(dim=0)
        top1 = torch.nn.functional.one_hot(
            effective_probs.argmax(dim=1), num_classes=self.num_experts
        ).float().mean(dim=0)
        stats = {
            "prob": effective_probs.mean(dim=0).detach(),
            "load": load.detach(),
            "top1": top1.detach(),
            "aux": aux_loss.detach(),
            "shared_gate": gate_density[0].detach(),
            "private_gate": gate_density[1].detach(),
        }
        return out, aux_loss, stats


class MoEFFNBlock(nn.Module):
    """A timm ViT block with the MLP/FFN replaced by a routed MoE FFN."""

    def __init__(
        self,
        block,
        num_experts=4,
        top_k=2,
        router_jitter=0.0,
        router_condition_dim=0,
        expert_layout="routed",
        num_private_experts=0,
    ):
        super().__init__()
        self.norm1 = copy.deepcopy(block.norm1)
        self.attn = copy.deepcopy(block.attn)
        self.ls1 = copy.deepcopy(block.ls1)
        self.drop_path1 = copy.deepcopy(block.drop_path1)
        self.norm2 = copy.deepcopy(block.norm2)
        experts = [copy.deepcopy(block.mlp) for _ in range(int(num_experts))]
        self.expert_layout = str(expert_layout).lower()
        if self.expert_layout == "routed":
            self.moe = RoutedMoEFFN(
                experts=experts,
                dim=block.norm2.normalized_shape[0],
                top_k=top_k,
                router_jitter=router_jitter,
                router_condition_dim=router_condition_dim,
            )
        elif self.expert_layout == "shared_level_private":
            self.moe = SharedLevelPrivateFFN(
                experts=experts,
                dim=block.norm2.normalized_shape[0],
                num_private_experts=num_private_experts,
                router_jitter=router_jitter,
                router_condition_dim=router_condition_dim,
            )
        else:
            raise ValueError(f"Unsupported expert_layout: {expert_layout!r}")
        self.ls2 = copy.deepcopy(block.ls2)
        self.drop_path2 = copy.deepcopy(block.drop_path2)

    def forward(self, x, router_condition=None, private_weights=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if self.expert_layout == "shared_level_private":
            y, aux_loss, stats = self.moe(
                self.norm2(x),
                private_weights=private_weights,
                router_condition=router_condition,
            )
        else:
            y, aux_loss, stats = self.moe(
                self.norm2(x), router_condition=router_condition
            )
        x = x + self.drop_path2(self.ls2(y))
        return x, aux_loss, stats


class LevelMoEFFNTimmModel(nn.Module):
    """ViT model with a single attention path and MoE FFNs in late blocks."""

    def __init__(
        self,
        model_name,
        pretrained=True,
        img_size=383,
        levels=("L1", "L2", "L3"),
        moe_start_block=7,
        num_experts=4,
        top_k=2,
        router_jitter=0.0,
        router_condition="none",
        expert_layout="routed",
        default_level="L3",
    ):
        super().__init__()

        if "vit" not in model_name:
            raise ValueError("LevelMoEFFNTimmModel currently supports ViT/timm backbones only.")

        self.img_size = img_size
        self.levels = tuple(str(level).upper() for level in levels)
        self.level_to_id = {level: idx for idx, level in enumerate(self.levels)}
        self.default_level = str(default_level).upper()
        self.moe_start_block = int(moe_start_block)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_layout = str(expert_layout).lower()
        if self.expert_layout not in {"routed", "shared_level_private"}:
            raise ValueError(
                "expert_layout must be 'routed' or 'shared_level_private', "
                f"got {expert_layout!r}."
            )
        if self.expert_layout == "shared_level_private":
            expected_experts = len(self.levels) + 1
            if self.num_experts != expected_experts:
                raise ValueError(
                    "shared_level_private requires one shared expert plus one "
                    f"expert per level: expected {expected_experts}, got "
                    f"{self.num_experts}."
                )
            if self.top_k != 2:
                raise ValueError(
                    "shared_level_private executes shared + private and requires top_k=2."
                )
        self.router_condition_mode = str(router_condition).lower()
        if self.router_condition_mode not in {"none", "scale"}:
            raise ValueError(
                "router_condition must be 'none' or 'scale', "
                f"got {router_condition!r}."
            )
        self.router_condition_dim = 4 if self.router_condition_mode == "scale" else 0
        level_values = torch.tensor(
            [_level_value_from_name(level) for level in self.levels],
            dtype=torch.float32,
        )
        self.register_buffer("level_value_lut", level_values, persistent=False)
        if not torch.all(level_values[1:] > level_values[:-1]):
            raise ValueError(
                "levels must be ordered from coarse to fine for continuous "
                f"private routing, got {self.levels}."
            )
        self.level_value_min = float(level_values.min().item())
        self.level_value_max = float(level_values.max().item())

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

        total_blocks = len(self.model.blocks)
        if not (0 <= self.moe_start_block <= total_blocks):
            raise ValueError(
                f"moe_start_block must be in [0, {total_blocks}], got {self.moe_start_block}."
            )

        blocks = list(self.model.blocks)
        for idx in range(self.moe_start_block, total_blocks):
            blocks[idx] = MoEFFNBlock(
                blocks[idx],
                num_experts=self.num_experts,
                top_k=self.top_k,
                router_jitter=router_jitter,
                router_condition_dim=self.router_condition_dim,
                expert_layout=self.expert_layout,
                num_private_experts=len(self.levels),
            )
        self.model.blocks = nn.ModuleList(blocks)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.last_moe_stats = {}

    def get_config(self):
        return timm.data.resolve_model_data_config(self.model)

    def set_grad_checkpointing(self, enable=True):
        # This model executes blocks manually to collect MoE aux losses.
        # Keep this method for compatibility with existing train scripts.
        return None

    def _level_ids(self, levels, batch_size, device):
        if levels is None:
            level_id = self.level_to_id[self.default_level]
            return torch.full((batch_size,), level_id, dtype=torch.long, device=device)
        if isinstance(levels, str):
            level_id = self.level_to_id[levels.upper()]
            return torch.full((batch_size,), level_id, dtype=torch.long, device=device)
        if isinstance(levels, int):
            return torch.full((batch_size,), int(levels), dtype=torch.long, device=device)
        if isinstance(levels, torch.Tensor):
            return levels.to(device=device, dtype=torch.long)
        mapped = [
            self.level_to_id[item.upper()] if isinstance(item, str) else int(item)
            for item in levels
        ]
        return torch.tensor(mapped, dtype=torch.long, device=device)

    def _level_values(self, levels, batch_size, device):
        if levels is None:
            value = _level_value_from_name(self.default_level)
            return torch.full((batch_size,), value, dtype=torch.float32, device=device)
        if isinstance(levels, str):
            value = _level_value_from_name(levels)
            return torch.full((batch_size,), value, dtype=torch.float32, device=device)
        if isinstance(levels, int):
            if 0 <= int(levels) < len(self.level_value_lut):
                value = float(self.level_value_lut[int(levels)].item())
            else:
                value = float(levels)
            return torch.full((batch_size,), value, dtype=torch.float32, device=device)
        if isinstance(levels, float):
            return torch.full((batch_size,), float(levels), dtype=torch.float32, device=device)
        if isinstance(levels, torch.Tensor):
            levels = levels.to(device=device)
            if torch.is_floating_point(levels):
                return levels.to(dtype=torch.float32)
            level_ids = levels.to(dtype=torch.long)
            if level_ids.numel() > 0:
                min_id = int(level_ids.min().item())
                max_id = int(level_ids.max().item())
                if min_id < 0 or max_id >= len(self.level_value_lut):
                    return level_ids.to(dtype=torch.float32)
            return self.level_value_lut.to(device=device)[level_ids].to(dtype=torch.float32)
        values = []
        for item in levels:
            if isinstance(item, str):
                values.append(_level_value_from_name(item))
            elif isinstance(item, int) and 0 <= int(item) < len(self.level_value_lut):
                values.append(float(self.level_value_lut[int(item)].item()))
            else:
                values.append(float(item))
        return torch.tensor(values, dtype=torch.float32, device=device)

    def _router_condition_features(self, levels, batch_size, device):
        if self.router_condition_mode == "none":
            return None
        level_values = self._level_values(levels, batch_size, device)
        span = max(1e-6, self.level_value_max - self.level_value_min)
        z = (level_values - self.level_value_min) / span
        z = z.clamp(-1.0, 2.0)
        return torch.stack(
            (
                z,
                z * z,
                torch.sin(z * torch.pi),
                torch.cos(z * torch.pi),
            ),
            dim=-1,
        )

    def _private_expert_weights(self, level_values):
        anchors = self.level_value_lut.to(device=level_values.device)
        values = torch.maximum(torch.minimum(level_values, anchors[-1]), anchors[0])
        upper = torch.bucketize(values, anchors, right=False)
        upper = upper.clamp(1, len(anchors) - 1)
        lower = upper - 1
        lower_values = anchors[lower]
        upper_values = anchors[upper]
        alpha = (values - lower_values) / (upper_values - lower_values).clamp_min(1e-6)
        weights = values.new_zeros((values.shape[0], len(anchors)))
        weights.scatter_(1, lower.unsqueeze(1), (1.0 - alpha).unsqueeze(1))
        weights.scatter_add_(1, upper.unsqueeze(1), alpha.unsqueeze(1))
        return weights

    def encode(self, img, levels=None, return_moe_loss=False):
        router_condition = self._router_condition_features(levels, img.shape[0], img.device)
        private_weights = None
        if self.expert_layout == "shared_level_private":
            level_values = self._level_values(levels, img.shape[0], img.device)
            private_weights = self._private_expert_weights(level_values)

        x = self.model.patch_embed(img)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        aux_losses = []
        stats = {}
        for block_idx, block in enumerate(self.model.blocks):
            if isinstance(block, MoEFFNBlock):
                x, aux_loss, block_stats = block(
                    x,
                    router_condition=router_condition,
                    private_weights=private_weights,
                )
                aux_losses.append(aux_loss)
                stats[int(block_idx)] = block_stats
            else:
                x = block(x)

        x = self.model.norm(x)
        x = self.model.pool(x)
        x = self.model.fc_norm(x)
        x = self.model.head_drop(x)
        x = self.model.head(x)
        self.last_moe_stats = stats

        if return_moe_loss:
            if aux_losses:
                return x, torch.stack(aux_losses).mean()
            return x, x.new_tensor(0.0)
        return x

    def forward(self, img1, img2=None, levels=None, return_moe_loss=False):
        if img2 is not None:
            if return_moe_loss:
                image_features1, aux1 = self.encode(img1, levels=levels, return_moe_loss=True)
                image_features2, aux2 = self.encode(img2, levels=levels, return_moe_loss=True)
                return image_features1, image_features2, (aux1 + aux2) / 2
            image_features1 = self.encode(img1, levels=levels)
            image_features2 = self.encode(img2, levels=levels)
            return image_features1, image_features2

        return self.encode(img1, levels=levels, return_moe_loss=return_moe_loss)
