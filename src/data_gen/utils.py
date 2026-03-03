
import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from turtle import down
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import CLIPModel, CLIPProcessor

from math import sqrt

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x.to(device) if hasattr(x, "to") else x

@torch.no_grad()
def pca_channel_per_sample(x: torch.Tensor, low_c: int, center: bool = True, whiten: bool = False, eps: float = 1e-6):
    assert x.dim() == 4, "x must be [B, C, H, W]"
    B, C, H, W = x.shape
    N = H * W
    k = min(low_c, min(C, N))

    # [B, C, N]
    X = x.reshape(B, C, N)
    orig_dtype = X.dtype
    X = X.float()  # Numerical stability

    if center:
        mean = X.mean(dim=2, keepdim=True)  # [B, C, 1]
        Xc = X - mean
    else:
        mean = None
        Xc = X

    # Perform per-sample SVD: Xc = U Σ V^T
    # U: [B, C, r], S: [B, r], Vh: [B, r, N], r = min(C, N)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    # Keep the first k principal components (first k columns of U)
    Uk = U[:, :, :k]             # [B, C, k]
    basis = Uk                   # Principal-component basis (one component per column)

    # Project to lower dimension: Y = Uk^T @ Xc  -> [B, k, N]
    Y = torch.matmul(Uk.transpose(1, 2), Xc)  # [B, k, N]

    if whiten:
        # Whiten by dividing with corresponding singular values
        Sk = S[:, :k].unsqueeze(-1)          # [B, k, 1]
        Y = Y / (Sk + eps)

    # Reshape back to [B, k, H, W]
    Y = Y.reshape(B, k, H, W).to(orig_dtype)

    return Y

# -------------------------------
# Activation Grabber (hooks UNet)
# -------------------------------
class ActivationGrabber:
    """Register forward-hooks on selected UNet decoder blocks and stash outputs.

    We read the *last ResNet* of several up_blocks to capture coarse-to-fine features.
    After each step, call `pop_last_pooled()` to obtain pooled vectors.
    """

    def __init__(self, unet: nn.Module, down_blocks: List[Tuple[int, int]], up_blocks: List[Tuple[int, int]]):
        # blocks: list of (up_block_index, resnet_index_within_block)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.latest_feats: List[torch.Tensor] = []
        for (bi, ri) in down_blocks:
            module = unet.down_blocks[bi].resnets[ri]
            h = module.register_forward_hook(self._hook_fn)
            self.handles.append(h)
        for (bi, ri) in up_blocks:
            module = unet.up_blocks[bi].resnets[ri]
            h = module.register_forward_hook(self._hook_fn)
            self.handles.append(h)
    def _hook_fn(self, module, inputs, output):
        self.latest_feats.append(output[0::2].detach().cpu())  # (B, C, H, W)

    def pop_last_pooled(self) -> Optional[torch.Tensor]:
        pooled = [f.mean(dim=(2, 3)) for f in self.latest_feats]  # GAP, shape (B, C)
        self.latest_feats.clear()
        return torch.cat(pooled, dim=1)  # (B, sum_C)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

class ActivationGrabber_upblock:
    """Register forward-hooks on selected UNet decoder blocks and stash outputs.

    We read the *last ResNet* of several up_blocks to capture coarse-to-fine features.
    After each step, call `pop_last_pooled()` to obtain pooled vectors.
    """

    def __init__(self, unet: nn.Module):
        # blocks: list of (up_block_index, resnet_index_within_block)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.latest_feats: List[torch.Tensor] = []
        module = unet.up_blocks[2].resnets[1]
        h = module.register_forward_hook(self._hook_fn)
        self.handles.append(h)

    def _hook_fn(self, module, inputs, output):
        self.latest_feats.append(output[0::2].detach())  # (B, C, H, W)

    def get_pca_feature(self) -> Optional[torch.Tensor]:
        feats = self.latest_feats[-1]
        self.latest_feats.clear()
        feats = pca_channel_per_sample(feats, low_c=48).cpu()
        return feats

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

class ActivationGrabber_SD3:
    """Register forward-hooks on selected UNet decoder blocks and stash outputs.

    We read the *last ResNet* of several up_blocks to capture coarse-to-fine features.
    After each step, call `pop_last_pooled()` to obtain pooled vectors.
    """

    def __init__(self, dit: nn.Module):
        # blocks: list of (up_block_index, resnet_index_within_block)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.latest_feats: List[torch.Tensor] = []
        module = dit.transformer_blocks[20]
        self.patch_size: int = dit.config.patch_size
        h = module.register_forward_hook(self._hook_fn)
        self.handles.append(h)

    def _hook_fn(self, module, inputs, output):
        ori_feat = output[1][0::2]
        bs, c = ori_feat.shape[0], ori_feat.shape[2]
        new_c = c // self.patch_size // self.patch_size
        h = w = int(sqrt(ori_feat.shape[1]))
        new_feat = ori_feat.view(bs, h, w, self.patch_size, self.patch_size, new_c).permute(0, 5, 1, 3, 2, 4).reshape(bs, new_c, h * self.patch_size, w * self.patch_size)
        new_feat = F.interpolate(new_feat, size=(48, 48), mode='bilinear', align_corners=False)
        self.latest_feats.append(new_feat.detach())  # (B, C, H, W)

    def get_pca_feature(self) -> Optional[torch.Tensor]:
        feats = self.latest_feats[-1]
        self.latest_feats.clear()
        feats = pca_channel_per_sample(feats, low_c=48).cpu()
        return feats

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
