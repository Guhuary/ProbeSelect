import os
from typing import Any, Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
from pathlib import Path
import pickle 
from operator import iconcat
from functools import reduce, partial
from tqdm import trange
import shutil

from .text_embeder import * 

from lightning import LightningDataModule
from torchvision.transforms import transforms, v2

from ...utils import RankedLogger
logger = RankedLogger(__name__, rank_zero_only=True)


class LatentQualityDataset(Dataset):
    """Dataset for loading latent features and corresponding quality metrics."""
    def __init__(
        self,
        data_dir: str,
        stage: str, 
        file_indices: List[int],
        transform: Optional[callable] = None, # type: ignore
        start: int = 19,
        end: int = 49,
        num_sampling_steps: int = 50,
        known_ts: List[float] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96],
        priors: Dict[str, List] = None, # type: ignore
        keys: List[str] = None, # type: ignore
        fix_time: bool | float = False # type: ignore
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing .pt files
            file_indices: List of file indices to use
            use_last_latent: Whether to use only the last latent timestep
            transform: Optional transform to apply to latent features
        """
        self.data_dir = Path(data_dir)
        self.file_indices = file_indices
        self.stage = stage
        self.transform = transform
        self.start = start 
        self.end = end
        self.num_sampling_steps = num_sampling_steps
        self.priors = priors
        self.keys = keys
        self.known_ts = torch.tensor(known_ts)
        self.num_t = len(self.known_ts)
        self.t_min = self.known_ts.min()
        self.t_max = self.known_ts.max()
        self.fix_time = fix_time
        if fix_time:
            self.fix_time = torch.tensor([fix_time])

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_indices)
    
    def __getitem__(self, sample_idx: int) -> Dict[str, torch.Tensor]:
        file_idx = self.file_indices[sample_idx]
        # try:
        #     data = torch.load(self.data_dir / f"{file_idx}.pt")
        # except:
        #     print(f"Loading: {self.data_dir / f'{file_idx}.pt'}")  # 加这一行，看是哪个文件报错
        #     raise ValueError(f"Loading: {self.data_dir / f'{file_idx}.pt'}")
        # for k in data.keys():
        #     if isinstance(data[k], torch.Tensor):
        #         if data[k].requires_grad:
        #             raise ValueError(f"Tensor {k} requires grad with id {self.data_dir}, {file_idx}")
        
        data = torch.load(self.data_dir / f"{file_idx}.pt")
        latent = data['records']
        bs = latent[0]['feat'].shape[0]
        if self.fix_time:
            t = self.fix_time
            t_idx_left = torch.searchsorted(self.known_ts, t) - 1
            assert self.known_ts[t_idx_left] <= t <= self.known_ts[t_idx_left + 1], f"t: {t}, t_idx_left: {t_idx_left}, t_idx_left + 1: {t_idx_left + 1}"
            data['time'] = t.repeat(bs) # type: ignore
            data['records'] = latent[t_idx_left]['feat'] + \
                (t - self.known_ts[t_idx_left]) / (self.known_ts[t_idx_left + 1] - self.known_ts[t_idx_left]) * \
                (latent[t_idx_left + 1]['feat'] - latent[t_idx_left]['feat'])
        else:
            if torch.rand(1) < 0.5:
                t_idx = torch.randint(self.num_t, size=(1, ))
                # t = self.known_ts[t_idx]
                data['records'] = latent[t_idx]['feat']
                data['time'] = (torch.tensor(latent[t_idx]['t'])).repeat(bs)
            else:
                t = torch.rand(size=(1, )) * (self.t_max - self.t_min) + self.t_min
                t_idx_left = torch.searchsorted(self.known_ts, t) - 1
                assert self.known_ts[t_idx_left] <= t <= self.known_ts[t_idx_left + 1], f"t: {t}, t_idx_left: {t_idx_left}, t_idx_left + 1: {t_idx_left + 1}"
                data['time'] = t.repeat(bs)
                data['records'] = latent[t_idx_left]['feat'] + \
                    (t - self.known_ts[t_idx_left]) / (self.known_ts[t_idx_left + 1] - self.known_ts[t_idx_left]) * \
                    (latent[t_idx_left + 1]['feat'] - latent[t_idx_left]['feat'])
        data['records'] = data['records'].clamp(min=-500, max=500)
        data['as_t'] = data['BLIP_ITC_t'].clone()
        for key in self.keys:
            data[key] = (data[key] - self.priors[key][0]) / self.priors[key][1]
        
        return data
