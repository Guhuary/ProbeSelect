
import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

from ...utils import RankedLogger
logger = RankedLogger(__name__, rank_zero_only=True)

def create_data_splits(
    num_files: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42, 
    random: bool = True
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits of file indices.
    
    Args:
        num_files: Total number of files
        train_ratio: Ratio of files for training
        val_ratio: Ratio of files for validation
        test_ratio: Ratio of files for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Create shuffled indices
    rng = np.random.RandomState(seed)
    indices = list(range(num_files))
    if random:
        rng.shuffle(indices)
    
    # Calculate split points
    n_train = int(num_files * train_ratio)
    n_val = int(num_files * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    logger.info(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


# def create_dataloaders(
#     config,
#     train_indices: List[int],
#     val_indices: List[int],
#     test_indices: List[int]
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Create dataloaders for train/val/test sets.
    
#     Args:
#         config: Configuration object
#         train_indices: Training file indices
#         val_indices: Validation file indices
#         test_indices: Test file indices
        
#     Returns:
#         Tuple of (train_loader, val_loader, test_loader)
#     """
#     # Create datasets
#     train_dataset = LatentQualityDataset(
#         data_dir=config.data.data_dir,
#         stage='train', 
#         num_files=config.data.num_files,
#         file_indices=train_indices
#     )
#     val_dataset = LatentQualityDataset(
#         data_dir=config.data.data_dir,
#         stage='val', 
#         num_files=config.data.num_files,
#         file_indices=val_indices
#     )
#     test_dataset = LatentQualityDataset(
#         data_dir=config.data.data_dir,
#         stage='test', 
#         num_files=config.data.num_files,
#         file_indices=test_indices
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=config.data.shuffle_train,
#         num_workers=config.data.num_workers,
#         pin_memory=config.data.pin_memory,
#         persistent_workers=config.data.num_workers > 0
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=False,
#         num_workers=config.data.num_workers,
#         pin_memory=config.data.pin_memory,
#         persistent_workers=config.data.num_workers > 0
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=False,
#         num_workers=config.data.num_workers,
#         pin_memory=config.data.pin_memory,
#         persistent_workers=config.data.num_workers > 0
#     )
    
#     return train_loader, val_loader, test_loader

def custom_collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for key in keys:
        if key in ['prompt', 'image_paths']:
            continue
        out[key] = torch.cat([item[key] for item in batch], dim=0)
    return out
