import os
import random, pickle
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import json
from datetime import datetime
from torch.amp import autocast

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..utils import RankedLogger
logger = RankedLogger(__name__, rank_zero_only=True)

def plot_predictions(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    save_path: Union[str, Path]
) -> None:
    """
    Plot predictions vs targets for quality metrics.
    
    Args:
        predictions: Dictionary of predicted values
        targets: Dictionary of target values
        save_path: Path to save the plot
    """
    metrics = predictions.keys()
    n_rows = len(metrics) // 3 + (len(metrics) % 3 > 0)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    # metrics = ['clip16', 'clip32', 'pick']  # 'as', 
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(targets[metric], predictions[metric], alpha=0.5, s=10)
        
        # Perfect prediction line
        # min_val = min(targets[metric].min(), predictions[metric].min())
        # max_val = max(targets[metric].max(), predictions[metric].max())
        # ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Calculate Correlation
        # from sklearn.metrics import r2_score
        r2 = np.corrcoef(targets[metric], predictions[metric])[0, 1]
        sp_r2 = stats.spearmanr(targets[metric], predictions[metric]).statistic
        ax.set_xlabel(f'True {metric}')
        ax.set_ylabel(f'Predicted {metric}')
        ax.set_title(f'{metric} (PearSonCorr = {r2:.3f}), Spearman Corr= {sp_r2:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Predictions plot saved to {save_path}")

@torch.no_grad()
def visualize_predictions(model, datamodule, path, keys=['clip16', 'clip32', 'as', 'pick']):
    # Collect predictions for visualization
    all_predictions = {key: [] for key in keys}
    all_targets = {key: [] for key in keys}
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    testset = datamodule.data_test
    test_loader = DataLoader(testset, batch_size=16, collate_fn=datamodule.collate_fn)
    # model = model.to(torch.bfloat16)
    
    for batch in tqdm(test_loader):
        # Move batch to device
        device = next(model.parameters()).device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)  # .float()
        # Get predictions
        with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(batch)
            predictions = {key: output[0] for key, output in predictions.items()}
        
        # Collect results
        for key in all_predictions.keys():
            all_predictions[key].extend(predictions[key].cpu().numpy())
            all_targets[key].extend(batch[key].cpu().numpy())

    # Convert to numpy arrays
    for key in all_predictions.keys():
        all_predictions[key] = np.array(all_predictions[key]) # type: ignore
        all_targets[key] = np.array(all_targets[key]) # type: ignore

    plot_predictions(
        predictions=all_predictions, # type: ignore
        targets=all_targets, # type: ignore
        save_path=path
    )