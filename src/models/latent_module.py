"""PyTorch Lightning module for quality prediction."""
from ast import mod
import sys, copy
from doctest import debug
import hydra
from sympy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from typing import Dict, List, Optional, Any

from torchmetrics import SpearmanCorrCoef # MeanAbsoluteError, MeanSquaredError, R2Score, 
from omegaconf import DictConfig

from ..utils import RankedLogger
from .components.EMA import ExponentialMovingAverage
# from .loss.rankloss import RankingLoss
logger = RankedLogger(__name__, rank_zero_only=True)


class QualityPredictionModule(pl.LightningModule):
    """Lightning module for training quality prediction models."""
    
    def __init__(self, args): # type: ignore
        super().__init__()
        
        self.save_hyperparameters()
        self.args = args

        self.model = hydra.utils.instantiate(args.net_para) # type: ignore
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_params}")
        # assert 0==1
        if self.args.use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(self.args.ema_rate)
            self.ema = torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn=avg_fn)
            for param in self.ema.parameters():
                param.requires_grad = False
        # Metrics
        self.keys = args.used_keys

        # Loss function
        self.criterion = {key: hydra.utils.instantiate(args.loss.get(key)) for key in self.keys}

        self.metrics = nn.ModuleDict({
            f'{key}_spearman': SpearmanCorrCoef() for key in self.keys
        })

        # self.val_predictions, self.val_targets = [], []
        # self.test_predictions, self.test_targets = [], []
        
        # For tracking best metrics
        self.best_val_loss = float('inf')
        
    def forward(self, batch: Dict, training=False) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        if not training and self.ema is not None:
            model = self.ema
        else:
            model = self.model
        return model(batch)
    
    def _compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor], 
        text_embs: Dict[str, torch.Tensor | None],
        epoch: int | None = 1,
        max_epoch: int | None = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all quality metrics.
        Args:
            predictions: Dictionary of predicted values
            targets: Dictionary of target values
            epoch: Current epoch number
            max_epoch: Total number of epochs
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Compute individual losses
        for key in self.keys:
            prediction, matrix = outputs[key]
            text_emb = text_embs[key]
            target = targets[key]
            loss_dict = self.criterion[key](predictions=prediction, matrix=matrix, text_emb=text_emb, 
                                            targets=target, epoch=epoch, max_epoch=max_epoch)
            for sub_k,v in loss_dict.items():
                losses[f'{key}_{sub_k}'] = v
        
        # Compute weighted total loss
        # You can adjust these weights based on metric importance
        weights = {
            'clip16': 1.0,
            'clip32': 1.0,
            'as': 1.0,
            'pick': 1.0,
            'ICT': 1.0,
            'HP': 1.0,
            'ICT_HP': 1.0,
            'BLIP_ITC': 1.0,
            'BLIP_ITM': 1.0,
            'ImgReward': 1.0,
            'HPSv20': 1.0,
            'HPSv21': 1.0,
        }
        
        total_loss = sum(
            weights[key] * losses[f'{key}_loss'] 
            for key in self.keys
        )
        losses['total_loss'] = total_loss
        
        return losses

    def general_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, epoch=1, max_epoch=1, training=False):
        """General step for training and validation."""
        targets = {
            key: batch[key] 
            for key in self.keys
        }
        text_embs = {
            key: batch.get(f'{key}_t', None)
            for key in self.keys
        }

        # Forward pass
        outputs = self(batch, training)
        predictions = {key: output[0] for key, output in outputs.items()}
        # Compute losses
        losses = self._compute_loss(outputs, targets, text_embs, epoch, max_epoch)
        return losses, predictions, targets
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""

        losses, predictions, targets = self.general_step(batch, batch_idx, self.current_epoch, self.trainer.max_epochs, training=True) # type: ignore
        batch_size = batch['records'].size(0)
        # Compute losses
        
        # Log losses
        for key, value in losses.items():
            self.log(f'train/{key}', value, on_step=False, on_epoch=True, prog_bar=key == 'total_loss', batch_size=batch_size, sync_dist=True)
        
        # Log learning rate
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True) # type: ignore
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        # Extract inputs and targets
        losses, predictions, targets = self.general_step(batch, batch_idx, self.current_epoch, self.trainer.max_epochs) # type: ignore
        batch_size = batch['records'].size(0)

        # Log losses
        for key, value in losses.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=key == 'total_loss', batch_size=batch_size, sync_dist=True)

        for key in self.keys:
            pres = predictions[key]
            tars = targets[key]
            self.metrics[f'{key}_spearman'].update(pres, tars) # type: ignore
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Log overall metrics
        val = 0
        for metric_name, metric in self.metrics.items():
            value = metric.compute() # type: ignore
            self.log(f'val/{metric_name}', value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            val += value
            metric.reset() # type: ignore
        self.log('val/total_corr', val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        # Extract inputs and targets
        targets = {
            key: batch[key] 
            for key in self.keys
        }
        
        # Forward pass
        outputs = self(batch)
        predictions = {key: output[0] for key, output in outputs.items()}
        batch_size = batch['records'].size(0)

        for key in self.keys:
            pres = predictions[key]
            tars = targets[key]
            self.metrics[f'{key}_spearman'].update(pres, tars) # type: ignore
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Log overall metrics
        for metric_name, metric in self.metrics.items():
            value = metric.compute() # type: ignore
            self.log(f'test/{metric_name}', value)
            logger.info(f"{metric_name}: {value:.4f}")
            metric.reset() # type: ignore
    
    def configure_optimizers(self): # type: ignore
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = hydra.utils.instantiate(self.args.optimizer_para, params=self.model.parameters()) # type: ignore
        t_max = getattr(self.trainer, "max_epochs", 400)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # 每个 epoch 调整一次
                "frequency": 1,
                "name": "cosine_anneal"
            }
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # grad_max = 0
        # grad_min = 1.0
        for name, p in self.model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                logger.info(
                    f"Gradients were nan for {name}, and skip_nan_grad_updates was enabled."
                    " Zeroing grad for this batch."
                )
                self.optimizer_zero_grad(epoch, batch_idx, optimizer) # type: ignore
                break
        # logger.info(f"Max grad: {grad_max}, min grad: {grad_min}")
        optimizer.step(closure=optimizer_closure)
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update_parameters(self.model) # type: ignore
