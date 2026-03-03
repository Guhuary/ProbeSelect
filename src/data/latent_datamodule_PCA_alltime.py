import os
from typing import Any, Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
from pathlib import Path
import pickle 
from functools import reduce, partial
from tqdm import trange
import shutil

from lightning import LightningDataModule
from torchvision.transforms import transforms, v2

# from .components.text_embeder import 
from .components.Latent_PCA_alltime import LatentQualityDataset

from ..utils import RankedLogger
from .utils.utils import create_data_splits, custom_collate_fn
logger = RankedLogger(__name__, rank_zero_only=True)

'''
data_dir: str = "data/",
train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
num_files: int = 500,
batch_size: int = 64,
num_workers: int = 0,
pin_memory: bool = False,
start_step: int = 19,
end_step: int = 49,
num_sampling_steps: int = 50,
priors: Dict[str, List] = None, # type: ignore
used_keys: List[str] = None # type: ignore
'''
class LatentQualityDataModule(LightningDataModule):
    def __init__(self, args: Dict[str, Any]
        ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.args = args
        train_ratio, val_ratio, test_ratio = args.train_val_test_split # type: ignore
        self.train_indices, self.val_indices, self.test_indices = create_data_splits(
            num_files=args.num_files, # type: ignore
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random=False,
        )

        # For evaluation-only datasets, it's convenient to set train/val splits to 0.
        # In that case, we still want a valid `test_dataloader()` over all files.
        if train_ratio == 0 and val_ratio == 0:
            self.train_indices = []
            self.val_indices = []
            self.test_indices = list(range(args.num_files))  # type: ignore

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.collate_fn = custom_collate_fn
    
    def prepare_data(self) -> None:
        pass 

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.args.batch_size % self.trainer.world_size != 0: # type: ignore
                raise RuntimeError(
                    f"Batch size ({self.args.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})." # type: ignore
                )
            self.batch_size_per_device = self.args.batch_size // self.trainer.world_size # type: ignore

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Evaluation-only mode: allow train/val to be empty when split ratios are 0.
            self.data_train = (
                LatentQualityDataset(
                    data_dir=self.args.data_dir,
                    stage='train',
                    file_indices=self.train_indices,
                    start=self.args.start_step,
                    end=self.args.end_step,
                    num_sampling_steps=self.args.num_sampling_steps,
                    priors=self.args.priors,
                    keys=self.args.used_keys,
                    known_ts=self.args.known_ts,
                    fix_time=self.args.fix_time,
                )
                if len(self.train_indices) > 0
                else None
            )
            self.data_val = (
                LatentQualityDataset(
                    data_dir=self.args.data_dir,
                    stage='val',
                    file_indices=self.val_indices,
                    start=self.args.start_step,
                    end=self.args.end_step,
                    num_sampling_steps=self.args.num_sampling_steps,
                    priors=self.args.priors,
                    keys=self.args.used_keys,
                    known_ts=self.args.known_ts,
                    fix_time=self.args.fix_time,
                )
                if len(self.val_indices) > 0
                else None
            )
            self.data_test = LatentQualityDataset(
                data_dir=self.args.data_dir,
                stage='test',
                file_indices=self.test_indices,
                start=self.args.start_step,
                end=self.args.end_step,
                num_sampling_steps=self.args.num_sampling_steps,
                priors=self.args.priors,
                keys=self.args.used_keys,
                known_ts=self.args.known_ts,
                fix_time=self.args.fix_time,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train, # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.args.num_workers, # type: ignore
            pin_memory=True, # type: ignore, self.args.pin_memory
            shuffle=True,
            prefetch_factor=4, 
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val, # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.args.num_workers, # type: ignore
            pin_memory=self.args.pin_memory, # type: ignore
            shuffle=False,
            prefetch_factor=4, 
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test, # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.args.num_workers, # type: ignore
            pin_memory=self.args.pin_memory, # type: ignore
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

if __name__ == "__main__":
    _ = LatentQualityDataModule()
