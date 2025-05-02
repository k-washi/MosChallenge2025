from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from track1.core.config import Config
from track1.core.dataset.dataset import MOSDataset

EMPTY_LIST = []
CPU_DEVICE = torch.device("cpu")


class MOSDataModule(LightningDataModule):
    """Data module for the MOS dataset.

    Args:
    ----
        config (Config): Configuration object containing dataset parameters.
        train_csv (str): Path to the training CSV file.
        val_csv (str): Path to the validation CSV file.
        test_csv (str): Path to the test CSV file.
        wav_dir (str): Path to the directory containing WAV files.

    """

    def __init__(
        self,
        config: Config,
        train_audio_list: list[str | Path],
        train_prompt_list: list[str],
        train_score_list: list[tuple[float, float]],
        val_audio_list: list[str | Path],
        val_prompt_list: list[str],
        val_score_list: list[tuple[float, float]],
        test_audio_list: list[str | Path] = EMPTY_LIST,
        test_prompt_list: list[str] = EMPTY_LIST,
        test_score_list: list[tuple[float, float]] = EMPTY_LIST,
        device: torch.device = CPU_DEVICE,
    ) -> None:
        """Initialize the MOSDataModule."""
        super().__init__()
        self.c = config
        self.train_audio_list = train_audio_list
        self.train_prompt_list = train_prompt_list
        self.train_score_list = train_score_list
        self.val_audio_list = val_audio_list
        self.val_prompt_list = val_prompt_list
        self.val_score_list = val_score_list
        self.test_audio_list = test_audio_list
        self.test_prompt_list = test_prompt_list
        self.test_score_list = test_score_list

        self.device = device
        self.setup()

    def setup(self, stage: str | None = None) -> None:
        """Set the dataset for training, validation, and testing.

        Args:
        ----
            stage (str, optional): Stage of the data module. Can be 'fit', 'validate', or 'test'.

        """
        if stage == "fit" or stage is None:
            self.train_dataset = MOSDataset(
                config=self.c,
                audio_list=self.train_audio_list,
                prompt_list=self.train_prompt_list,
                score_list=self.train_score_list,
                is_aug=True,
                device=self.device,
            )
            self.val_dataset = MOSDataset(
                config=self.c,
                audio_list=self.val_audio_list,
                prompt_list=self.val_prompt_list,
                score_list=self.val_score_list,
                is_aug=False,
                device=self.device,
            )
        if stage == "test" or stage is None:
            self.test_dataset = MOSDataset(
                config=self.c,
                audio_list=self.test_audio_list,
                prompt_list=self.test_prompt_list,
                score_list=self.test_score_list,
                is_aug=False,
                device=self.device,
            )

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader.

        Returns
        -------
            DataLoader: DataLoader for the training dataset.

        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.c.ml.batch_size,
            shuffle=True,
            num_workers=self.c.ml.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader.

        Returns
        -------
            DataLoader: DataLoader for the validation dataset.

        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.c.ml.test_batch_size,
            shuffle=False,
            num_workers=self.c.ml.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader.

        Returns
        -------
            DataLoader: DataLoader for the test dataset.

        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.c.ml.test_batch_size,
            shuffle=False,
            num_workers=self.c.ml.num_workers,
            pin_memory=True,
        )
