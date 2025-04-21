from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from track3.core.config import Config
from track3.core.dataset.mosdataset import MOSDataset

EMPTY_LIST = []


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
        train_label_list: list[int],
        val_audio_list: list[str | Path],
        val_label_list: list[int],
        test_audio_list: list[str | Path],
        test_label_list: list[int],
        train_user_id_list: list[int] = EMPTY_LIST,
        val_user_id_list: list[int] = EMPTY_LIST,
        test_user_id_list: list[int] = EMPTY_LIST,
    ) -> None:
        """Initialize the MOSDataModule."""
        super().__init__()
        self.c = config
        self.train_audio_list = train_audio_list
        self.train_label_list = train_label_list
        self.val_audio_list = val_audio_list
        self.val_label_list = val_label_list
        self.test_audio_list = test_audio_list
        self.test_label_list = test_label_list
        self.train_user_id_list = train_user_id_list if len(train_user_id_list) > 0 else [0] * len(train_audio_list)
        self.val_user_id_list = val_user_id_list if len(val_user_id_list) > 0 else [0] * len(val_audio_list)
        self.test_user_id_list = test_user_id_list if len(test_user_id_list) > 0 else [0] * len(test_audio_list)

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
                label_list=self.train_label_list,
                user_id_list=self.train_user_id_list,
                is_transform=True,
            )
            self.val_dataset = MOSDataset(
                config=self.c,
                audio_list=self.val_audio_list,
                label_list=self.val_label_list,
                user_id_list=self.val_user_id_list,
                is_transform=False,
            )
        if stage == "test" or stage is None:
            self.test_dataset = MOSDataset(
                config=self.c,
                audio_list=self.test_audio_list,
                label_list=self.test_label_list,
                user_id_list=self.test_user_id_list,
                is_transform=False,
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
            collate_fn=self.train_dataset.collate_fn,
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
            collate_fn=self.val_dataset.collate_fn,
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
            collate_fn=self.test_dataset.collate_fn,
        )
