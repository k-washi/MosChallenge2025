from pathlib import Path

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from track3.core.config import Config
from track3.core.dataset.utmosds.mosdataset import MOSDataset

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
        train_dataset_list: list[tuple[str | Path, float, str]],
        val_dataset_list: list[tuple[str | Path, float, str]],
        test_dataset_list: list[tuple[str | Path, float, str]],
    ) -> None:
        """Initialize the MOSDataModule."""
        super().__init__()
        self.c = config
        self.train_dataset_list = train_dataset_list
        self.val_dataset_list = val_dataset_list

        self.test_dataset_list = test_dataset_list

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
                dataset_list=self.train_dataset_list,
                is_transform=True,
                is_train=True,
            )
            self.val_dataset = MOSDataset(
                config=self.c,
                dataset_list=self.val_dataset_list,
                is_transform=False,
            )
        if stage == "test" or stage is None:
            self.test_dataset = MOSDataset(
                config=self.c,
                dataset_list=self.test_dataset_list,
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
