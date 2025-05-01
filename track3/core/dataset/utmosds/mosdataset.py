# audio file, mos, user_idを返す

import math
from pathlib import Path

import torch
import torchaudio

from src.audio import load_wave
from track3.core.config import Config

SQUEEZE_DIM = 2

DATA_ID = 0
CL_ID = 1


class MOSDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset class for loading and processing MOS (Mean Opinion Score) dataset."""

    def __init__(
        self,
        config: Config,
        dataset_list: list[tuple[str | Path, float, str]],
        is_transform: bool = False,
        is_train: bool = False,
    ) -> None:
        """Set up the dataset.

        Args:
        ----
            config (Config): Configuration object containing dataset parameters.
            dataset_list (list[tuple[str | Path, float, str]]): List of dataset file paths, scores, and user IDs.
            contrastive_dataset_list (list[tuple[tuple[str | Path, float], tuple[str | Path, float]]])
                :List of contrastive dataset file paths and scores.
            is_transform (bool): Whether to apply transformations to the audio data.
            is_train (bool): Whether the dataset is for training or not.

        """
        super().__init__()

        self.dataset_list = dataset_list
        self.c = config
        self.is_transform = is_transform
        self.is_train = is_train

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataset_list)

    def __getitem__(self, idx: int) -> tuple:
        """Get items.

        Args:
        ----
            idx (int): Index of the sample to fetch.

        Returns:
        -------
            tuple: A tuple containing (
                audio1, mos_score1, str(audio_file1),
                audio2, mos_score2, str(audio_file2)
            ).

        """
        audio_file1, mos_score1, dataset_name = self.dataset_list[idx]
        if self.is_train:
            dataset_id = self.c.data.train_dataset_dict.get(dataset_name, -1)
        else:
            dataset_id = self.c.data.test_dataset_dict.get(dataset_name, -1)
        assert dataset_id >= 0, f"Dataset {dataset_name} not found in dataset_dict."

        audio1, sr = load_wave(str(audio_file1), sample_rate=self.c.data.sample_rate, mono=True, is_torch=True)
        assert isinstance(audio1, torch.Tensor), f"Audio file {audio_file1} is not a torch tensor."
        if self.is_transform:
            audio1 = self.aug(audio1, sr)

        # 最大長さを調整
        audio_length1 = audio1.shape[-1]
        audio_time_length1 = audio_length1 / sr
        if audio_time_length1 > self.c.data.max_duration:
            random_start = torch.randint(0, audio_length1 - math.ceil(self.c.data.max_duration * sr), (1,)).item()
            audio1 = audio1[random_start : random_start + int(self.c.data.max_duration * sr)]
        mos_score1 = (mos_score1 - (self.c.data.label_min + self.c.data.label_max) / 2.0) / (
            self.c.data.label_norm_max - self.c.data.label_norm_min
        )

        # normalize
        # a audio1 = audio1 / (torch.max(torch.abs(audio1)).item() * self.c.data.normalize_scale)
        audio1 = (audio1 - audio1.mean()) / torch.sqrt(torch.var(audio1) + 1e-7)
        return audio1, mos_score1, str(audio_file1), dataset_id

    def aug(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply augmentation to the audio data.

        Args:
        ----
            x (torch.Tensor): Input audio tensor.
            sr (int): Sample rate of the audio.

        Returns:
        -------
            torch.Tensor: Augmented audio tensor.

        """
        if self.c.data.aug_rate < torch.rand(1).item():
            # a 0.8 < randam valueでデータ拡張は無視
            return x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pitch_shift = int(torch.randint(-self.c.data.pitch_shift_max, self.c.data.pitch_shift_max + 1, (1,)).item())
        time_wrap = float(torch.empty(1).uniform_(self.c.data.time_wrap_min, self.c.data.time_wrap_max).item())
        effects = [
            ["tempo", str(time_wrap)],
            ["pitch", str(pitch_shift)],
        ]
        x, _ = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
        if x.dim() == SQUEEZE_DIM:
            x = x.squeeze(0)
        return x

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """Collate function to combine a list of samples into a batch.

        Args:
        ----
            batch (list): List of tuples containing (audio, mos_score, user_id).

        Returns:
        -------
            tuple: A tuple containing the batch of audio, mos_score and user_id.

        """
        wavs1, scores1, audio_files1, dataset_ids = zip(*batch, strict=False)
        wavs1 = list(wavs1)

        max_len = max([wav.shape[-1] for wav in wavs1])

        wave1_tensor = torch.zeros((len(wavs1), max_len), dtype=torch.float32)
        attention_mask1 = torch.zeros((len(wavs1), max_len), dtype=torch.long)
        score_tensor1 = torch.tensor(scores1, dtype=torch.float32)
        dataset_ids = torch.tensor(dataset_ids, dtype=torch.long)
        for i, wav in enumerate(wavs1):
            wave1_tensor[i, : wav.shape[-1]] = wav
            attention_mask1[i, : wav.shape[-1]] = 1

        return (wave1_tensor, attention_mask1, score_tensor1, audio_files1, dataset_ids)


if __name__ == "__main__":
    """data loader test
    wav, mos, user_id, audio_file
    --------------------
    torch.Size([2, 75360])
    tensor([1, 2])
    tensor([0, 1])
    ('src/examples/sample.wav', 'src/examples/sample.wav')
    --------------------
    torch.Size([1, 75360])
    tensor([3])
    tensor([2])
    ('src/examples/sample.wav',)
    --------------------
    """
