# audio file, mos, user_idを返す

import math
from pathlib import Path

import torch

from src.audio import load_wave
from track3.core.config import Config


class MOSDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset class for loading and processing MOS (Mean Opinion Score) dataset."""

    def __init__(
        self,
        config: Config,
        audio_list: list[str | Path],
        label_list: list[int],
        user_id_list: list[int],
        is_transform: bool = False,
    ) -> None:
        """Set up the dataset.

        Args:
        ----
            config (Config): Configuration object containing dataset parameters.
            audio_list (list[str | Path]): List of audio file paths.
            label_list (list[int]): List of MOS scores.
            user_id_list (list[int]): List of user IDs.
            is_transform (bool): Whether to apply transformations to the audio data.

        """
        self.audio_list = audio_list
        self.label_list = label_list
        self.user_id_list = user_id_list

        assert (
            len(audio_list) == len(label_list) == len(user_id_list)
        ), "Length of audio_list, label_list and user_id_list must be the same."

        self.c = config
        self.is_transform = is_transform

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.audio_list)

    def __getitem__(self, idx: int) -> tuple:
        """Get items.

        Args:
        ----
            idx (int): Index of the sample to fetch.

        Returns:
        -------
            tuple: A tuple containing (audio, mos_score, user_id, audio_file).

        """
        audio_file = self.audio_list[idx]
        mos_score = self.label_list[idx]
        user_id = self.user_id_list[idx]

        # Load the audio file
        audio, sr = load_wave(str(audio_file), sample_rate=self.c.data.sample_rate, mono=True, is_torch=True)
        assert isinstance(audio, torch.Tensor), f"Audio file {audio_file} is not a torch tensor."

        # normalize
        audio = audio / (torch.max(torch.abs(audio)).item() * self.c.data.normalize_scale)

        # 最大長さを調整
        audio_length = audio.shape[-1]
        audio_time_length = audio_length / sr
        if audio_time_length > self.c.data.max_duration:
            random_start = torch.randint(0, audio_length - math.ceil(self.c.data.max_duration * sr), (1,)).item()
            audio = audio[random_start : random_start + int(self.c.data.max_duration * sr)]

        return audio, mos_score, user_id, str(audio_file)

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
        wavs, scores, user_ids, audio_files = zip(*batch, strict=False)
        wavs = list(wavs)
        max_len = max([wav.shape[-1] for wav in wavs])

        wave_tensor = torch.zeros((len(wavs), max_len))
        attention_mask = torch.zeros((len(wavs), max_len))
        score_tensor = torch.tensor(scores, dtype=torch.long)
        user_tensor = torch.tensor(user_ids, dtype=torch.long)
        for i, wav in enumerate(wavs):
            wave_tensor[i, : wav.shape[-1]] = wav
            attention_mask[i, : wav.shape[-1]] = 1
        return wave_tensor, attention_mask, score_tensor, user_tensor, audio_files


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
    import torch

    from track3.core.config import Config
    from track3.core.dataset.mosdataset import MOSDataset

    f = "src/examples/sample.wav"
    config = Config()

    dataset = MOSDataset(config, [f, f, f], [1, 2, 3], [0, 1, 2])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=MOSDataset.collate_fn)

    for wav, mos, user_id, audio_file in data_loader:
        print(wav.shape)
        print(mos)
        print(user_id)
        print(audio_file)
        print("-" * 20)
