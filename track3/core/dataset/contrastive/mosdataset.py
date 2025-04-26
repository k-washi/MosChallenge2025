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
        contrastive_dataset_list: list[tuple[tuple[str | Path, float], tuple[str | Path, float]]],
        is_transform: bool = False,
        is_dataextend: bool = False,
    ) -> None:
        """Set up the dataset.

        Args:
        ----
            config (Config): Configuration object containing dataset parameters.
            dataset_list (list[tuple[str | Path, float, str]]): List of dataset file paths, scores, and user IDs.
            contrastive_dataset_list (list[tuple[tuple[str | Path, float], tuple[str | Path, float]]])
                :List of contrastive dataset file paths and scores.
            is_transform (bool): Whether to apply transformations to the audio data.
            is_dataextend (bool): Whether to extend the dataset.

        """
        super().__init__()

        print(f"dataset_list: {len(dataset_list)}, contrastive_dataset_list: {len(contrastive_dataset_list)}")
        self.dataset_map = {}
        for audio_file, score, dataset_name in dataset_list:
            if dataset_name not in self.dataset_map:
                self.dataset_map[dataset_name] = []
            self.dataset_map[dataset_name].append((audio_file, score))

        # データセット数をcontrastiveデータに合わせる
        if is_dataextend:
            dataset_length = len(dataset_list)
            contrastive_dataset_length = len(contrastive_dataset_list)
            if dataset_length < contrastive_dataset_length:
                new_dataset_list = []
                multiple = math.ceil(contrastive_dataset_length * config.data.extend_rate / dataset_length)
                for _ in range(multiple):
                    new_dataset_list.extend(dataset_list)
                dataset_list = new_dataset_list
        self.dataset_list = [*dataset_list, *contrastive_dataset_list]
        self.dataset_id_list = [*[DATA_ID] * len(dataset_list), *[CL_ID] * len(contrastive_dataset_list)]
        print(f"update dataset_list: {len(dataset_list)}, contrastive_dataset_list: {len(contrastive_dataset_list)}")
        self.c = config
        self.is_transform = is_transform

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataset_id_list)

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
        dataset_id = self.dataset_id_list[idx]
        if dataset_id == DATA_ID:
            audio_file1, mos_score1, dataset_name = self.dataset_list[idx]
            rand_idx = torch.randint(low=0, high=len(self.dataset_map[dataset_name]), size=(1,)).item()
            audio_file2, mos_score2 = self.dataset_map[dataset_name][rand_idx]
            if mos_score2 > mos_score1:
                audio_file1, audio_file2 = audio_file2, audio_file1
                mos_score2, mos_score1 = mos_score1, mos_score2
        elif dataset_id == CL_ID:
            (audio_file1, mos_score1), (audio_file2, mos_score2) = self.dataset_list[idx]
            dataset_name = "contrastive"
        else:
            emsg = f"Invalid dataset_id: {dataset_id}"
            raise ValueError(emsg)
        # Load the audio file
        assert Path(audio_file1).exists(), f"Audio file {audio_file1} does not exist."
        assert Path(audio_file2).exists(), f"Audio file {audio_file2} does not exist."

        audio1, sr = load_wave(str(audio_file1), sample_rate=self.c.data.sample_rate, mono=True, is_torch=True)
        assert isinstance(audio1, torch.Tensor), f"Audio file {audio_file1} is not a torch tensor."
        audio2, sr = load_wave(str(audio_file2), sample_rate=self.c.data.sample_rate, mono=True, is_torch=True)
        assert isinstance(audio2, torch.Tensor), f"Audio file {audio_file2} is not a torch tensor."

        # normalize
        audio1 = audio1 / (torch.max(torch.abs(audio1)).item() * self.c.data.normalize_scale)
        audio2 = audio2 / (torch.max(torch.abs(audio2)).item() * self.c.data.normalize_scale)

        if self.is_transform:
            audio1 = self.aug(audio1, sr)
            audio2 = self.aug(audio2, sr)

        # 最大長さを調整
        audio_length1 = audio1.shape[-1]
        audio_time_length1 = audio_length1 / sr
        if audio_time_length1 > self.c.data.max_duration:
            random_start = torch.randint(0, audio_length1 - math.ceil(self.c.data.max_duration * sr), (1,)).item()
            audio1 = audio1[random_start : random_start + int(self.c.data.max_duration * sr)]
        audio_length2 = audio2.shape[-1]
        audio_time_length2 = audio_length2 / sr
        if audio_time_length2 > self.c.data.max_duration:
            random_start = torch.randint(0, audio_length2 - math.ceil(self.c.data.max_duration * sr), (1,)).item()
            audio2 = audio2[random_start : random_start + int(self.c.data.max_duration * sr)]

        mos_score1 = (mos_score1 - (self.c.data.label_min + self.c.data.label_max) / 2.0) / (
            self.c.data.label_norm_max - self.c.data.label_norm_min
        )
        mos_score2 = (mos_score2 - (self.c.data.label_min + self.c.data.label_max) / 2.0) / (
            self.c.data.label_norm_max - self.c.data.label_norm_min
        )
        return audio1, mos_score1, str(audio_file1), audio2, mos_score2, str(audio_file2)

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
        wavs1, scores1, audio_files1, wavs2, scores2, audio_files2 = zip(*batch, strict=False)
        wavs1 = list(wavs1)
        wavs2 = list(wavs2)

        max_len1 = max([wav.shape[-1] for wav in wavs1])
        max_len2 = max([wav.shape[-1] for wav in wavs2])
        max_len = max(max_len1, max_len2)

        wave1_tensor = torch.zeros((len(wavs1), max_len))
        attention_mask1 = torch.zeros((len(wavs1), max_len))
        score_tensor1 = torch.tensor(scores1, dtype=torch.float32)

        for i, wav in enumerate(wavs1):
            wave1_tensor[i, : wav.shape[-1]] = wav
            attention_mask1[i, : wav.shape[-1]] = 1

        wave2_tensor = torch.zeros((len(wavs2), max_len))
        attention_mask2 = torch.zeros((len(wavs2), max_len))
        score_tensor2 = torch.tensor(scores2, dtype=torch.float32)
        for i, wav in enumerate(wavs2):
            wave2_tensor[i, : wav.shape[-1]] = wav
            attention_mask2[i, : wav.shape[-1]] = 1

        return (
            wave1_tensor,
            attention_mask1,
            score_tensor1,
            audio_files1,
            wave2_tensor,
            attention_mask2,
            score_tensor2,
            audio_files2,
        )


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

    f = "src/examples/sample.wav"
    config = Config()

    dataset = MOSDataset(config, [(f, 1, "dd"), (f, 2, "dd"), (f, 3, "dd")], [((f, 1), (f, 2))], is_transform=True)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=MOSDataset.collate_fn)

    for wav1, att_mask1, mos1, audio_file1, wav2, att_mask2, mos2, audio_file2 in data_loader:
        print(wav1.shape, wav2.shape)
        print(att_mask1.shape, att_mask2.shape)
        print(mos1)
        print(mos2)
        print(audio_file1)
        print(audio_file2)
        print("-" * 20)
