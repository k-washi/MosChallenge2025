from pathlib import Path

import torch
import torchaudio

from track1.core.config import Config

SQUEEZE_DIM = 2
CPU_DEVICE = torch.device("cpu")


class MOSDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset class for loading."""

    def __init__(
        self,
        config: Config,
        audio_list: list[str | Path],
        score_list: list[tuple[float, float]],
        is_aug: bool = False,
        device: torch.device = CPU_DEVICE,
    ) -> None:
        """Mos dataset class.

        Args:
        ----
            config (Config): Configuration object containing dataset parameters.
            audio_list (list[str|Path]): List of audio file paths.
            prompt_list (list[str]): List of text prompts corresponding to the audio files.
            score_list (list[tuple[float, float]]): List of tuples containing scores for the audio files.
            is_aug (bool): Flag indicating whether to apply data augmentation. Default is False.
            device (torch.device): Device to use for computations. Default is CPU.

        """
        super().__init__()
        self.config = config
        self.audio_list = audio_list
        self.score_list = score_list

        assert len(self.audio_list) == len(
            self.score_list
        ), f"audio_list and score_list must have the same length. {len(self.audio_list)} != {len(self.score_list)}"

        self.is_aug = is_aug
        self.device = device

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.audio_list)

    @torch.inference_mode()
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
        # Load the audio file
        wav_path = self.audio_list[idx]
        score = self.score_list[idx]
        _, _ = wav_path, score

        if self.is_aug:
            # Apply data augmentation here if needed
            pass
        audio_emb = torch.zeros(512, dtype=torch.float32)
        prompt_emb = torch.zeros(512, dtype=torch.float32)
        aux_emb = torch.zeros(512, dtype=torch.float32)
        score_mi = torch.tensor(score[0], dtype=torch.float32)
        score_ta = torch.tensor(score[1], dtype=torch.float32)
        return audio_emb, prompt_emb, aux_emb, score_mi, score_ta, str(wav_path)

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
        if self.config.data.aug_rate < torch.rand(1).item():
            # a 0.8 < randam valueでデータ拡張は無視
            return x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pitch_shift = int(torch.randint(-self.config.data.pitch_shift_max, self.config.data.pitch_shift_max + 1, (1,)).item())
        time_wrap = float(torch.empty(1).uniform_(self.config.data.time_wrap_min, self.config.data.time_wrap_max).item())
        effects = [
            ["tempo", str(time_wrap)],
            ["pitch", str(pitch_shift)],
        ]
        x, _ = torchaudio.sox_effects.apply_effects_tensor(x, sr, effects)
        if x.dim() == SQUEEZE_DIM:
            x = x.squeeze(0)
        return x


if __name__ == "__main__":
    # Example usage
    config = Config()
    test_fp = "data/MusicEval-phase1/wav/audiomos2025-track1-S001_P001.wav"
    audio_list: list[str | Path] = [test_fp, test_fp]
    prompt_list = ["prompt1", "prompt2"]
    score_list = [(3.5, 4.0), (2.5, 3.0)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mos_dataset = MOSDataset(config, audio_list, score_list, is_aug=True, device=device)
    dataloader = torch.utils.data.DataLoader(mos_dataset, batch_size=2, shuffle=False, num_workers=0)
    for batch in dataloader:
        audio_emb, prompt_emb, aux_emb, score_mi, score_ta, wav_str = batch
        print(f"Audio Embedding Shape: {audio_emb.shape}")
        print(f"Prompt Embedding Shape: {prompt_emb.shape}")
        print(f"Auxiliary Features Shape: {aux_emb.shape}")
        print(f"Score MI: {score_mi}")
        print(f"Score TA: {score_ta}")
        print(f"Audio File Path: {wav_str}")
        break
