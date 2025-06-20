import random
from pathlib import Path

import torch
import torch.nn.functional as F

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
    ) -> None:
        """Mos dataset class.

        Args:
        ----
            config (Config): Configuration object containing dataset parameters.
            audio_list (list[str|Path]): List of audio file paths.
            prompt_list (list[str]): List of text prompts corresponding to the audio files.
            score_list (list[tuple[float, float]]): List of tuples containing scores for the audio files.
            is_aug (bool): Flag indicating whether to apply data augmentation. Default is False.

        """
        super().__init__()
        self.config = config
        self.audio_list = audio_list
        self.score_list = score_list

        self.feat_dir = Path(config.data.featdir)

        assert len(self.audio_list) == len(
            self.score_list
        ), f"audio_list and score_list must have the same length. {len(self.audio_list)} != {len(self.score_list)}"

        self.is_aug = is_aug

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

        wav_name = Path(wav_path).stem

        if self.is_aug:
            # audio embedding
            target_fp = self.feat_dir / self.config.data.audio_feat / f"{wav_name}.pt"
            if self.config.data.aug_rate < torch.rand(1).item():
                # a 0.8 < randam valueでデータ拡張は無視
                audio_emb = torch.load(str(target_fp), map_location="cpu")
            else:
                aug_audio_dir = self.feat_dir / self.config.data.audio_feat_aug / wav_name
                embedding_list = sorted(list(aug_audio_dir.glob("*.pt")))
                embedding_list.append(target_fp)
                embedding_fp = random.choice(embedding_list)
                audio_emb = torch.load(embedding_fp, map_location="cpu")

            # prompt embedding
            aug_prompt_dir = self.feat_dir / self.config.data.prompt_feat_aug / wav_name
            embedding_list = sorted(list(aug_prompt_dir.glob("*.pt")))
            target_fp = self.feat_dir / self.config.data.prompt_feat / f"{wav_name}.pt"
            if len(embedding_list) == 0 or self.config.data.aug_rate < torch.rand(1).item():
                # a 0.8 < randam valueでデータ拡張は無視
                prompt_emb = torch.load(str(target_fp), map_location="cpu")
            else:
                aug_prompt_dir = self.feat_dir / self.config.data.prompt_feat_aug / wav_name
                embedding_list = sorted(list(aug_prompt_dir.glob("*.pt")))
                embedding_list.append(target_fp)
                embedding_fp = random.choice(embedding_list)
                prompt_emb = torch.load(embedding_fp, map_location="cpu")
            # aux embedding
            target_fp = self.feat_dir / self.config.data.aux_feat / f"{wav_name}.pt"
            if self.config.data.aug_rate < torch.rand(1).item():
                # a 0.8 < randam valueでデータ拡張は無視
                aux_emb = torch.load(str(target_fp), map_location="cpu")
            else:
                aug_aux_dir = self.feat_dir / self.config.data.aux_feat_aug / wav_name
                embedding_list = sorted(list(aug_aux_dir.glob("*.pt")))
                embedding_list.append(target_fp)
                embedding_fp = random.choice(embedding_list)
                aux_emb = torch.load(embedding_fp, map_location="cpu")

        else:
            audio_emb = torch.load(str(self.feat_dir / self.config.data.audio_feat / f"{wav_name}.pt"), map_location="cpu")
            prompt_emb = torch.load(str(self.feat_dir / self.config.data.prompt_feat / f"{wav_name}.pt"), map_location="cpu")
            aux_emb = torch.load(str(self.feat_dir / self.config.data.aux_feat / f"{wav_name}.pt"), map_location="cpu")
        score_mi, score_ta = score
        score_mi = (score_mi - (self.config.data.label_min + self.config.data.label_max) / 2.0) / (
            self.config.data.label_norm_max - self.config.data.label_norm_min
        )
        score_ta = (score_ta - (self.config.data.label_min + self.config.data.label_max) / 2.0) / (
            self.config.data.label_norm_max - self.config.data.label_norm_min
        )
        score_mi = torch.tensor(score_mi, dtype=torch.float32)
        score_ta = torch.tensor(score_ta, dtype=torch.float32)

        audio_emb = F.normalize(audio_emb, p=2, dim=-1)
        prompt_emb = F.normalize(prompt_emb, p=2, dim=-1)

        return audio_emb, prompt_emb, aux_emb, score_mi, score_ta, str(wav_path)


if __name__ == "__main__":
    # Example usage
    config = Config()
    test_fp = "data/MusicEval-phase1/wav/audiomos2025-track1-S001_P001.wav"
    audio_list: list[str | Path] = [test_fp, test_fp]
    score_list = [(3.5, 4.0), (2.5, 3.0)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mos_dataset = MOSDataset(config, audio_list, score_list, is_aug=True)
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
