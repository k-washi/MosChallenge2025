import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoTokenizer  # pyright: ignore[reportPrivateImportUsage]

from src.audio.utils import load_wave
from track1.core.config import Config

SQUEEZE_DIM = 2
CPU_DEVICE = torch.device("cpu")


class MOSDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset class for loading."""

    def __init__(
        self,
        config: Config,
        audio_list: list[str | Path],
        prompt_list: list[str],
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
        self.prompt_list = prompt_list

        self.feat_dir = Path(config.data.featdir)

        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        prompt = self.prompt_list[idx]
        score = self.score_list[idx]
        wav_name = Path(wav_path).stem

        audio, sr = load_wave(
            str(wav_path),
            sample_rate=self.config.data.sample_rate,
            mono=True,
            is_torch=True,
        )
        assert isinstance(audio, torch.Tensor), f"Audio file {wav_path} is not a torch tensor."

        prompt = self.tok(prompt, return_tensors="pt", padding=True)

        if self.is_aug:
            audio = self.aug(audio, sr)

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

        # 最大長さを調整
        audio_length1 = audio.shape[-1]
        audio_time_length = audio_length1 / sr
        if audio_time_length > self.config.data.max_duration:
            random_start = torch.randint(0, audio_length1 - math.ceil(self.config.data.max_duration * sr), (1,)).item()
            audio = audio[random_start : random_start + int(self.config.data.max_duration * sr)]
        audio = (audio - audio.mean()) / torch.sqrt(torch.var(audio) + 1e-7)

        return audio_emb, prompt_emb, aux_emb, score_mi, score_ta, str(wav_path), audio, prompt

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

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """Collate function to combine a list of samples into a batch."""
        audio_emb, prompt_emb, aux_emb, score_mi, score_ta, wav_str, audio, prompt_bert = zip(*batch, strict=False)
        audio_emb = torch.stack(audio_emb, dim=0)
        prompt_emb = torch.stack(prompt_emb, dim=0)
        aux_emb = torch.stack(aux_emb, dim=0)
        score_mi = torch.stack(score_mi, dim=0)
        score_ta = torch.stack(score_ta, dim=0)

        # audio process
        audio_list = list(audio)
        max_len = max([a.shape[-1] for a in audio_list])
        audio_tensor = torch.zeros((len(audio_list), max_len), dtype=torch.float32)
        audio_attention_mask = torch.zeros((len(audio_list), max_len), dtype=torch.float32)
        for i, audio in enumerate(audio_list):
            audio_len = audio.shape[-1]
            audio_tensor[i, :audio_len] = audio
            audio_attention_mask[i, :audio_len] = 1.0

        # prompt process
        prompt_max_len = max([p["attention_mask"].shape[-1] for p in prompt_bert])
        prompt_tensor = torch.zeros((len(prompt_bert), prompt_max_len), dtype=torch.long)
        prompt_attention_mask = torch.zeros((len(prompt_bert), prompt_max_len), dtype=torch.long)
        for i, p in enumerate(prompt_bert):
            prompt_len = p["attention_mask"].shape[-1]
            prompt_tensor[i, :prompt_len] = p["input_ids"]
            prompt_attention_mask[i, :prompt_len] = p["attention_mask"]
        return (
            audio_emb,
            prompt_emb,
            aux_emb,
            score_mi,
            score_ta,
            wav_str,
            audio_tensor,
            audio_attention_mask,
            prompt_tensor,
            prompt_attention_mask,
        )


if __name__ == "__main__":
    # Example usage
    config = Config()
    test_fp = "data/MusicEval-phase1/wav/audiomos2025-track1-S001_P001.wav"
    audio_list: list[str | Path] = [test_fp, test_fp]
    prompt_list = ["prompt1", "prompt2 hello world"]
    score_list = [(3.5, 4.0), (2.5, 3.0)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mos_dataset = MOSDataset(config, audio_list, prompt_list, score_list, is_aug=True)
    dataloader = torch.utils.data.DataLoader(
        mos_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=mos_dataset.collate_fn
    )
    for batch in dataloader:
        audio_emb, prompt_emb, aux_emb, score_mi, score_ta, wav_str, audio, audio_m, prompt_bert_i, prompt_bert_m = batch
        print(f"Audio Embedding Shape: {audio_emb.shape}")
        print(f"Prompt Embedding Shape: {prompt_emb.shape}")
        print(f"Auxiliary Features Shape: {aux_emb.shape}")
        print(f"Score MI: {score_mi}")
        print(f"Score TA: {score_ta}")
        print(f"Audio File Path: {wav_str}")
        print(f"Audio Shape: {audio.shape}")
        print(f"Audio Mask Shape: {audio_m.shape}")
        print(f"Prompt Input IDs Shape: {prompt_bert_i.shape}")
        print(f"Prompt Attention Mask Shape: {prompt_bert_m.shape}")
        break
