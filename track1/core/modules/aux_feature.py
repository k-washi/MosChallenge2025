import librosa
import numpy as np
import pyloudnorm as pyln
import torch

from src.audio import load_wave
from track1.core.modules.cnn14.models import Cnn14_16k

# ── PANNs (CNN14) モデル ───────────────────────────────────────────
CNN14_SR = 16000
CNN14_WS = 512
CNN14_HOP_SIZE = 160
CNN14_MELBINS = 64
CNN14_FMIN = 50
CNN14_FMAX = 8000
CNN14_CLASS_NUM = 527


class AudioTagCNN1C4(torch.nn.Module):
    """Extract auxiliary features from audio using CLAP and PANNs (CNN14)."""

    def __init__(self, ckpt_fp: str = "pretrained/Cnn14_mAP=0.431.pth") -> None:
        """Initialize the CNN14 model."""
        super().__init__()
        ckpt = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
        # モデル定義を省略（github.com/qiuqiangkong/audioset_tagging_cnn）
        self.model = Cnn14_16k(
            sample_rate=CNN14_SR,
            window_size=CNN14_WS,
            hop_size=CNN14_HOP_SIZE,
            mel_bins=CNN14_MELBINS,
            fmin=CNN14_FMIN,
            fmax=CNN14_FMAX,
            classes_num=CNN14_CLASS_NUM,
        )  # Skeleton
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.inference_mode()
    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        with torch.no_grad():
            return self.model(wav_16k)["clipwise_output"]  # (B, 527)


TARGET_SR = 16000


class ExtractAuxFeatures:
    """extract auxiliary features from audio using CLAP and PANNs (CNN14)."""

    def __init__(
        self,
        cnn14_ckpt: str = "pretrained/Cnn14_16k_mAP=0.438.pth",
    ) -> None:
        """Initialize the feature extractor."""
        self.tag_model = AudioTagCNN1C4(ckpt_fp=cnn14_ckpt)

    def extract_aux_features(
        self,
        audio_fp: str,
    ) -> torch.Tensor:
        """Extract auxiliary features from audio.

        557-dim 補助特徴を返す。
        torch.Size([557])
        """
        # 1) ロード & Mono 化
        wav_pt, _ = load_wave(audio_fp, sample_rate=TARGET_SR, mono=True, is_torch=True)
        assert isinstance(wav_pt, torch.Tensor), f"Audio file {audio_fp} is not a torch tensor."
        wav_pt = wav_pt / (wav_pt.abs().max() * 1.0001)  # Normalize

        wav_np = wav_pt.numpy().astype(np.float32)
        # 2) Loudness / RMS / Dynamic Range
        meter = pyln.Meter(TARGET_SR)  # EBU-R128
        loudness = meter.integrated_loudness(wav_np)
        rms = np.sqrt(np.mean(wav_np**2))
        try:
            dr = np.percentile(np.abs(wav_np), 95) / np.percentile(np.abs(wav_np), 5)  # Dynamic-Range proxy
        except ZeroDivisionError:
            dr = 0.0
        # 3) 時間周波数特徴（統計量は μ, σ の 2-値）
        s = np.abs(librosa.stft(wav_np, n_fft=2048, hop_length=1024))
        mel = librosa.feature.melspectrogram(S=s, sr=TARGET_SR, n_mels=96, fmax=TARGET_SR // 2)
        centroid = librosa.feature.spectral_centroid(S=s, sr=TARGET_SR)
        bandwidth = librosa.feature.spectral_bandwidth(S=s, sr=TARGET_SR)
        contrast = librosa.feature.spectral_contrast(S=s, sr=TARGET_SR)
        rolloff = librosa.feature.spectral_rolloff(S=s, sr=TARGET_SR)
        zcr = librosa.feature.zero_crossing_rate(wav_np)

        def stats(x: np.ndarray, norm_value: int | None = None) -> list[float]:  # → (2,)
            if norm_value is not None:
                return [float(np.mean(x)) / norm_value, float(np.std(x)) / norm_value]
            return [float(np.mean(x)), float(np.std(x))]

        spec_stats = (
            stats(centroid, norm_value=TARGET_SR // 2)
            + stats(bandwidth, norm_value=TARGET_SR // 2)
            + stats(contrast)
            + stats(rolloff, norm_value=TARGET_SR // 2)
            + stats(zcr)
            + stats(mel)[:2]
        )  # 12 dims

        # 4) ビート・テンポ
        tempo, _ = librosa.beat.beat_track(y=wav_np, sr=TARGET_SR)  # BPM
        if isinstance(tempo, list | np.ndarray):
            tempo = tempo[0]
        onset_env = librosa.onset.onset_strength(y=wav_np, sr=TARGET_SR)
        onset_den = float(np.mean(onset_env > np.percentile(onset_env, 90)))

        # 5) 調性（Key） ― chroma STFT から最大エネルギ基音
        chroma = librosa.feature.chroma_stft(y=wav_np, sr=TARGET_SR)
        key_pc = int(np.argmax(np.mean(chroma, axis=1)))  # 0-11
        key_vec = np.eye(12)[key_pc]  # one-hot 12

        # 6) PANNs タグ確率
        with torch.no_grad():
            tag_logits = self.tag_model(wav_pt.unsqueeze(0))  # (1, 527)
        tag_probs = torch.sigmoid(tag_logits).squeeze(0).numpy()

        # 7) メタ
        duration = len(wav_np) / TARGET_SR

        # 正規化
        loudness = float(loudness) / 100.0  # max 100db
        dr = min(1, float(dr) / 10000000)
        tempo = tempo / 360

        # 8) 結合
        parts = [
            np.array([loudness, rms, dr], dtype=np.float32),
            np.array(spec_stats, dtype=np.float32),
            np.array([tempo, onset_den], dtype=np.float32),
            key_vec,
            tag_probs,
            np.array([duration], dtype=np.float32),
        ]
        feats = np.hstack(parts)

        # 動的な次元チェック
        expected_dim = 3 + len(spec_stats) + 2 + key_vec.shape[0] + tag_probs.shape[0] + 1
        assert feats.ndim == 1, f"Unexpected feature dim: {feats.shape[0]} != {expected_dim}"
        assert feats.shape[0] == expected_dim, f"Unexpected feature dim: {feats.shape[0]} != {expected_dim}"
        return torch.tensor(feats, dtype=torch.float32)


if __name__ == "__main__":
    # Example usage
    model = ExtractAuxFeatures()
    x = model.extract_aux_features(
        "data/MusicEval-phase1/wav48k_audiosr/28_04_2025_14_49_29/audiomos2025-track1-S002_P099_AudioSR_Processed_48K.wav"
    )
    print(x.shape)  # Should print the shape of the extracted features
    print(x)
