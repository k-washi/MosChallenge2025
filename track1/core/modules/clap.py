import torch
import torchaudio
from parrot import Parrot  # pyright: ignore[reportPrivateImportUsage]
from transformers import (  # pyright: ignore[reportPrivateImportUsage]
    AutoModelForSeq2SeqLM,  # pyright: ignore[reportPrivateImportUsage]
    AutoTokenizer,  # pyright: ignore[reportPrivateImportUsage]
    ClapModel,  # pyright: ignore[reportPrivateImportUsage]
    ClapProcessor,  # pyright: ignore[reportPrivateImportUsage]
    pipeline,  # pyright: ignore[reportPrivateImportUsage]
)

from src.audio import load_wave

CLAP_SR = 48000
CPU_DEVICE = torch.device("cpu")
SQUEEZE_DIM = 2
RANDOM_TH = 0.5


class ClapEmbedding:
    """A class for extracting audio and text features using CLAP (Contrastive Language-Audio Pretraining)."""

    def __init__(self, pretrained_model_name: str = "laion/clap-htsat-fused", device: torch.device = CPU_DEVICE) -> None:
        """Initialize the CLAP model."""
        super().__init__()
        self.device = device
        self.processor = ClapProcessor.from_pretrained(pretrained_model_name)
        self.model = ClapModel.from_pretrained(pretrained_model_name).to(device)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

        self.en2de_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.en2de = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.de2en_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        self.de2en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        self.parrot = Parrot(use_gpu=False)

    def embedding_audio(self, wav_path: str, is_aug: bool = False) -> torch.Tensor:
        """Extract audio features using CLAP.

        Args:
        ----
            wav_path (str): Path to the audio file.
            is_aug (bool): Flag indicating whether to apply data augmentation. Default is False.

        Returns:
        -------
            torch.Tensor: Extracted audio features. [512]

        """
        # Load the audio file
        wav, _ = load_wave(wav_path, sample_rate=CLAP_SR, mono=True, is_torch=True)
        assert isinstance(wav, torch.Tensor), f"Audio file {wav_path} is not a torch tensor."
        if is_aug:
            # Augmentation
            wav = self.wavpt_aug(wav)
        return self.embedding_audio_from_tensor(wav)

    def embedding_audio_from_tensor(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract audio features using CLAP from a tensor.

        Args:
        ----
            wav (torch.Tensor): Audio tensor.

        Returns:
        -------
            torch.Tensor: Extracted audio features. [512]

        """
        assert isinstance(wav, torch.Tensor), "Input must be a torch tensor."
        wav = wav / (wav.abs().max() * 1.0001)
        # Preprocess the audio
        inputs = self.processor(audios=wav, sampling_rate=CLAP_SR, return_tensors="pt", padding=True).to(self.device)  # pyright: ignore[reportCallIssue]
        # Extract features
        with torch.no_grad():
            out = self.model.get_audio_features(**inputs)  # pyright: ignore[reportArgumentType]
        return out.squeeze(0)

    def wavpt_aug(self, x: torch.Tensor, random_length_th: float = 3) -> torch.Tensor:
        """Apply augmentation to the audio tensor."""
        # 時間をランダムに短くする
        if torch.rand(1).item() < RANDOM_TH:
            audio_length1 = x.shape[-1]
            audio_time_length = audio_length1 / CLAP_SR
            if audio_time_length > random_length_th:
                random_length = torch.randint(int(random_length_th * CLAP_SR), audio_length1, (1,)).item()
                random_start = torch.randint(0, audio_length1 - int(random_length), (1,)).item()
                x = x[random_start : random_start + int(random_length)]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pitch_shift = int(torch.randint(-150, 150 + 1, (1,)).item())
        time_wrap = float(torch.empty(1).uniform_(0.95, 1.05).item())
        effects = [
            ["tempo", str(time_wrap)],
            ["pitch", str(pitch_shift)],
        ]
        x, _ = torchaudio.sox_effects.apply_effects_tensor(x, CLAP_SR, effects)
        if x.dim() == SQUEEZE_DIM:
            x = x.squeeze(0)
        return x

    def embedding_text(self, text: str) -> torch.Tensor:
        """Extract text features using CLAP.

        Args:
        ----
            text (str): Text to be processed.

        Returns:
        -------
            torch.Tensor: Extracted text features. [512]

        """
        # Preprocess the text
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)  # pyright: ignore[reportCallIssue]

        # Extract features
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)  # pyright: ignore[reportArgumentType]
        return out.squeeze(0)

    def back_translate(self, text: str) -> str:
        """Back translate the text from English to German and then back to English.

        Args:
        ----
            text (str): Text to be processed.

        Returns:
        -------
            str: Back translated text.

        """
        # Translate from English to German
        de = pipeline("translation", model=self.en2de, tokenizer=self.en2de_tok)(text)[0]["translation_text"]  # pyright: ignore[reportIndexIssue,reportOptionalSubscript,reportArgumentType]
        en = pipeline("translation", model=self.de2en, tokenizer=self.de2en_tok)(de)[0]["translation_text"]  # pyright: ignore[reportIndexIssue,reportOptionalSubscript,reportArgumentType]
        assert isinstance(en, str), f"Back translation failed: {en}"
        return en

    def parrot_aug(self, text: str) -> list[str]:
        """Paraphrase the text using the Parrot model.

        Args:
        ----
            text (str): Text to be paraphrased.

        Returns:
        -------
            str: Paraphrased text.

        """
        return self.parrot.augment(text, max_return_phrases=10, use_gpu=True, max_length=128)  # pyright: ignore[reportReturnType]


if __name__ == "__main__":
    # Example usage
    model = ClapEmbedding()
    x = model.embedding_audio("src/examples/sample.wav")
    print(x.shape)  # Should print the shape of the extracted features

    y = model.embedding_text("a man voice.")
    print(y.shape)  # Should print the shape of the extracted features

    sim = torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
    print(sim)  # Should print the cosine similarity between audio and text features
