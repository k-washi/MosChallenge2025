import torch
from transformers import ClapModel, ClapProcessor  # pyright: ignore[reportPrivateImportUsage]

from src.audio import load_wave

CLAP_SR = 48000
CPU_DEVICE = torch.device("cpu")


class ClapEmbedding:
    """A class for extracting audio and text features using CLAP (Contrastive Language-Audio Pretraining)."""

    def __init__(self, pretrained_model_name: str = "laion/clap-htsat-fused", device: torch.device = CPU_DEVICE) -> None:
        """Initialize the CLAP model."""
        super().__init__()
        self.device = device
        self.processor = ClapProcessor.from_pretrained(pretrained_model_name)
        self.model = ClapModel.from_pretrained(pretrained_model_name).to(device)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    def embedding_audio(self, wav_path: str) -> torch.Tensor:
        """Extract audio features using CLAP.

        Args:
        ----
            wav_path (str): Path to the audio file.

        Returns:
        -------
            torch.Tensor: Extracted audio features. [512]

        """
        # Load the audio file
        wav, _ = load_wave(wav_path, sample_rate=CLAP_SR, mono=True, is_torch=True)
        assert isinstance(wav, torch.Tensor), f"Audio file {wav_path} is not a torch tensor."
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


if __name__ == "__main__":
    # Example usage
    model = ClapEmbedding()
    x = model.embedding_audio("src/examples/sample.wav")
    print(x.shape)  # Should print the shape of the extracted features

    y = model.embedding_text("a man voice.")
    print(y.shape)  # Should print the shape of the extracted features

    sim = torch.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))
    print(sim)  # Should print the cosine similarity between audio and text features
