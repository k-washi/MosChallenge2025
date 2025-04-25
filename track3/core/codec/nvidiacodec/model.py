import torch
from nemo.collections.tts.models import AudioCodecModel


class NCodec:
    """ESCodec class for encoding and decoding audio using the ESC model."""

    def __init__(
        self,
        device: torch.device,
    ) -> None:
        """Initialize the ESCodec with the given configuration and model paths.

        Args:
        ----
            device (torch.device): Device to run the model on.

        """
        self.model = AudioCodecModel.from_pretrained("nvidia/low-frame-rate-speech-codec-22khz")
        self.model = self.model.to(device)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.eval()  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of the model.

        Returns
        -------
            int: Sample rate of the model.

        """
        return self.model.sample_rate  # pyright: ignore[reportAttributeAccessIssue]

    def encode(self, audio: torch.Tensor, audio_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input audio using the ESCodec model.

        Args:
        ----
            audio (torch.Tensor): Input audio tensor.
            audio_len (torch.Tensor): Length of the input audio tensor.

        Returns:
        -------
            torch.Tensor: Encoded audio tensor.

        """
        with torch.no_grad():
            enctokens, enc_len = self.model.encode(audio=audio, audio_len=audio_len)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
        return enctokens, enc_len

    def decode(self, encoded_audio: torch.Tensor, enc_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode the encoded audio using the ESCodec model.

        Args:
        ----
            encoded_audio (torch.Tensor): Encoded audio tensor.
            enc_len (torch.Tensor): Length of the encoded audio tensor.

        Returns:
        -------
            torch.Tensor: Decoded audio tensor.

        """
        with torch.no_grad():
            decoded_audio, audio_len = self.model.decode(tokens=encoded_audio, tokens_len=enc_len)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
        return decoded_audio, audio_len
