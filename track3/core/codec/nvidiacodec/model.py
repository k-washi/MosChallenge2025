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
        self.model.to(device)  # pyright: ignore[reportAttributeAccessIssue]
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

    def decode(self, encoded_audio: torch.Tensor, enc_len: torch.Tensor) -> torch.Tensor:
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
            decoded_audio, _ = self.model.decode(tokens=encoded_audio, tokens_len=enc_len)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
        return decoded_audio

    def encdec(self, audio: torch.Tensor, audio_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode the input audio using the ESCodec model.

        Args:
        ----
            audio (torch.Tensor): Input audio tensor.
            audio_len (torch.Tensor): Length of the input audio tensor.

        Returns:
        -------
            tuple: Encoded and decoded audio tensors.

        """
        codes, c_len = self.encode(audio, audio_len)
        decoded_audio = self.decode(codes, c_len)
        return codes, decoded_audio


if __name__ == "__main__":
    import torchaudio

    from src.audio import load_wave, save_wave

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_pth = "pretrained/esc9kbps_large_non_adversarial/config.yaml"
    model_pth = "pretrained/esc9kbps_large_non_adversarial/model.pth"
    esc = NCodec(device=device)

    test_fp = "data/librittsr/14_208_000001_000000.wav"
    audio, sr = load_wave(test_fp, esc.sample_rate, mono=True)
    assert isinstance(audio, torch.Tensor), f"Expected torch.Tensor, got {type(audio)}"
    audio = audio.unsqueeze(0)
    audio = audio.to(device)
    audio_len = torch.tensor([audio.shape[-1]], device=device)
    codes, c_len = esc.encode(audio, audio_len)
    rand_int = 50
    rand_min = 4
    # cand1 codes = torch.clamp(codes, min=0, max=int(codes.max()*9/10)) #p1
    # cand2 codes = torch.clamp(codes, min=0, max=int(codes.max() * 7 / 10))  # p1

    decoded_audio = esc.decode(codes, c_len)
    decoded_audio = torchaudio.functional.resample(decoded_audio, 22050, 16000)
    save_wave(decoded_audio.squeeze(0).detach().cpu(), "data/librittsr/ndecoded.wav", 16000)
