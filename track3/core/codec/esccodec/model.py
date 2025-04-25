from pathlib import Path

import torch
import yaml
from esc import ESC  # pyright: ignore[reportMissingImports]


def sub_padding(x: torch.Tensor, multiple: int = 160, sub_multiple: int = 4) -> torch.Tensor:
    """Sub pad the input tensor to the nearest multiple of the specified value.

    Args:
    ----
        x (torch.Tensor): Input tensor.
        multiple (int): The value to pad to. Defaults to 160.
        sub_multiple (int): The value to sub pad to. Defaults to 4.

    Returns:
    -------
        torch.Tensor: Padded tensor.

    """
    _, t = x.shape
    pad = (t) % multiple
    if pad > 0:
        x = x[:, :-pad]
    # multipleの倍数になっている
    _, t = x.shape
    sub_t = t / multiple
    if sub_t % sub_multiple != 0:
        sub_t = int(sub_t)
        pad = (sub_t) % sub_multiple
        if pad > 0:
            x = x[:, : -pad * multiple]

    return x


class ESCodec:
    """ESCodec class for encoding and decoding audio using the ESC model."""

    def __init__(
        self,
        config_pth: str,
        model_pth: str,
    ) -> None:
        """Initialize the ESCodec with the given configuration and model paths.

        Args:
        ----
            config_pth (str): Path to the configuration file.
            model_pth (str): Path to the model file.

        """
        self.config_pth = config_pth
        self.model_pth = model_pth
        with Path(config_pth).open("r") as f:
            config = yaml.safe_load(f)
        state_dict = torch.load(model_pth, map_location="cpu")["model_state_dict"]
        self.model = ESC(**config["model"])
        self.model.load_state_dict(state_dict)

    def encode(
        self,
        audio: torch.Tensor,
        num_streams: int = 6,
        multiple: int = 160,
        sub_multiple: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input audio using the ESCodec model.

        Args:
        ----
            audio (torch.Tensor): Input audio tensor.
            num_streams (int): Number of streams to use for encoding (num_streams*1.5kbps).
                Defaults to 6.
            multiple (int): The value to pad to. Defaults to 160.
            sub_multiple (int): The value to sub pad to. Defaults to 4.

        Returns:
        -------
            torch.Tensor: Encoded audio tensor.

        """
        audio = sub_padding(audio, multiple=multiple, sub_multiple=sub_multiple)
        with torch.no_grad():
            codes, f_shape = self.model.encode(audio, num_streams)
        return codes, f_shape

    def decode(self, encoded_audio: torch.Tensor, f_shape: torch.Tensor) -> torch.Tensor:
        """Decode the encoded audio using the ESCodec model.

        Args:
        ----
            encoded_audio (torch.Tensor): Encoded audio tensor.
            f_shape (torch.Tensor): Shape of the encoded audio tensor.

        Returns:
        -------
            torch.Tensor: Decoded audio tensor.

        """
        with torch.no_grad():
            decoded_audio = self.model.decode(encoded_audio, f_shape)
        return decoded_audio

    def encdec(self, audio: torch.Tensor, num_streams: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode the input audio using the ESCodec model.

        Args:
        ----
            audio (torch.Tensor): Input audio tensor.
            num_streams (int): Number of streams to use for encoding.
                Defaults to 6.
                num_streams*1.5kbps

        Returns:
        -------
            tuple: Encoded and decoded audio tensors.

        """
        codes, f_shape = self.encode(audio, num_streams)
        decoded_audio = self.decode(codes, f_shape)
        return codes, decoded_audio


if __name__ == "__main__":
    from src.audio import load_wave, save_wave

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_pth = "pretrained/esc9kbps_large_non_adversarial/config.yaml"
    model_pth = "pretrained/esc9kbps_large_non_adversarial/model.pth"
    esc = ESCodec(config_pth, model_pth)
    esc.model.to(device)
    esc.model.eval()
    print(esc)

    test_fp = "data/librittsr/14_208_000001_000000.wav"
    audio, sr = load_wave(test_fp, 16000, mono=True)
    assert isinstance(audio, torch.Tensor), f"Expected torch.Tensor, got {type(audio)}"
    audio = audio.unsqueeze(0)
    audio = audio.to(device)
    print(audio.shape)
    codes, f_shape = esc.encode(audio, 6)
    print(codes.max(), codes.min())
    rand_int = 50
    rand_min = 4
    codes[:, 1:6, :, 100:120] += (
        torch.randint_like(codes[:, 1:6, :, 100:120], rand_int, device=device, dtype=codes.dtype) - rand_int // 2
    )
    codes = torch.clamp(codes, 0, 1021)
    print(codes.max(), codes.min())

    print(codes.shape, f_shape)
    decoded_audio = esc.decode(codes, f_shape)
    print(decoded_audio.shape)
    save_wave(decoded_audio.squeeze(0).detach().cpu(), "data/librittsr/decoded_n6.wav", 16000)

    """
    torch.Size([1, 144000])
    torch.Size([1, 6, 3, 450]) (2, 900)
    torch.Size([1, 143920])
    --------------------
    torch.Size([1, 4, 3, 450]) (2, 900)
    torch.Size([1, 143920])
    --------------------
    torch.Size([1, 2, 3, 450]) (2, 900)
    torch.Size([1, 143920])
    --------------------
    torch.Size([1, 1, 3, 450]) (2, 900)
    torch.Size([1, 143920])
    """
