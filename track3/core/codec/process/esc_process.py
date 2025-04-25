from pathlib import Path

import torch
from tqdm import tqdm

from src.audio import load_wave, save_wave
from track3.core.codec.esccodec.model import ESCodec

CODEC_LIST = [("88", 6), ("87", 2), ("86", 1)]

TARGET_SR = 16000

DIRLEN = 3


def codec_process(
    target_dir: str,
    esc_config_pth: str,
    esc_model_pth: str,
) -> None:
    """Process the target directory using the ESCodec model.

    Args:
    ----
        target_dir (str): Path to the target directory containing wav files.
        esc_config_pth (str): Path to the ESCodec config file.
        esc_model_pth (str): Path to the ESCodec model file.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec_model = ESCodec(
        config_pth=esc_config_pth,
        model_pth=esc_model_pth,
    )
    codec_model.model.to(device)
    codec_model.model.eval()

    target_name = Path(target_dir).stem
    assert len(target_name) == DIRLEN, f"target_name {target_name} is not valid"
    audio_list = sorted(list(Path(target_dir).glob("*")))
    output_dir = Path(target_dir).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for codec_name, num_streams in CODEC_LIST:
        print(f"Processing {codec_name}")
        output_codec_dir = output_dir / f"{target_name[0]}{codec_name}"
        output_codec_dir.mkdir(parents=True, exist_ok=True)
        for audio_fp in tqdm(audio_list, desc=f"Processing {output_codec_dir.name}"):
            audio, _ = load_wave(str(audio_fp), sample_rate=TARGET_SR, mono=True, is_torch=True)
            assert isinstance(audio, torch.Tensor), f"Expected torch.Tensor, got {type(audio)}"
            audio = audio.unsqueeze(0).to(device)
            try:
                _, audio = codec_model.encdec(audio, num_streams)
            except Exception as e:
                print(f"Error processing {audio_fp.name}: {e}")
                continue
            assert isinstance(audio, torch.Tensor), f"Expected torch.Tensor, got {type(audio)}"
            audio = audio.squeeze(0).detach().cpu()
            output_wav_fp = output_codec_dir / audio_fp.name
            save_wave(audio, str(output_wav_fp), sample_rate=TARGET_SR)
        print(f"Finished processing {codec_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process the target directory using the ESCodec model.")
    parser.add_argument(
        "--target_dir", type=str, default="/data/mosranking/track3/199", help="Path to the target directory containing wav files."
    )
    parser.add_argument(
        "--esc_config_pth",
        type=str,
        default="pretrained/esc9kbps_large_non_adversarial/config.yaml",
        help="Path to the ESCodec config file.",
    )
    parser.add_argument(
        "--esc_model_pth",
        type=str,
        default="pretrained/esc9kbps_large_non_adversarial/model.pth",
        help="Path to the ESCodec model file.",
    )

    args = parser.parse_args()

    codec_process(args.target_dir, args.esc_config_pth, args.esc_model_pth)
