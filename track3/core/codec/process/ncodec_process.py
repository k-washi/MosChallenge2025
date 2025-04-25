from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from src.audio import load_wave, save_wave
from track3.core.codec.nvidiacodec.model import NCodec

CODEC_LIST = [
    ("76", 7.0 / 10),
    ("77", 9.0 / 10),
]

TARGET_SR = 16000
TRIM_LEN = 256
DIRLEN = 3


def codec_process(
    target_dir: str,
    batch_size: int = 1,
) -> None:
    """Process the target directory using the ESCodec model.

    Args:
    ----
        target_dir (str): Path to the target directory containing wav files.
        batch_size (int): Batch size for processing. Defaults to 1.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec_model = NCodec(
        device=device,
    )
    target_name = Path(target_dir).stem
    assert len(target_name) == DIRLEN, f"target_name {target_name} is not valid"
    audio_list = sorted(list(Path(target_dir).glob("*")))
    output_dir = Path(target_dir).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for codec_name, no_noise_rate in CODEC_LIST:
        print(f"Processing {codec_name}")
        output_codec_dir = output_dir / f"{target_name[0]}{codec_name}"
        output_codec_dir.mkdir(parents=True, exist_ok=True)

        batch_audio_list, batch_len_list, batch_fp_list = [], [], []
        audio_max_len = 0
        for audio_fp in tqdm(audio_list, desc=f"Processing {output_codec_dir.name}"):
            if len(batch_audio_list) < batch_size:
                audio, _ = load_wave(str(audio_fp), sample_rate=codec_model.sample_rate, mono=True, is_torch=True)
                audio_len = audio.shape[-1]
                if audio_max_len < audio_len:
                    audio_max_len = audio_len
                batch_audio_list.append(audio)
                batch_len_list.append(audio_len)
                batch_fp_list.append(audio_fp)
                continue

            batch_audio = torch.zeros((len(batch_audio_list), audio_max_len), dtype=torch.float32)
            for i in range(len(batch_audio_list)):
                audio = batch_audio_list[i]
                audio_len = batch_len_list[i]
                batch_audio[i, :audio_len] = audio
            batch_len = torch.tensor(batch_len_list, dtype=torch.int32)
            batch_audio = batch_audio.to(device)
            batch_len = batch_len.to(device)
            codes, c_len = codec_model.encode(batch_audio, batch_len)
            if no_noise_rate < 1:
                for i in range(len(batch_audio_list)):
                    codes_max = int(codes[i].max())
                    codes_length = int(codes_max * no_noise_rate)
                    rand_min_value = torch.randint(0, codes_max, (1,)).item()
                    noise_length = codes_max - codes_length
                    noise_length = torch.randint(noise_length // 2, noise_length, (1,)).item()
                    rand_max_value = rand_min_value + noise_length
                    renge_index = torch.where((codes[i] >= rand_min_value) & (codes[i] <= rand_max_value))
                    codes[i][renge_index] = torch.randint_like(codes[i][renge_index], codes_max, device=device, dtype=codes.dtype)
            batch_audio, batch_len = codec_model.decode(codes, c_len)

            batch_audio = batch_audio.cpu()
            batch_len = batch_len.cpu()
            for audio, audio_len, audio_fp in zip(batch_audio, batch_len, batch_fp_list, strict=False):
                audio = audio[:audio_len]
                output_wav_fp = output_codec_dir / audio_fp.name
                if codec_model.sample_rate != TARGET_SR:
                    audio = torchaudio.functional.resample(audio, codec_model.sample_rate, TARGET_SR)
                audio = audio[:-TRIM_LEN]
                save_wave(audio, str(output_wav_fp), sample_rate=TARGET_SR)
            batch_audio_list, batch_len_list, batch_fp_list = [], [], []
            audio_max_len = 0

        # 最後のバッチ
        if len(batch_audio_list) == 0:
            continue
        # 最後のプロセス
        batch_audio = torch.zeros((len(batch_audio_list), audio_max_len), dtype=torch.float32)
        for i in range(len(batch_audio_list)):
            audio = batch_audio_list[i]
            audio_len = batch_len_list[i]
            batch_audio[i, :audio_len] = audio
        batch_len = torch.tensor(batch_len_list, dtype=torch.int32)
        batch_audio = batch_audio.to(device)
        batch_len = batch_len.to(device)
        codes, c_len = codec_model.encode(batch_audio, batch_len)
        if no_noise_rate < 1:
            for i in range(len(batch_audio_list)):
                codes_max = int(codes[i].max())
                codes_length = int(codes_max * no_noise_rate)
                rand_min_value = torch.randint(0, codes_max, (1,)).item()
                noise_length = codes_max - codes_length
                noise_length = torch.randint(noise_length // 2, noise_length, (1,)).item()
                rand_max_value = rand_min_value + noise_length
                renge_index = torch.where((codes[i] >= rand_min_value) & (codes[i] <= rand_max_value))
                codes[i][renge_index] = torch.randint_like(codes[i][renge_index], codes_max, device=device, dtype=codes.dtype)
        batch_audio, batch_len = codec_model.decode(codes, c_len)
        batch_audio = batch_audio.cpu()
        batch_len = batch_len.cpu()
        for audio, audio_len, audio_fp in zip(batch_audio, batch_len, batch_fp_list, strict=False):
            audio = audio[:audio_len]
            output_wav_fp = output_codec_dir / audio_fp.name
            if codec_model.sample_rate != TARGET_SR:
                audio = torchaudio.functional.resample(audio, codec_model.sample_rate, TARGET_SR)
            audio = audio[:-TRIM_LEN]
            save_wave(audio, str(output_wav_fp), sample_rate=TARGET_SR)

        print(f"Finished processing {codec_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process the target directory using the ESCodec model.")
    parser.add_argument(
        "--target_dir", type=str, default="/data/mosranking/track3/199", help="Path to the target directory containing wav files."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")

    args = parser.parse_args()

    codec_process(args.target_dir, args.batch_size)
