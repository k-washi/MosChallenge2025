from pathlib import Path

from tqdm import tqdm

from src.audio import load_wave, save_wave

TARGET_SR = 16000


def convert_wav_to_flac16k(input_dir: str) -> None:
    """Convert wav files to flac16k format."""
    dataset_list = sorted(list(Path(input_dir).glob("*")))
    for dataset in dataset_list:
        if not dataset.is_dir():
            continue
        wav_list = sorted(list(dataset.glob("*/*.wav")))
        print(f"Converting {dataset.name}, {len(wav_list)} files")
        for wav_fp in tqdm(wav_list, desc=f"Converting {dataset.name}"):
            if wav_fp.suffix != ".wav":
                print(f"Skipping {wav_fp.name}, not a .wav file")
                continue
            output_wav_fp = wav_fp.with_suffix(".flac")
            audio, sr = load_wave(str(wav_fp), sample_rate=TARGET_SR, mono=True, is_torch=True)
            save_wave(audio, str(output_wav_fp), sample_rate=TARGET_SR)
            wav_fp.unlink(missing_ok=True)
        print(f"Finished converting {dataset.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert wav files to flac16k format.")
    parser.add_argument(
        "--input_dir", type=str, default="/data/mosranking", help="Path to the input directory containing wav files."
    )
    args = parser.parse_args()

    convert_wav_to_flac16k(args.input_dir)
