# 48kに拡張したデータを修正する


import shutil
from pathlib import Path

from tqdm import tqdm


def sr_to_norm(
    sr_dir: str,
    output_dir: str,
) -> None:
    """Convert the sample rate of audio files in a directory to a normalized format.

    Args:
    ----
        sr_dir (str): Directory containing audio files with original sample rates.
        output_dir (str): Directory to save the converted audio files.

    """
    sr_audio_fp_list = sorted(list(Path(sr_dir).glob("*")))
    assert len(sr_audio_fp_list) > 0, f"sr_dir: {sr_dir} is empty"
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    for sr_audio_fp in tqdm(sr_audio_fp_list):
        # remove xxx_AudioSR_Processed_48K
        audio_name = sr_audio_fp.name.replace("_AudioSR_Processed_48K", "")
        output_fp = odir / audio_name

        shutil.copy(str(sr_audio_fp), str(output_fp))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert audio sample rate")
    parser.add_argument(
        "--sr_dir",
        type=str,
        default="data/MusicEval-phase1/wav48k_audiosr/28_04_2025_14_49_29",
        help="Directory containing audio files with original sample rates.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/mos/track3/wav48k",
        help="Directory to save the converted audio files.",
    )

    args = parser.parse_args()
    sr_to_norm(args.sr_dir, args.output_dir)
