from pathlib import Path
import shutil
from tqdm import tqdm
def copy_wav(
    input_dir: str,
    output_dir: str,
) -> None:
    
    """
    Copy wav files from input directory to output directory.
    
    Args:
        input_dir (str): Path to the input directory containing wav files.
        output_dir (str): Path to the output directory where wav files will be copied.
    """
    indir = Path(input_dir)
    outdir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    track_lsit = sorted(list(indir.glob("*")))
    for track in tqdm(track_lsit):
        if not track.is_dir():
            continue
        wav_list = sorted(list(track.glob("*/*.wav")))
        for wav in wav_list:
            wav_name = wav.name
            output_wav_fp = outdir / wav_name
            output_wav_fp.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(wav, output_wav_fp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Copy wav files from input directory to output directory.")
    parser.add_argument("--input_dir", type=str, default="data/LibriTTS_R/train-clean-360", help="Path to the input directory containing wav files.")
    parser.add_argument("--output_dir", type=str, default="data/librittsr/wav", help="Path to the output directory where wav files will be copied.")
    
    args = parser.parse_args()
    
    copy_wav(args.input_dir, args.output_dir)