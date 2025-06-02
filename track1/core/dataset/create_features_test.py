from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from track1.core.modules.aux_feature import ExtractAuxFeatures
from track1.core.modules.clap import ClapEmbedding

WAV_DIR = "wav"
PROMPT_INFO_TXT = "prompt_info.txt"
DEMO_PROMPT_INFO_TEXT = "demo_prompt_info.txt"

PROMPT_OUTDIR = "prompt"
AUDIOEMB_OUTDIR = "audio"
AUXEMB_OUTDIR = "aux"

AUG_TIME = 20


def create_features(
    dataset_dir: str,
    output_dir: str,
    clap_model: str = "laion/clap-htsat-fused",
    audio_tag_ckpt: str = "pretrained/Cnn14_16k_mAP=0.438.pth",
) -> None:
    """Create features from audio files in a dataset directory.

    Args:
    ----
        dataset_dir (str): Directory containing audio files.
        output_dir (str): Directory to save the created features.
        clap_model (str): Name of the CLAP model to use.
        audio_tag_ckpt (str): Path to the audio tag checkpoint.

    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    prompt_outdir = Path(output_dir) / PROMPT_OUTDIR
    audio_outdir = Path(output_dir) / AUDIOEMB_OUTDIR
    auxout_dir = Path(output_dir) / AUXEMB_OUTDIR
    prompt_outdir.mkdir(parents=True, exist_ok=True)
    audio_outdir.mkdir(parents=True, exist_ok=True)
    auxout_dir.mkdir(parents=True, exist_ok=True)

    audio16k_dir = Path(dataset_dir) / WAV_DIR
    prompt_text_fp = Path(dataset_dir) / PROMPT_INFO_TXT
    demo_prompt_text_fp = Path(dataset_dir) / DEMO_PROMPT_INFO_TEXT

    device = torch.device("cpu")
    clap = ClapEmbedding(device=device, pretrained_model_name=clap_model)
    auxfeat_extractor = ExtractAuxFeatures(
        cnn14_ckpt=audio_tag_ckpt,
    )

    prompt_df = pd.read_csv(prompt_text_fp, sep="\t", header=None).dropna()
    demo_prompt_df = pd.read_csv(demo_prompt_text_fp, sep="\t", header=None).dropna()
    demo_prompt_dict = {row[0]: row[1] for row in demo_prompt_df.to_numpy()}
    prompt_dict = {row[0]: row[1] for row in prompt_df.to_numpy() if row[0].startswith("P")}

    for audio_fp in tqdm(list(audio16k_dir.glob("*"))):
        output_pt_fp = audio_outdir / f"{audio_fp.stem}.pt"
        audio_embedding = clap.embedding_audio(str(audio_fp))
        torch.save(audio_embedding, output_pt_fp)

        # aux:
        output_pt_fp = auxout_dir / f"{audio_fp.stem}.pt"
        aux_emebedding = auxfeat_extractor.extract_aux_features(
            str(audio_fp),
        )
        torch.save(aux_emebedding, output_pt_fp)

        # prompt:
        output_pt_fp = prompt_outdir / f"{audio_fp.stem}.pt"
        if audio_fp.name in demo_prompt_dict:
            prompt = demo_prompt_dict[audio_fp.name]
            text_embedding = clap.embedding_text(prompt)
            torch.save(text_embedding, output_pt_fp)
        elif audio_fp.stem.split("_")[-1] in prompt_dict:
            prompt = prompt_dict[audio_fp.stem.split("_")[-1]]
            text_embedding = clap.embedding_text(prompt)
            torch.save(text_embedding, output_pt_fp)
        else:
            emsg = f"Audio file {audio_fp.name} not found in prompt dictionary."
            raise ValueError(emsg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create features from audio files.")
    parser.add_argument(
        "--dataset_dir", type=str, default="data/audiomos2025-track1-eval-phase/DATA", help="Directory containing audio files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/audiomos2025-track1-eval-phase/DATA",
        help="Directory to save the created features.",
    )
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-fused", help="CLAP model name.")
    parser.add_argument("--audio_tag_ckpt", type=str, default="pretrained/Cnn14_16k_mAP=0.438.pth", help="Audio tag checkpoint.")
    args = parser.parse_args()

    create_features(args.dataset_dir, args.output_dir, args.clap_model, args.audio_tag_ckpt)
