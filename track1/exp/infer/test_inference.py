"""Test Inference.

python ./track1/exp/infer/test_inference.py \
--ckpt_file ./logs/mosnet_aux/v03003/ckpt/ckpt-4032/model.ckpt \
--output_fp ./data/track1/3003_4032.csv \
--dataset_dir ./data/audiomos2025-track1-eval-phase/DATA/
"""

import math
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from tqdm import tqdm
from transformers import AutoTokenizer  # pyright: ignore[reportPrivateImportUsage]

from src.audio.utils import load_wave
from track1.core.config import Config
from track1.core.model.mospl_aux import MOSPredictorModule
from track1.exp.infer.getdataset import get_dataset

cfg = Config()
seed_everything(cfg.ml.seed)
##########
# Params #
##########

# model
cfg.model.model_name = "mosnet_aux"
cfg.model.mosnet.input_dim = 512
cfg.model.mosnet.aux_dim = 557
cfg.model.mosnet.hidden_dim = 1024
cfg.model.mosnet.tf_n_head = 16
cfg.model.mosnet.tf_n_layers = 3
cfg.model.mosnet.tf_dropout = 0.15
cfg.model.mosnet.head_dropout = 0.3


def inference(
    ckpt_file: str,
    output_fp: str,
    dataset_dir: str = "data/MusicEval-phase1",
) -> None:
    """Train the model."""
    audio_list, prompt_list = get_dataset(
        dataset_dir=dataset_dir,
    )
    assert len(audio_list) > 0, f"No audio files found in {dataset_dir}"
    model = MOSPredictorModule(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOSPredictorModule(cfg)
    model.load_state_dict(
        torch.load(ckpt_file),
        strict=True,
    )
    model.to(device)

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    output_list = []
    for audio_fp, prompt in tqdm(zip(audio_list, prompt_list, strict=False), total=len(audio_list)):
        assert Path(audio_fp).exists(), f"Audio file {audio_fp} does not exist."
        audio, sr = load_wave(
            str(audio_fp),
            sample_rate=cfg.data.sample_rate,
            mono=True,
            is_torch=True,
        )
        prompt_tok = tok(prompt, return_tensors="pt")

        # a audioをmax_durationごとに分割 (最後の部分はmax_duration)
        audio_length = audio.shape[-1]
        audio_split_length = int(cfg.data.max_duration * sr)
        audio_split_num = math.ceil(audio_length / audio_split_length)

        audio_feat_dir = Path(audio_fp).parent.parent
        clip_audio_emb = torch.load(
            str(audio_feat_dir / "audio" / f"{Path(audio_fp).stem}.pt"),
            map_location=device,
        )
        clip_prompt_emb = torch.load(
            str(audio_feat_dir / "prompt" / f"{Path(audio_fp).stem}.pt"),
            map_location=device,
        )

        tmp_audio_list = []
        for i in range(audio_split_num):
            start = i * audio_split_length
            end = min((i + 1) * audio_split_length, audio_length)
            if i > 0 and end - start < audio_split_length:
                # 最初の区切りではなく、かつ、長さがmax_durationの1/2以上なら処理に加える
                # 最後の部分はmax_durationにする
                if end - start < audio_split_length // 2:
                    # 1/2以下ならスキップ
                    continue
                start = end - audio_split_length
            tmp_audio = audio[start:end]
            tmp_audio = (tmp_audio - tmp_audio.mean()) / (tmp_audio.std() + 1e-7)
            tmp_audio_list.append(tmp_audio)

        # inference
        clip_audio_emb = clip_audio_emb.unsqueeze(0).to(device)
        clip_prompt_emb = clip_prompt_emb.unsqueeze(0).to(device)
        prompt_btensor = prompt_tok.input_ids.to(device)
        prompt_atm = torch.ones((1, prompt_tok["attention_mask"].shape[-1]), dtype=torch.float32, device=device)
        mi_list, ta_list = [], []
        for audio in tmp_audio_list:
            audio_atm = torch.ones((1, audio.shape[-1]), dtype=torch.float32, device=device)
            audio = audio.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_mi, pred_ta = model(
                    z_audio=clip_audio_emb,
                    z_prompt=clip_prompt_emb,
                    z_aux=None,
                    wav=audio,
                    wav_len=audio_atm,
                    text_ids=prompt_btensor,
                    text_len=prompt_atm,
                )
            pred_mi = pred_mi.cpu().item()
            pred_ta = pred_ta.cpu().item()
            pred_mi = (
                pred_mi * (cfg.data.label_norm_max - cfg.data.label_norm_min) + (cfg.data.label_min + cfg.data.label_max) / 2.0
            )
            pred_ta = (
                pred_ta * (cfg.data.label_norm_max - cfg.data.label_norm_min) + (cfg.data.label_min + cfg.data.label_max) / 2.0
            )
            mi_list.append(pred_mi)
            ta_list.append(pred_ta)
        assert len(mi_list) > 0, f"mi_list is empty for {audio_fp}"
        assert len(ta_list) > 0, f"ta_list is empty for {audio_fp}"
        pred_mi = sum(mi_list) / len(mi_list)
        pred_ta = sum(ta_list) / len(ta_list)
        sample_id = f"{Path(audio_fp).stem}"
        output = {
            "sample_id": sample_id,
            "pred_mi": pred_mi,
            "pred_ta": pred_ta,
        }
        output_list.append(output)

    output_csv = Path(output_fp)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(output_list)
    output_df.to_csv(output_csv, index=False)

    # to text file
    output_txt = output_csv.with_suffix(".txt")
    with output_txt.open("w") as f:
        for _, row in output_df.iterrows():
            f.write(f"{row['sample_id']},{row['pred_mi']},{row['pred_ta']}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference script for MOSPL model.")
    parser.add_argument("--ckpt_file", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--output_fp", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument(
        "--dataset_dir", type=str, default="data/audiomos2025-track1-eval-phase/DATA", help="Directory containing the dataset."
    )

    args = parser.parse_args()

    inference(
        ckpt_file=args.ckpt_file,
        output_fp=args.output_fp,
        dataset_dir=args.dataset_dir,
    )
