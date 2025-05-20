import math
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from tqdm import tqdm

from src.audio import load_wave
from track3.core.config import Config
from track3.core.dataset.contrastive.utils import get_labeldata_list
from track3.core.models.utmospl_sfds import MOSPredictorModule

cfg = Config()
seed_everything(cfg.ml.seed)
# model
cfg.model.model_name = "wavlm_conformer_ds"
cfg.model.w2v2.dropout = 0.3
cfg.model.w2v2.n_heads = 8
cfg.model.w2v2.n_layers = 3
cfg.model.w2v2.is_freeze_ssl = False
cfg.model.w2v2.ds_hidden_dim = 32
cfg.data.test_dataset_dict = {"track3": 0, "bvccmain": 1, "somos": 2}
cfg.data.max_duration = 5
cfg.data.is_label_normalize = True


def inference(ckpt_file: str, test_csv: str, output_dir: str) -> None:
    """Run inference on the test dataset and save the results."""
    _, test_dataset_list = get_labeldata_list(
        dataset_csv_list=[test_csv],
    )
    print(f"test_len: {len(test_dataset_list)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOSPredictorModule(cfg)
    model.load_state_dict(
        torch.load(ckpt_file),
        strict=True,
    )
    model.to(device)
    output_list = []
    for audio_file, true_mos, dataset_num in tqdm(test_dataset_list):
        assert Path(audio_file).exists(), f"Audio file {audio_file} does not exist."
        dataset_id = cfg.data.test_dataset_dict.get(dataset_num, -1)
        assert dataset_id >= 0, f"Dataset {dataset_num} not found in dataset_dict."

        audio, sr = load_wave(
            str(audio_file),
            sample_rate=cfg.data.sample_rate,
            mono=True,
            is_torch=True,
        )

        # a audioをmax_durationごとに分割 (最後の部分はmax_duration)
        audio_length = audio.shape[-1]
        audio_split_length = int(cfg.data.max_duration * sr)
        audio_split_num = math.ceil(audio_length / audio_split_length)
        audio_list = []
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
            audio_list.append(audio[start:end])
        mos_list = []
        for _, audio in enumerate(audio_list):
            audio = (audio - audio.mean()) / torch.sqrt(torch.var(audio) + 1e-7)
            audio = audio.unsqueeze(0)
            dataset_id = torch.tensor([dataset_id], dtype=torch.long)
            attention_mask = torch.ones(audio.shape[0], audio.shape[1], dtype=torch.long)
            audio = audio.to(device)
            dataset_id = dataset_id.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                pred_mos = model(audio, dataset_id, attention_mask)
            pred_mos = pred_mos.cpu().item()
            pred_mos = (
                pred_mos * (cfg.data.label_norm_max - cfg.data.label_norm_min) + (cfg.data.label_min + cfg.data.label_max) / 2.0
            )
            mos_list.append(pred_mos)

        assert len(mos_list) > 0, f"mos_list is empty for {audio_file}"
        pred_mos = sum(mos_list) / len(mos_list)
        sample_id = f"{Path(audio_file).stem}"
        output_list.append(
            {
                "sample_id": sample_id,
                "true_mos": true_mos,
                "pred_mos": pred_mos,
            }
        )
    # Save the output to a CSV file
    output_csv = Path(output_dir) / f"{Path(test_csv).stem}.csv"
    output_df = pd.DataFrame(output_list)
    output_df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")


if __name__ == "__main__":
    # Set the paths for the checkpoint file and test CSV
    import argparse

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--ckpt_file", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files")
    args = parser.parse_args()
    ckpt_file = args.ckpt_file
    test_csv = args.test_csv
    output_dir = args.output_dir
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Run inference
    inference(ckpt_file, test_csv, output_dir)
