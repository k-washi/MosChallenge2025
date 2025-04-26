from pathlib import Path

import pandas as pd
from tqdm import tqdm

CONTRASTIVE_DATASET_LIST1 = [
    ("199", "188"),
    ("199", "187"),
    ("199", "186"),
    ("188", "187"),
    ("188", "186"),
    ("187", "186"),
    ("199", "177"),
    ("199", "176"),
    ("177", "176"),
]

SCORE_SET_DIR = ["199"]


def get_labeldata_list(
    dataset_csv_list: list[str],
) -> tuple[
    list[tuple[tuple[str | Path, int], tuple[str | Path, int]]],
    list[tuple[str | Path, int, str]],
]:
    """Get the file paths and labels from the dataset.

    Args:
    ----
        dataset_csv_list (list[str]): List of dataset CSV files.

    Returns:
    -------
        list[tuple[tuple[str | Path, int], tuple[str | Path, int]]]: List of tuples containing file paths and labels.
        list[tuple[str | Path, int, str]]: List of tuples containing file paths, labels, and dataset names.

    """
    output_list = []
    score_output_list = []

    for csv_fp in dataset_csv_list:
        data_df = pd.read_csv(csv_fp)
        dataset_dir = Path(csv_fp).parent
        dataset_name = dataset_dir.stem
        tmp_audio_list = data_df["wav"].tolist()
        tmp_score_list = data_df["score"].tolist()

        # 音質の差に関するデータの処理
        for better_dir_name, worse_dir_name in tqdm(
            [
                *CONTRASTIVE_DATASET_LIST1,
            ],
            desc=f"Processing {csv_fp}",
        ):
            is_better_set_score = better_dir_name in SCORE_SET_DIR
            is_worse_set_score = worse_dir_name in SCORE_SET_DIR

            for audio_fn, score in zip(tmp_audio_list, tmp_score_list, strict=False):
                beter_fn = dataset_dir / better_dir_name / f"{Path(audio_fn).stem}.flac"
                if not beter_fn.exists():
                    beter_fn = dataset_dir / better_dir_name / f"{Path(audio_fn).stem}.wav"
                    if not beter_fn.exists():
                        continue
                worse_fn = dataset_dir / worse_dir_name / f"{Path(audio_fn).stem}.flac"
                if not worse_fn.exists():
                    worse_fn = dataset_dir / worse_dir_name / f"{Path(audio_fn).stem}.wav"
                    if not worse_fn.exists():
                        continue
                better_score, worse_score = -1, -1
                if is_better_set_score:
                    better_score = score
                if is_worse_set_score:
                    worse_score = score
                output_list.append(((beter_fn, better_score), (worse_fn, worse_score)))

        # MOSによるデータの処理

        for audio_dir in SCORE_SET_DIR:
            for audio_fn, score in tqdm(zip(tmp_audio_list, tmp_score_list, strict=False), desc=f"Processing {csv_fp}"):
                data_fn = dataset_dir / audio_dir / f"{Path(audio_fn).stem}.flac"
                if not data_fn.exists():
                    data_fn = dataset_dir / audio_dir / f"{Path(audio_fn).stem}.wav"
                    if not data_fn.exists():
                        print(f"Data1 file {data_fn} does not exist.")
                        continue

                score_output_list.append((data_fn, score, dataset_name))

        print(f"{csv_fp}: {len(output_list)}")
    return output_list, score_output_list


if __name__ == "__main__":
    dataset_list = [
        "/data/mosranking/somos/train.csv",
        "/data/mosranking/bvccmain/train.csv",
        "/data/mosranking/track3/fold_0.csv",
    ]
    output_list, score_output_list = get_labeldata_list(dataset_list)
    print(f"len(output_list): {len(output_list)}")
    for data in output_list[-10:]:
        print(data)

    print(f"len(score_output_list): {len(score_output_list)}")
    for data in score_output_list[-10:]:
        print(data)
