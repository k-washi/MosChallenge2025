from pathlib import Path

from tqdm import tqdm

CONTRASTIVE_DATASET_LIST1 = [
    ("299", "288"),
    ("299", "287"),
    ("299", "286"),
    ("288", "287"),
    ("288", "286"),
    ("287", "286"),
    ("299", "277"),
    ("299", "276"),
    ("277", "276"),
]

CONTRASTIVE_DATASET_LIST2 = [
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

CONTRASTIVE_DATASET_LIST3 = [("299", "199"), ("299", "188"), ("299", "187"), ("299", "186"), ("299", "177"), ("299", "176")]


def get_nolabel_list(dataset_dir: str) -> list[tuple[tuple[str | Path, int], tuple[str | Path, int]]]:
    """Get the file paths from the dataset without labels.

    データのMOSの大小

    dataset_dir:
        -- 199
        -- 188
        -- 187
        -- 186
        -- 177
        -- 176
        -- 299
        -- 288
        -- 287
        -- 286
        -- 277
        -- 276
    299 > 288 > 287 > 286
    299 > 277 > 276
    199 > 188 > 187 > 186
    199 > 177 > 176

    x99のxは大きいほど音質が良い
    299 > 199, 188, 187, 186
    299 > 177, 176

    Args:
    ----
        dataset_dir (str): dataset dir

    Returns:
    -------
        list: A list of audio file paths.
        [((wav1, score1), (wav2, score2)), ...]
        score1 > score2 if score1 and score2 is not -1
        empty score is -1

    """
    output_list = []
    for better_mos_id, worse_mos_id in tqdm(
        [*CONTRASTIVE_DATASET_LIST1, *CONTRASTIVE_DATASET_LIST2, *CONTRASTIVE_DATASET_LIST3], desc="get_nolabel_list"
    ):
        better_mos_dir = Path(dataset_dir) / str(better_mos_id)
        worse_mos_dir = Path(dataset_dir) / str(worse_mos_id)
        if not better_mos_dir.is_dir():
            print(f"better dir is no exist {better_mos_dir}")
            continue
        if not worse_mos_dir.is_dir():
            print(f"worse dir is no exist {worse_mos_dir}")
            continue
        better_mos_list = sorted(list(better_mos_dir.glob("*")))
        for better_mos_fp in better_mos_list:
            worse_mos_fp = worse_mos_dir / better_mos_fp.name
            if not worse_mos_fp.exists():
                continue
            output_list.append(((better_mos_fp, -1), (worse_mos_fp, -1)))
    return output_list


if __name__ == "__main__":
    # a len(output_list): 2609076
    dataset_dir = "/data/mosranking/libritts"
    output_list = get_nolabel_list(dataset_dir)
    for (better_mos_fp, _), (worse_mos_fp, _) in output_list[:10]:
        print(better_mos_fp, worse_mos_fp)

    print(f"len(output_list): {len(output_list)}")
