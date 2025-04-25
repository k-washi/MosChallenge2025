from pathlib import Path

import pandas as pd

csv_list = [
    "utt_16k.csv",
    "utt_24k.csv",
    "utt_48k.csv",
]


def input_dir(
    input_dir: str,
    output_dir: str,
    fold: int = 5,
) -> None:
    """Copy wav files from input directory to output directory.

    Args:
    ----
        input_dir (str): Path to the input directory containing wav files.
        output_dir (str): Path to the output directory where wav files will be copied.
        fold (int): Number of folds for cross-validation.

    """
    indir = Path(input_dir)
    outdir = Path(output_dir)

    # Create output directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    utt_map = {}
    utt_list = []
    for csv_file in csv_list:
        csv_fp = indir / csv_file
        if not csv_fp.exists():
            print(f"CSV file {csv_file} does not exist in {input_dir}")
            continue

        data_df = pd.read_csv(csv_fp)
        for wav, score in zip(data_df["wav"], data_df["score"], strict=False):
            utt_name = Path(wav).stem.split("_")[-1]
            if utt_name not in utt_map:
                utt_map[utt_name] = {}
                utt_list.append(utt_name)

            if wav not in utt_map[utt_name]:
                utt_map[utt_name][wav] = []
            utt_map[utt_name][wav].append(score)

    utt_list = sorted(utt_list)
    fold_list = [utt_list[i::fold] for i in range(fold)]
    for i in range(fold):
        fold_output_list = []
        for utt_name in fold_list[i]:
            if utt_name not in utt_map:
                continue
            for wav, score_list in utt_map[utt_name].items():
                score = sum(score_list) / len(score_list)
                fold_output_list.append((wav, score))
        fold_output_df = pd.DataFrame(fold_output_list, columns=["wav", "score"])  # pyright: ignore[reportArgumentType]
        fold_output_fp = outdir / f"fold_{i}.csv"
        fold_output_df.to_csv(fold_output_fp, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy wav files from input directory to output directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing wav files.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory where wav files will be copied."
    )
    parser.add_argument("--fold", type=int, default=5, help="Number of folds for cross-validation.")
    args = parser.parse_args()

    input_dir(args.input_dir, args.output_dir, args.fold)
