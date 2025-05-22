from pathlib import Path

import pandas as pd


def to_answer(
    input_csv: str,
    output_txt: str,
) -> None:
    """Convert the input CSV file to a text file with sample_id and pred_mos."""
    # Read the CSV file into a dataframe
    data_df = pd.read_csv(input_csv)

    # Extract sample_id and pred_mos columns
    sample_id_list = data_df["audio"].tolist()
    pred_mi_list = data_df["pred_mi"].tolist()
    pred_ta_list = data_df["pred_ta"].tolist()

    sample_id_list = [Path(f).stem for f in sample_id_list]

    output_fp = Path(output_txt)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    # Write the sample_id and pred_mos to the output text file
    with output_fp.open("w") as f:
        for sample_id, pred_mi, pred_ta in zip(sample_id_list, pred_mi_list, pred_ta_list, strict=False):
            f.write(f"{sample_id},{pred_mi},{pred_ta}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV to text file")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to the output text file")
    args = parser.parse_args()
    input_csv = args.input_csv
    output_txt = args.output_txt
    # Run the to_answer function
    to_answer(input_csv, output_txt)
