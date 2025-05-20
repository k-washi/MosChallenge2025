from pathlib import Path

import pandas as pd


def gather_inference(
    input_dir: str,
    output_txt: str,
) -> None:
    """Gather inference results from multiple CSV files into a single CSV file."""
    # Initialize an empty list to store the dataframes
    # Iterate through all CSV files in the input directory
    sample_id_list = []
    pred_mos_list = []
    for csv_file in Path(input_dir).glob("*.csv"):
        # Read the CSV file into a dataframe
        data_df = pd.read_csv(csv_file)

        tmp_sample_id_list = data_df["sample_id"].tolist()
        tmp_pred_mos_list = data_df["pred_mos"].tolist()

        sample_id_list.extend(tmp_sample_id_list)
        pred_mos_list.extend(tmp_pred_mos_list)

    with Path(output_txt).open("w") as f:
        for sample_id, pred_mos in zip(sample_id_list, pred_mos_list, strict=False):
            f.write(f"{sample_id},{pred_mos}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gather inference results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the CSV files")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to the output text file")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_txt = args.output_txt
    # Run the gather_inference function
    gather_inference(input_dir, output_txt)
