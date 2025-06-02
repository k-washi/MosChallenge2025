from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def gather_inference(
    input_dir_list: str,
    output_txt: str,
) -> None:
    """Gather inference results from multiple CSV files into a single CSV file."""
    # Initialize an empty list to store the dataframes
    # Iterate through all CSV files in the input directory
    sample_id_dict = {}
    for input_dir in input_dir_list:
        for csv_file in Path(input_dir).glob("*.csv"):
            # Read the CSV file into a dataframe
            data_df = pd.read_csv(csv_file)

            tmp_sample_id_list = data_df["sample_id"].tolist()
            tmp_pred_mos_list = data_df["pred_mos"].tolist()

            for sample_id, pred_mos in zip(tmp_sample_id_list, tmp_pred_mos_list, strict=False):
                if sample_id not in sample_id_dict:
                    sample_id_dict[sample_id] = []
                sample_id_dict[sample_id].append(pred_mos)
    var_list = []
    with Path(output_txt).open("w") as f:
        for sample_id, pred_mos_list in sample_id_dict.items():
            pred_mos = sum(pred_mos_list) / len(pred_mos_list)
            var = sum((x - pred_mos) ** 2 for x in pred_mos_list) / len(pred_mos_list)
            var_list.append(var)
            f.write(f"{sample_id},{pred_mos}\n")

    plt.figure(figsize=(10, 6))
    plt.hist(var_list, bins=50, color="blue", alpha=0.7)
    plt.title("Variance of Predicted MOS")
    plt.xlabel("Variance")
    plt.ylabel("Frequency")
    plt.grid(visible=True)
    plt.savefig(Path(output_txt).parent / f"{Path(output_txt).stem}_variance_histogram.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gather inference results")
    parser.add_argument(
        "--input_dir_list", type=str, nargs="+", required=True, help="List of directories containing the CSV files"
    )
    parser.add_argument("--output_txt", type=str, required=True, help="Path to the output text file")
    args = parser.parse_args()
    input_dir_list = args.input_dir_list
    output_txt = args.output_txt
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    # Run the gather_inference function
    gather_inference(input_dir_list, output_txt)
