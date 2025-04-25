import pandas as pd


def create_mean_mos_csv(
    incsv: str,
    outcsv: str,
) -> None:
    """Create a CSV file with mean MOS values for each audio file.

    Args:
    ----
        incsv (str): Path to the input CSV file containing audio file paths and MOS values.
        outcsv (str): Path to the output CSV file to save the mean MOS values.

    """
    # Read the input CSV file
    datadf = pd.read_csv(incsv)

    # Group by 'audio' and calculate mean MOS
    df_grouped = datadf.groupby("wav")["score"].mean().reset_index()

    # Save the result to the output CSV file
    df_grouped.to_csv(outcsv, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a CSV file with mean MOS values for each audio file.")
    parser.add_argument("--incsv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--outcsv", type=str, required=True, help="Path to the output CSV file.")

    args = parser.parse_args()

    create_mean_mos_csv(args.incsv, args.outcsv)
