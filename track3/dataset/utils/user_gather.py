# user情報をまとめる
from pathlib import Path
import pandas as pd
def user_gather(
    dataset_dir:str,
) -> None:
    """
    Gather user information from the dataset directory.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
    """
    ddir = Path(dataset_dir)
    # Create a dictionary to store user information
    each_dataset_list = sorted([s for s in ddir.glob("*") if s.is_dir()])
    user_list = []
    for dataset_dir_fp in each_dataset_list:
        user_fp = dataset_dir_fp / "user.csv"
        
        assert user_fp.exists(), f"User file {user_fp} does not exist"
        tmp_user_list = pd.read_csv(user_fp).to_numpy().tolist()
        user_list.extend(tmp_user_list)
    user_list = sorted(user_list, key=lambda x: x[0])
    outptu_fp = ddir / "user.csv"
    print(f"Output user file: {outptu_fp}")
    pd.DataFrame(user_list, columns=["user_id", "user_name"]).to_csv(outptu_fp, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gather user information from the dataset directory.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    args = parser.parse_args()
    
    user_gather(args.dataset_dir)