from pathlib import Path
import shutil
import pandas as pd


OUTPUT_WAV_DIRNAME = "wav"
USER_DICT_DEF = {}

def to_user_name(user_id:int) -> str:
    """
    Convert user ID to user name.
    
    Args:
        user_id (int): User ID.
        
    Returns:
        str: User name.
    """
    return f"bvccmain{user_id:05d}"

def format_process(
    info_csv:str, 
    wav_dir:str, 
    output_dir:str, 
    output_csv_name:str="train.csv",
    user_dict:dict=USER_DICT_DEF,
) ->dict:
    """
    Process the BVCC dataset and save the formatted data to the output directory.
    
    Args:
        info_csv (str): Path to the CSV file containing information about the dataset.
        wav_dir (str): Path to the directory containing WAV files.
        output_dir (str): Path to the output directory.
        output_csv_name (str): Name of the output CSV file.
        user_dict (dict): Dictionary to map user IDs to user names.
    """
    # Read the CSV file
    data_df = pd.read_csv(info_csv)
    data_list = data_df.to_numpy().tolist()
    data_list = [(item[1], item[2], item[-1]) for item in data_list]
    print(f"Total {len(data_list)} samples in {info_csv}")
    
    user_count = -1
    for v in user_dict.values():
        if v > user_count:
            user_count = v
    
    for _, _, user_id in data_list:
        if user_id not in user_dict:
            user_count += 1
            user_dict[user_id] = user_count
    print(f"Total {user_count} users")
    
    output_list = []
    
    for wav_name, mos_score, user_id in data_list:
        map_user_id = to_user_name(user_dict[user_id])
        wav_path = Path(wav_dir) / wav_name
        if not wav_path.exists():
            print(f"File {wav_path} does not exist")
            continue
        output_list.append((wav_name, mos_score, map_user_id))
        output_wav_fp = Path(output_dir) / OUTPUT_WAV_DIRNAME / wav_name
        output_wav_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(wav_path, output_wav_fp)
    print(f"Total {len(output_list)} samples")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Write the output CSV file
    output_csv_path = Path(output_dir) / output_csv_name
    output_df = pd.DataFrame(output_list, columns=["wav", "score", "user_id"])
    output_df.to_csv(output_csv_path, index=False)
    
    return user_dict
    
    
def bvcc_format_dataset(dataset_dir, output_dir):
    """
    Format the BVCC dataset for training and validation.
    
    Args:
        dataset_dir (str): Path to the BVCC dataset directory.
        output_dir (str): Path to the output directory.
    """
    data_dir = Path(dataset_dir) / "DATA"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    user_map = format_process(
        info_csv=str(data_dir / "sets" / "TRAINSET"),
        wav_dir=str(data_dir / "wav"),
        output_dir=str(output_dir),
        output_csv_name="train.csv"
    )
    
    user_map = format_process(
        info_csv=str(data_dir / "sets" / "DEVSET"),
        wav_dir=str(data_dir / "wav"),
        output_dir=str(output_dir),
        output_csv_name="val.csv",
        user_dict=user_map
    )
    
    user_map = format_process(
        info_csv=str(data_dir / "sets" / "TESTSET"),
        wav_dir=str(data_dir / "wav"),
        output_dir=str(output_dir),
        output_csv_name="test.csv",
        user_dict=user_map
    )
    
    user_list = sorted(user_map.items(), key=lambda x: x[1])
    user_list = [(to_user_name(v), k) for k, v in user_list]
    output_user_df = pd.DataFrame(user_list, columns=["user_id", "user_name"])
    output_user_df.to_csv(Path(output_dir) / "user.csv", index=False)
    print(f"Total {len(user_list)} users")
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Format BVCC dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the BVCC dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    
    args = parser.parse_args()
    bvcc_format_dataset(args.dataset_dir, args.output_dir)