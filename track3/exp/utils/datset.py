from pathlib import Path

import pandas as pd


def get_user_map(
    user_csv_fp: str | Path,
) -> dict[str, int]:
    """Get the user map from the user CSV file.

    Args:
    ----
        user_csv_fp (str | Path): Path to the user CSV file.

    Returns:
    -------
        dict: A dictionary mapping user names to user IDs.

    """
    data_df = pd.read_csv(user_csv_fp)
    user_map = {user: idx for idx, user in enumerate(data_df["user_id"].tolist())}
    print(f"Load {len(user_map)} users from {user_csv_fp}")
    return user_map


def get_dataset_fp_label_userid(
    dataset_list: list[str],
    user_map: dict[str, int],
) -> tuple[list[str | Path], list[int], list[int]]:
    """Get the file paths, labels, and user IDs from the dataset.

    Args:
    ----
        dataset_list (list[str]): List of dataset csv file.
        user_map (dict[str, int]): Dictionary mapping user names to user IDs.

    Returns:
    -------
        tuple: A tuple containing lists of audio file paths, labels, and user IDs.

    """
    audio_list = []
    label_list = []
    user_id_list = []

    for csv_fp in dataset_list:
        data_df = pd.read_csv(csv_fp)
        wav_dir = Path(csv_fp).parent / "wav"
        tmp_audio_list = data_df["wav"].tolist()
        tmp_audio_list = [wav_dir / Path(audio) for audio in tmp_audio_list]
        tmp_label_list = data_df["score"].tolist()
        tmp_user_list = data_df["user_id"].tolist()
        tmp_user_list = [user_map[user] for user in tmp_user_list]
        audio_list.extend(tmp_audio_list)
        label_list.extend(tmp_label_list)
        user_id_list.extend(tmp_user_list)
    assert (
        len(audio_list) == len(label_list) == len(user_id_list)
    ), "Length of audio_list, label_list and user_id_list must be the same."
    for audio_fp in audio_list:
        assert audio_fp.exists(), f"Audio file {audio_fp} does not exist."

    return audio_list, label_list, user_id_list
