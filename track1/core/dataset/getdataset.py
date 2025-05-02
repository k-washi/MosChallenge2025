from pathlib import Path

import pandas as pd

WAV_DIR = "wav"
PROMPT_INFO_TXT = "prompt_info.txt"
DEMO_PROMPT_INFO_TEXT = "demo_prompt_info.txt"


def get_dataset(
    dataset_dir: str, mos_text: str = "data/MusicEval-phase1/person_mos/train_person_mos.txt"
) -> tuple[
    list[str | Path],
    list[str],
    list[tuple[float, float]],
]:
    """Get dataset from the specified directory.

    Args:
    ----
        dataset_dir (str): Directory containing audio files.
        mos_text (str): Path to the text file containing Mean Opinion Scores (MOS).

    Returns:
    -------
        tuple: A tuple containing three lists:
            - List of audio file paths.
            - List of text prompts corresponding to the audio files.
            - List of tuples containing scores for the audio files.

    """
    audio_list = []
    prompt_list = []
    score_list = []

    mos_df = pd.read_csv(mos_text, sep=",", header=None).dropna()
    mos_list = [(row[0], row[1], row[2]) for row in mos_df.to_numpy()]

    prompt_text_fp = Path(dataset_dir) / PROMPT_INFO_TXT
    demo_prompt_text_fp = Path(dataset_dir) / DEMO_PROMPT_INFO_TEXT
    prompt_df = pd.read_csv(prompt_text_fp, sep="\t", header=None).dropna()
    demo_prompt_df = pd.read_csv(demo_prompt_text_fp, sep="\t", header=None).dropna()
    demo_prompt_dict = {row[0]: row[1] for row in demo_prompt_df.to_numpy()}
    prompt_dict = {row[0]: row[1] for row in prompt_df.to_numpy() if row[0].startswith("P")}

    for audio_name, mos_mi, mos_ta in mos_list:
        # audio file path
        audio_fp = Path(dataset_dir) / WAV_DIR / audio_name
        if not audio_fp.exists():
            emsg = f"Audio file {audio_fp} does not exist."
            raise FileNotFoundError(emsg)

        # get prompt
        if audio_name in demo_prompt_dict:
            prompt = demo_prompt_dict[audio_name]
        elif Path(audio_name).stem.split("_")[-1] in prompt_dict:
            prompt = prompt_dict[Path(audio_name).stem.split("_")[-1]]
        else:
            emsg = f"Audio file {audio_name} not found in prompt dictionary."
            raise KeyError(emsg)

        mos_score = (mos_mi, mos_ta)

        audio_list.append(audio_fp)
        prompt_list.append(prompt)
        score_list.append(mos_score)

    return audio_list, prompt_list, score_list
