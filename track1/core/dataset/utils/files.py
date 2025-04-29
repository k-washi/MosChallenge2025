from pathlib import Path


def get_audio_name(
    audio_fp: str,
) -> str:
    """Extract the audio name from the file path.

    audiomos2025-track1-S002_P044.wav => P044
    Args:
        audio_fp (str): File path of the audio file.

    Returns
    -------
        str: Audio name without extension.

    """
    return Path(audio_fp).stem.split("_")[-1]
