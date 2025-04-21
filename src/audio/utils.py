import torch
import torchaudio

def load_wave(wave_file_path:str, sample_rate:int, is_torch:bool=True, mono:bool=False):
    """
    Load a wave file from the given file path.

    Args:
        wave_file_path (str): The path to the wave file.
        sample_rate (int, optional): The desired sample rate of the loaded wave file. 
        is_torch (bool, optional): Whether to return the wave as a torch tensor. 
            If set to False, the wave will be converted to a numpy array. 
            Defaults to True.
        mono (bool, optional): Whether to convert the wave to mono. 
            If set to True, only the first channel will be returned. 
            Defaults to False.

    Returns:
        tuple: A tuple containing the loaded wave and the sample rate.
    """
    
    wave, sr = torchaudio.load(wave_file_path)
    if mono:
        wave = wave[0]
    if sample_rate > 0 and sample_rate != sr:
        wave = torchaudio.transforms.Resample(sr, sample_rate)(wave)
    else:
        sample_rate=sr
    if not is_torch:
        wave = wave.cpu().detach().numpy().copy()
    return wave, sample_rate

def save_wave(wave, output_path, sample_rate:int=16000):
    """
    Save a waveform as a WAV file.

    Args:
        wave (numpy.ndarray or torch.Tensor): The waveform to be saved.
        output_path (str): The path to save the WAV file.
        sample_rate (int, optional): The sample rate of the waveform. Defaults to 16000.
    """
    if not isinstance(wave, torch.Tensor):
        wave = torch.from_numpy(wave)

    if wave.dim() == 1: wave = wave.unsqueeze(0)
    torchaudio.save(uri=str(output_path), src=wave.to(torch.float32), sample_rate=sample_rate)