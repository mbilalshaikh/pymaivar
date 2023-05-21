import numpy as np
import librosa

# Utility functions


def loadAudio(filePath, sampleRate=22050):
    """Load audio sample

    Args:
        filePath (str): Path of the audio file
        sampleRate (int): Sampling rate
    Returns:
        x (np.ndarray): Audio sample values sampled at sampleRate
        sr (int): Sampling rate
    """
    x, sr = librosa.load(path=filePath, sr=sampleRate)
    # x = essentia.standard.MonoLoader(filename=filePath, sampleRate=sampleRate)()
    return x, sr
