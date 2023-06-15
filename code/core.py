"""This code contains functions for audio feature visualization using the Librosa library. Librosa is a Python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa.display
import sklearn


def get_specfilename(name, folder):
    """Generate the name of the file that will hold the plotted spectral feature."""
    name = name.split(".")[2]
    name = name.split("/")[1].split("_")
    name = folder + "_".join(name) + ".png"
    return name


def normalize(x, axis=0):
    """Normalize the spectral feature data for visualization."""
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def gen_sc(audio):
    """Compute and plot spectral centroid for an audio file."""
    x, sr = librosa.load(audio, sr=None)
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color="r")
    plt.axis("off")

    out = get_specfilename(audio, "../results/sc-")
    plt.savefig(out)


def gen_mfcc(audio):
    """Compute and plot MFCCs for an audio file."""
    x, sr = librosa.load(audio, sr=None)
    mfccs = librosa.feature.mfcc(x, sr=sr)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.axis("off")

    out = get_specfilename(audio, "../results/mfcc-")
    plt.savefig(out)


def gen_waveplot(audio):
    """Generate waveplot for an audio file."""
    x, sr = librosa.load(audio, sr=None)

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.axis("off")

    out = get_specfilename(audio, "../results/wp-")
    plt.savefig(out)


def gen_spec1(audio):
    """Compute and plot the Short-Time Fourier Transform (STFT) of an audio file."""
    x, sr = librosa.load(audio, sr=None)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
    plt.axis("off")

    out = get_specfilename(audio, "../results/specshow1-")
    plt.savefig(out)


def gen_spec2(audio):
    """Compute and plot the Short-Time Fourier Transform (STFT) of an audio file with log-scale y-axis."""
    x, sr = librosa.load(audio, sr=None)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")
    plt.axis("off")

    out = get_specfilename(audio, "../results/specshow2-")
    plt.savefig(out)


def gen_specrf(audio):
    """Compute and plot the spectral roll-off for an audio file."""
    x, sr = librosa.load(audio, sr=None)
    spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
    frames = range(len(spectral_rolloff))
    t = librosa.frames_to_time(frames)

    plt.figure(figsize=(10, 5))
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_rolloff), color="r")
    plt.axis("off")

    out = get_specfilename(audio, "../results/specrolloff-")
    plt.savefig(out)


def gen_mfccs(audio):
    """Compute and plot the MFCCs for an audio file with feature scaling."""
    x, sr = librosa.load(audio, sr=None)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.axis("off")

    out = get_specfilename(audio, "../results/mfccs-")
    plt.savefig(out)


def gen_chrom(audio):
    """Compute and plot the chroma feature for an audio file."""
    x, sr = librosa.load(audio, sr=None)
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(
        chromagram,
        x_axis="time",
        y_axis="chroma",
        hop_length=hop_length,
        cmap="coolwarm",
    )
    plt.axis("off")

    out = get_specfilename(audio, "../results/chrom-")
    plt.savefig(out)
    print("-")
