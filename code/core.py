import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa.display
import sklearn

# Function to generate the name of the file that will hold the plotted spectral feature.
def get_specfilename(name, folder):
    name = name.split(".")[2]
    name = name.split("/")[1].split("_")
    name = folder + "_".join(name) + ".png"
    return name

# Function to normalize the spectral feature data for visualization.
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Function to compute and plot spectral centroid for an audio file.
def gen_sc(audio):
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

# Function to compute and plot MFCCs for an audio file.
def gen_mfcc(audio):
    x, sr = librosa.load(audio, sr=None)
    mfccs = librosa.feature.mfcc(x, sr=sr)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.axis("off")
    
    out = get_specfilename(audio, "../results/mfcc-")
    plt.savefig(out)

# Function to generate waveplot for an audio file.
def gen_waveplot(audio):
    x, sr = librosa.load(audio, sr=None)

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.axis("off")
    
    out = get_specfilename(audio, "../results/wp-")
    plt.savefig(out)

# Function to compute and plot the Short-Time Fourier Transform (STFT) of an audio file.
def gen_spec1(audio):
    x, sr = librosa.load(audio, sr=None)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
    plt.axis("off")
    
    out = get_specfilename(audio, "../results/specshow1-")
    plt.savefig(out)

# Similar to the previous function, but the y-axis is in log scale.
def gen_spec2(audio):
    x, sr = librosa.load(audio, sr=None)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")
    plt.axis("off")
    
    out = get_specfilename(audio, "../results/specshow2-")
    plt.savefig(out)

# Function to compute and plot the spectral roll-off for an audio file.
def gen_specrf(audio):
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

# Function to compute and plot the MFCCs for an audio file with feature scaling.
def gen_mfccs(audio):
    x, sr = librosa.load(audio, sr=None)
    mfccs = librosa.feature.mfcc(x, sr=sr)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time")
    plt.axis("off")
    
    out = get_specfilename(audio, "../results/mfccs-")
    plt.savefig(out)

# Function to compute and plot the chroma feature for an audio file.
def gen_chrom(audio):
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
    print('-')
