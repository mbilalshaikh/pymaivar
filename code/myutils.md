# Documentation for the myutils.py

This script provides a utility function, `loadAudio`, to load an audio file using the `librosa` library. The `librosa` library is a Python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

The `loadAudio` function loads an audio file from the specified file path and returns the audio time series and the sample rate.

## Dependencies

- **numpy**: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
  
- **librosa**: A Python library for audio and music analysis. It provides the building blocks necessary to create music information retrieval systems.

## Function

- `loadAudio(filePath, sampleRate=22050)`: This function loads an audio file from the given `filePath` at the specified `sampleRate` (default 22050Hz) using the `librosa` function `librosa.load()`. It returns a numpy array `x` containing the audio samples and an integer `sr` which is the actual sample rate used.
