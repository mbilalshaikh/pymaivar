# PyMAiVAR Documentation

This code contains functions for audio feature visualization using the Librosa library. Librosa is a Python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

## Dependencies
- Numpy
- Matplotlib
- Librosa
- Sklearn

## Functions

**1. `get_specfilename(name, folder)`**
   - **Parameters:**
     - `name` (str): The name of an audio file.
     - `folder` (str): The folder where the generated images will be saved. 
   - **Return:** This function formats the `name` argument, concatenates it with `folder`, and appends a ".png" extension to create a path where the result image will be saved.

**2. `normalize(x, axis=0)`**
   - **Parameters:**
     - `x` (1-D array): The array to be normalized.
     - `axis` (int, optional): The axis along which to normalize the array. Default is 0.
   - **Return:** This function returns a normalized version of `x` along the specified axis using sklearn's `minmax_scale()` function. 

**3. `gen_sc(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates a Spectral Centroid visualization. It computes its spectral centroid and plots it on a waveform. The plot is then saved as a PNG file.

**4. `gen_mfcc(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates Mel-frequency cepstral coefficients (MFCCs) visualization. It computes its MFCCs and plots them. The plot is then saved as a PNG file.

**5. `gen_waveplot(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates a waveplot for an audio file. It computes its spectral centroid and plots a waveplot. The waveplot is then saved as a PNG file.

**6. `gen_spec1(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates a spectrogram of an audio file using the linear frequency scale. It computes its spectrogram and plots it. The plot is then saved as a PNG file.

**7. `gen_spec2(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates a spectrogram of an audio file using the logarithmic frequency scale. It computes its spectrogram and plots it. The plot is then saved as a PNG file.

**8. `gen_specrf(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates a visualization of the spectral rolloff. It computes its spectral rolloff and plots it on a waveform. The plot is then saved as a PNG file.

**9. `gen_mfccs(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates a visualization of the scaled MFCCs. It computes its MFCCs, scales them, and plots them. The plot is then saved as a PNG file.

**10. `gen_chrom(audio)`**
   - **Parameters:**
     - `audio` (str): The path to the audio file.
   - **Return:** This function generates a Chromagram. It computes its chromagram and plots it. The plot is then saved as a PNG file.

## Usage
Each function accepts a string representing the path to an audio file and generates a visualization of a particular feature of that audio. The visualizations are saved as PNG files in the specified directory.

## Remarks
The functions load audio files and compute their features, as well as Matplotlib to generate plots. They're designed to be easy to use for both individual audio files and large sets of files in an automated workflow.

Be aware that audio processing can be computationally intensive, so these functions may take some time to run on longer audio files or large batches of files. Always ensure that you have adequate computational resources.
