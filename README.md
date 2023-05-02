# pymaivar
Repository for pymaivar software suit.

PyMAiVAR (Python Multimodal Audio-Image and Video Action Recognizer) is a Python package for multimodal action recognition. The package provides a CNN-based audio-image to video fusion model that leverages the representation process of Convolutional Neural Networks (CNNs) for incorporating image-based audio representations of actions in a task.

Installation:

PyMAiVAR can be installed using pip:

Copy code
pip install pymaivar
Dependencies:

PyMAiVAR depends on the following packages:

numpy
pandas
librosa
opencv-python
tensorflow
Usage:

To use PyMAiVAR, first import the package and instantiate the MAiVAR class:

python
Copy code
import pymaivar

maivar = pymaivar.MAiVAR()
Then, load your video and audio files using load_video and load_audio functions, respectively:

python
Copy code
maivar.load_video("path/to/video.mp4")
maivar.load_audio("path/to/audio.wav")
Next, extract features from the video and audio data using extract_video_features and extract_audio_features functions, respectively:

python
Copy code
maivar.extract_video_features()
maivar.extract_audio_features()
Finally, fuse the features using the fuse_features function and predict the action using the predict function:

python
Copy code
maivar.fuse_features()
prediction = maivar.predict()
Contribution:

PyMAiVAR is an open-source package and contributions are always welcome. If you have any suggestions or want to report a bug, please feel free to open an issue on the GitHub repository.

License:

PyMAiVAR is distributed under the MIT license. See the LICENSE file for more details.
