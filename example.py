import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa.display
import sklearn
import glob as glob

def get_specfilename(name,folder):
    name = name.split('.')[2]
    print(name)
    name =  name.split('/')[1].split('_')
    name = folder+'_'.join(name)+'.png'
    return name

def gen_sc():
    
    audio_files = glob.glob('../data/*.wav')
    
    audio = audio_files[0]
    x , sr = librosa.load(audio,sr=None)
    plt.figure(figsize=(14, 5))
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    spectral_centroids.shape
    (775,)
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # Normalising the spectral centroid for visualisation
    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)
    #Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.axis('off')
    plt.plot(t, normalize(spectral_centroids), color='r')
    print(audio)
    out = get_specfilename(audio,'../results/sc-')
    #out = 'a4/a4.png'
    plt.savefig(out)
    plt.figure(figsize=(14, 5))
    mfccs = librosa.feature.mfcc(x, sr=sr)
    #print(mfccs.shape)
    #Displaying  the MFCCs:
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.axis('off')
    out = get_specfilename(audio,'../results/mfcc-')
    #out = 'a6/a6.png'
    plt.savefig(out)

if __name__ == "__main__":
    gen_sc()