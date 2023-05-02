import torch
import glob as glob
import subprocess
#import librosa
import matplotlib.pyplot as plt
#import librosa.display


def get_wavfilename(name):
    name = name.split('.')[0].split('/')[1]    
    #name =  name.split('/')[1].split('_')[1:]
    #name = 'wav/'+'_'.join(name)+'.wav'
    name = 'wav/'+name+'.wav'
    return name

def gen_avitowav(src,des):
    #!ffmpeg -i $src -ab 160k -ac 2 -ar 44100 -vn $des -nostats -loglevel 0
    pass
        

def convert_avi_to_wav(videos,audios):
    # take ucf data
    ucf101 = (glob.glob(videos+"/*"))
    ucf101.sort()

    i =0 
    t = 0
    for avi in ucf101:
        i+=1
        if i%50==0:
            print(i)
            print(avi)
        #generate wav file
        wav = get_wavfilename(avi)   

        #!ffmpeg -i $vid -ab 160k -ac 2 -ar 44100 -vn $name -nostats -loglevel 0
        gen_avitowav(avi,wav)
        wav_dir = (glob.glob("wav/*"))
        size = len(wav_dir) 
        if size == t:
            print(size,avi,wav)
        t = size

print('data preprocessing')
print(len(glob.glob('wav/*')))

