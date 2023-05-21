import core, myutils
import glob as glob


if __name__ == "__main__":

    audio_files = glob.glob("../data/*.wav")
    print(audio_files)
    audio = audio_files[0]
    x, sr = myutils.loadAudio(audio)
    core.gen_sc(audio)
    core.gen_waveplot(audio)
    core.gen_mfcc(audio)
    core.gen_mfccs(audio)
    core.gen_spec1(audio)
    core.gen_spec2(audio)
    core.gen_specrf(audio)
    core.gen_chrom(audio)
