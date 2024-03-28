# here write feature extraction funcs
import librosa as lb


def _featureExtractor(audioFile):
    sampleRate = lb.get_samplerate(y=audioFile)
    mel_spectrogram = lb.feature.melspectrogram(y=audioFile, sr=sampleRate)
    featureArr = [audioFile, sampleRate, mel_spectrogram]
    return featureArr
