#imports
import os
import pandas as pd
import soundfile as sf
from soundfile import SoundFile
import numpy as np
import scipy
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import librosa as lb
import librosa.display as lbd
from sklearn.model_selection import train_test_split
import setuptools
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from IPython.display import clear_output
import random
from sklearn.metrics import f1_score
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

#needed funcs
def getSpecto(wavFile,sampleRate):
    # Compute mel spectrogram
    mel_spectrogram = lb.feature.melspectrogram(y=wavFile, sr=sampleRate)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    #add channel num
    melTemp = np.ones((1, mel_spectrogram_db.shape[0], mel_spectrogram_db.shape[1]), dtype=mel_spectrogram_db.dtype)
    return melTemp


class voiceDB:
    def __init__(self, dataFrame):
        self.data = dataFrame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tempPath = self.data.iloc[idx]['Path']
        voicePath = path_prefix + tempPath
        label = 1 if self.data.iloc[idx]['Label'] == 'bona-fide' else 0 if self.data.iloc[idx][
                                                                               'Label'] == 'spoof' else None
        wavFile, sampleRate = lb.load(voicePath)

        max_length = 551052  # longest datapoint, 24 seconds

        # Pad or truncate the audio file to ensure consistent length
        if len(wavFile) < max_length:
            # Pad with zeros if the length is less than the maximum length
            wavFile = np.pad(wavFile, (0, max_length - len(wavFile)), 'constant')

        mel = getSpecto(wavFile, sampleRate)

        if wavFile.dtype != np.float32:
            wavFile = wavFile.astype(np.float32)
        if not isinstance(sampleRate, np.float32):
            sampleRate = np.array([sampleRate], dtype=np.float32)

        # Concatenate wavFile and sampleRate
        wavAndSamp = np.concatenate((wavFile, sampleRate))

        return wavAndSamp, mel, label


# Set your Hugging Face token if needed
token = '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'

# Function to download a file from Hugging Face Hub
def download_model(repo_id, filename, token):
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token)
    return file_path

# Function to load a model from Hugging Face
def load_model(repo_name, token):
    model = AutoModel.from_pretrained(repo_name, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(repo_name, use_auth_token=token)
    return model, tokenizer


#Main.
if __name__ == '__main__':


    # Repositories and filenames
    voice_model_repo = 'gbenari2/voice'
    specto_model_repo = 'gbenari2/specto'
    ensemble_model_repo = 'gbenari2/ensemble'
    voice_model_filename = 'voiceModeltemp.pth'
    specto_model_filename = 'spectoModeltemp.pth'
    #ensemble_model_filename = 'ensembleModeltemp.pth'

    # Download the models
    voice_model_path = download_model(voice_model_repo, voice_model_filename, token)
    specto_model_path = download_model(specto_model_repo, specto_model_filename, token)
    #ensemble_model_path = download_model(ensemble_model_repo, ensemble_model_filename, token)

    # Load the models and map to CPU
    voice_model = torch.load(voice_model_path, map_location=torch.device('cpu'))
    specto_model = torch.load(specto_model_path, map_location=torch.device('cpu'))
    #ensemble_model = torch.load(ensemble_model_path, map_location=torch.device('cpu'))

    # save models to Models folder
    torch.save(voice_model, 'Models/voiceModel.pth')
    torch.save(specto_model, 'Models/spectoModel.pth')
    #torch.save(ensemble_model, 'Models/ensembleModel.pth')


    # Ensure the models are in evaluation mode
    voice_model.eval()
    specto_model.eval()
    #ensemble_model.eval()

    tempHardcode = "C:/Users/Ynon Friedman/Downloads/6-spoof.wav"

    #request user input for path to the audio file
         #audioPath = input("Please enter the path to the audio file: ")


    #load the audio file
    wavFile, sampleRate = lb.load(tempHardcode)

    max_length = 551052  # longest datapoint, 24 seconds

    # Pad or truncate the audio file to ensure consistent length
    if len(wavFile) < max_length:
        # Pad with zeros if the length is less than the maximum length
        wavFile = np.pad(wavFile, (0, max_length - len(wavFile)), 'constant')

    #run into models
    mel = getSpecto(wavFile, sampleRate)
    wavAndSamp = np.concatenate(wavFile, sampleRate)
    wavAndSampT = torch.tensor(wavAndSamp)
    melT = torch.tensor(mel)

    #run into specto
    spectoInput = melT.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        spectoOutput = specto_model(spectoInput)
    print(spectoOutput)
    spectoProbs = torch.exp(spectoOutput)
    spectoProbs = spectoProbs / torch.sum(spectoProbs) * 100
    spectoProbs_list = spectoProbs.cpu().numpy().flatten().tolist()
    formatted_probs = [f"{prob:.2f}" for prob in spectoProbs_list]
    print(formatted_probs)

    # #run into voice
    # voiceInput = wavAndSampT.unsqueeze(0)  # Add batch dimension
    # with torch.no_grad():
    #     voiceOutput = voiceModel(voiceInput)
    # voiceProbs = torch.exp(voiceOutput)
    # voiceProbs = voiceProbs / torch.sum(voiceProbs) * 100
    # voiceProbs_list = voiceProbs.cpu().numpy().flatten().tolist()
    # formatted_voice_probs = [f"{prob:.2f}" for prob in voiceProbs_list]
    # print(formatted_voice_probs)

    # #run into ensemble
    # ensembleInput = torch.tensor([spectoProbs_list, voiceProbs_list], dtype=torch.float32)
    # ensembleInput = ensembleInput.unsqueeze(0)  # Add batch dimension and move to device
    # with torch.no_grad():
    #     ensembleOutput = ensembleModel(ensembleInput)
    # ensembleProbs = torch.exp(ensembleOutput)
    # ensembleProbs = ensembleProbs / torch.sum(ensembleProbs) * 100
    # ensembleProbs_list = ensembleProbs.cpu().numpy().flatten().tolist()
    # formatted_ensemble_probs = [f"{prob:.2f}" for prob in ensembleProbs_list]
    # print(formatted_ensemble_probs)


    #
    # #process output and return
    # if ensembleProbs[1] > ensembleProbs[0]:
    #     print("The audio file is spoof.")
    # else:
    #     print("The audio file is not spoof.")

