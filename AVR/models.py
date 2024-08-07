# Import necessary libraries
import torch
import librosa as lb
import numpy as np


# Utility function to compute mel spectrogram
def get_spectrogram(wav_file, sample_rate):
    mel_spectrogram = lb.feature.melspectrogram(y=wav_file, sr=sample_rate)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    return np.expand_dims(mel_spectrogram_db, axis=0)

# Function to process the audio file and classify it using pre-trained models
def query_function(file_path):
    # Load the audio file
    wav_file, sample_rate = lb.load(file_path)

    length = 96333  # Average length of audio sample

    # Adjust the audio file length
    wav_file = adjust_length(wav_file, length)

    # Compute mel spectrogram
    mel = get_spectrogram(wav_file, sample_rate)

    # Prepare tensors for the models
    wav_and_samp = np.concatenate([wav_file, [sample_rate]])
    wav_and_samp_t = torch.tensor(wav_and_samp, dtype=torch.float32)
    mel_t = torch.tensor(mel, dtype=torch.float32)

    # Load the pre-trained models and map them to CPU
    voice_model = torch.load("AVR/Models/voice_model.pth", map_location=torch.device('cpu'))
    specto_model = torch.load("AVR/Models/specto_model.pth", map_location=torch.device('cpu'))
    ensemble_model = torch.load("AVR/Models/ensemble_model.pth", map_location=torch.device('cpu'))

    # Set models to evaluation mode
    voice_model.eval()
    specto_model.eval()
    ensemble_model.eval()

    # Run the mel spectrogram through the spectrogram model
    specto_input = mel_t.unsqueeze(0)
    with torch.no_grad():
        specto_output = specto_model(specto_input)
    specto_probs = torch.exp(specto_output)

    # Run the waveform and sample rate through the voice model
    voice_input = wav_and_samp_t.unsqueeze(0)
    with torch.no_grad():
        voice_output = voice_model(voice_input)
    voice_probs = torch.exp(voice_output)

    # Extract spoof and not-spoof probabilities from the models
    specto_probs_spoof = specto_probs[0,1]
    Spectro_Prob_Not_Spoof = specto_probs[0,0]
    Voice_Prob_Spoof = voice_probs[0,1]
    Voice_Prob_Not_Spoof = voice_probs[0,0]

    # Combine the probabilities for the ensemble model
    ensemble_inputs = torch.tensor([specto_probs_spoof, Spectro_Prob_Not_Spoof, Voice_Prob_Spoof, Voice_Prob_Not_Spoof], dtype=torch.float32)

    # Run the combined inputs through the ensemble model
    ensemble_input = ensemble_inputs.clone().detach().unsqueeze(0).float()
    with torch.no_grad():
        ensemble_output = ensemble_model(ensemble_input)

    # Determine the final classification based on ensemble model output
    if ensemble_output <= 0.5:
        ensemble_prediction = 0
    else:
        ensemble_prediction = 1

    # Return the final classification result
    result = False if ensemble_prediction == 0 else True

    return result


# Function to adjust the length of the audio file
def adjust_length(wavFile, length):
    if len(wavFile) < length:
        # Pad with zeros if the length is less than the maximum length
        wavFile = np.pad(wavFile, (0, length - len(wavFile)), 'constant')
    elif len(wavFile) > length:
        # Truncate the audio file if the length is greater than the maximum length
        wavFile = wavFile[:length]
    return wavFile
