import torch
import librosa as lb
import numpy as np


# Utility function to compute mel spectrogram
def get_spectrogram(wav_file, sample_rate):
    mel_spectrogram = lb.feature.melspectrogram(y=wav_file, sr=sample_rate)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    return np.expand_dims(mel_spectrogram_db, axis=0)


def query_function(file_path):
    """
    Process the audio file and classify it using pre-trained models.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        str: Classification result.
    """

    wav_file, sample_rate = lb.load(file_path)

    length2 = 551052

    if len(wav_file) < length2:
        # Pad with zeros if the length is less than the maximum length
        wav_file = np.pad(wav_file, (0, length2 - len(wav_file)), 'constant')
    elif len(wav_file) > length2:
        # Truncate the audio file if the length is greater than the maximum length
        wav_file = wav_file[:length2]


    #length = 96333  # average length

    #wav_file = adjust_length(wav_file, length)

    # Compute mel spectrogram
    mel = get_spectrogram(wav_file, sample_rate)

    # Prepare tensors
    wav_and_samp = np.concatenate([wav_file, [sample_rate]])
    wav_and_samp_t = torch.tensor(wav_and_samp, dtype=torch.float32)  # Convert to float32
    mel_t = torch.tensor(mel, dtype=torch.float32)  # Convert to float32

    # Load the models and map to CPU
    voice_model = torch.load("Models/voice_model.pth", map_location=torch.device('cpu'))
    specto_model = torch.load("Models/specto_model.pth", map_location=torch.device('cpu'))
    ensemble_model = torch.load("Models/ensemble_model.pth", map_location=torch.device('cpu'))

    # Set models to evaluation mode
    voice_model.eval()
    specto_model.eval()
    ensemble_model.eval()

    # Run into specto model
    specto_input = mel_t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        specto_output = specto_model(specto_input)
    specto_probs = torch.exp(specto_output)

    # Specto percentage calculation
    specto_percentage = specto_probs / torch.sum(specto_probs) * 100
    specto_percentage_list = specto_percentage.cpu().numpy().flatten().tolist()

    # Run into voice model
    voice_input = wav_and_samp_t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        voice_output = voice_model(voice_input)
    voice_probs = torch.exp(voice_output)

    # Voice percentage calculation
    voice_percentage = voice_probs / torch.sum(voice_probs) * 100
    voice_percentage_list = voice_percentage.cpu().numpy().flatten().tolist()

    specto_probs_spoof = specto_probs[0,1]
    Spectro_Prob_Not_Spoof = specto_probs[0,0]
    Voice_Prob_Spoof = voice_probs[0,1]
    Voice_Prob_Not_Spoof = voice_probs[0,0]

    # Combine the probabilities for ensemble model
    ensemble_inputs = torch.tensor([specto_probs_spoof, Spectro_Prob_Not_Spoof, Voice_Prob_Spoof, Voice_Prob_Not_Spoof], dtype=torch.float32)

    # Run into ensemble model
    ensemble_input = ensemble_inputs.clone().detach().unsqueeze(0).float()   # Add batch dimension
    with torch.no_grad():
        ensemble_output = ensemble_model(ensemble_input)

    if ensemble_output <= 0.5:
        ensemble_prediction = 0
    else:
        ensemble_prediction = 1

    result = "The audio file is spoof." if ensemble_prediction == 0 else "The audio file is bona-fide."

    # Prepare voice_percentage_list and specto_percentage_list for the result, maybe show them in the UI? probabilities
    return result


def adjust_length(wavFile, length):
    if len(wavFile) < length:
        # Pad with zeros if the length is less than the maximum length
        wavFile = np.pad(wavFile, (0, length - len(wavFile)), 'constant')
    elif len(wavFile) > length:
        # Truncate the audio file if the length is greater than the maximum length
        wavFile = wavFile[:length]
    return wavFile

