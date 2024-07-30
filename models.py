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

    max_length = 551052  # longest datapoint, 24 seconds

    # Pad or truncate the audio file to ensure consistent length
    if (wav_length := len(wav_file)) < max_length:
        wav_file = np.pad(wav_file, (0, max_length - wav_length), 'constant')
    elif wav_length > max_length:
        wav_file = wav_file[:max_length]

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

    #specto precentage calculation
    specto_precentage = specto_probs / torch.sum(specto_probs) * 100
    specto_precentage_list = specto_precentage.cpu().numpy().flatten().tolist()

    # Run into voice model
    voice_input = wav_and_samp_t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        voice_output = voice_model(voice_input)
    voice_probs = torch.exp(voice_output)

    #voice precentage calculation
    voice_precentage = voice_probs / torch.sum(voice_probs) * 100
    voice_precentage_list = voice_precentage.cpu().numpy().flatten().tolist()

    specto_probs_spoof = specto_probs[0,1]
    Spectro_Prob_Not_Spoof = specto_probs[0,0]
    Voice_Prob_Spoof = voice_probs[0,1]
    Voice_Prob_Not_Spoof = voice_probs[0,0]

    # Combine the probabilities for ensemble model
    ensemble_inputs = torch.tensor([specto_probs_spoof, Spectro_Prob_Not_Spoof, Voice_Prob_Spoof, Voice_Prob_Not_Spoof], dtype=torch.float32)


    # Run into ensemble model
    ensemble_input = torch.tensor(ensemble_inputs, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        ensemble_output = ensemble_model(ensemble_input)

    print(f"Ensemble output: {ensemble_output}")

    ensemble_probs = torch.exp(ensemble_output)
    ensemble_probs = ensemble_probs / torch.sum(ensemble_probs) * 100
    ensemble_probs_list = ensemble_probs.cpu().numpy().flatten().tolist()

    spoof_probability = ensemble_probs_list[0]
    result = "The audio file is spoof." if spoof_probability > 50 else "The audio file is bona-fide."

    # return result #to change
    return result



