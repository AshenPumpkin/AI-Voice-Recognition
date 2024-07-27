# First, open terminal and run: pip install -r requirements.txt

# Imports
import os
import sys
import subprocess
import numpy as np
import librosa as lb
import torch
from huggingface_hub import hf_hub_download
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox


# Utility function to compute mel spectrogram
def get_spectrogram(wav_file, sample_rate):
    mel_spectrogram = lb.feature.melspectrogram(y=wav_file, sr=sample_rate)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    return np.expand_dims(mel_spectrogram_db, axis=0)


class AudioClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Audio Classifier')
        layout = QVBoxLayout()

        open_button = QPushButton('Open Audio File', self)
        open_button.clicked.connect(self.show_dialog)
        layout.addWidget(open_button)

        self.setLayout(layout)
        self.show()

    def show_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.mp3 *.wav)",
            options=options
        )
        if file_path:
            print(f"Selected file: {file_path}")  # Debug line
            try:
                result = query_function(file_path)
                QMessageBox.information(self, "Result", result)
                print("Classification result:", result)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")
                print(f"Error during classification: {e}")




def initialize_models():
    hf_login_token = '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'
    hf_models_token = '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'

    voice_model_repo = 'gbenari2/voice'
    specto_model_repo = 'gbenari2/specto'
    ensemble_model_repo = 'gbenari2/ensemble'
    voice_model_filename = 'voiceModel.pth'
    specto_model_filename = 'spectoModel.pth'
    ensemble_model_filename = 'ensembleModel.pth'

    # Download the models
    voice_model_path = download_model(voice_model_repo, voice_model_filename, hf_models_token)
    specto_model_path = download_model(specto_model_repo, specto_model_filename, hf_models_token)
    ensemble_model_path = download_model(ensemble_model_repo, ensemble_model_filename, hf_models_token)

    # Load the models and map to CPU
    #voice_model = torch.load(voice_model_path, map_location=torch.device('cpu'))
    specto_model = torch.load(specto_model_path, map_location=torch.device('cpu'))
    ensemble_model = torch.load(ensemble_model_path, map_location=torch.device('cpu'))

    # Set models to evaluation mode
    #voice_model.eval()
    specto_model.eval()
    ensemble_model.eval()

    # Define the path to the folder
    folder_path = 'Models'

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        os.makedirs(folder_path)

    # Save models to Models folder
    #torch.save(voice_model, 'Models/voice_model.pth')
    torch.save(specto_model, 'Models/specto_model.pth')
    torch.save(ensemble_model, 'Models/ensemble_model.pth')

    print("Initialization complete")


def main():
    initialize_models()
    test_file = "C:\\Users\\guybe\\OneDrive\\שולחן העבודה\\אפקה\\פרויקט גמר\\website\\11-real.wav"  # Replace with a valid file path
    result = query_function(test_file)
    print("test_file result:", result)

    app = QApplication(sys.argv)
    ex = AudioClassifierApp()
    sys.exit(app.exec_())


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
    if len(wav_file) < max_length:
        wav_file = np.pad(wav_file, (0, max_length - len(wav_file)), 'constant')

    # Run the audio through models
    mel = get_spectrogram(wav_file, sample_rate)
    wav_and_samp = np.concatenate([wav_file, [sample_rate]])
    wav_and_samp_t = torch.tensor(wav_and_samp)
    mel_t = torch.tensor(mel)

    # Load the models and map to CPU
    #voice_model = torch.load("Models/voice_model.pth", map_location=torch.device('cpu'))
    specto_model = torch.load("Models/specto_model.pth", map_location=torch.device('cpu'))
    ensemble_model = torch.load("Models/ensemble_model.pth", map_location=torch.device('cpu'))

    # Run into specto model
    specto_input = mel_t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        specto_output = specto_model(specto_input)
    specto_probs = torch.exp(specto_output)
    specto_probs = specto_probs / torch.sum(specto_probs) * 100
    specto_probs_list = specto_probs.cpu().numpy().flatten().tolist()
    print(specto_probs_list)

    # # Run into voice model
    # voice_input = wav_and_samp_t.unsqueeze(0)  # Add batch dimension
    # with torch.no_grad():
    #     voice_output = voice_model(voice_input)
    # voice_probs = torch.exp(voice_output)
    # voice_probs = voice_probs / torch.sum(voice_probs) * 100
    # voice_probs_list = voice_probs.cpu().numpy().flatten().tolist()

    # # Run into ensemble model
    # ensemble_input = torch.tensor([specto_probs_list, voice_probs_list], dtype=torch.float32)
    # ensemble_input = ensemble_input.unsqueeze(0)  # Add batch dimension and move to device
    # with torch.no_grad():
    #     ensemble_output = ensemble_model(ensemble_input)
    # ensemble_probs = torch.exp(ensemble_output)
    # ensemble_probs = ensemble_probs / torch.sum(ensemble_probs) * 100
    # ensemble_probs_list = ensemble_probs.cpu().numpy().flatten().tolist()

    # # Process output and return result
    # result = "The audio file is spoof." if ensemble_probs_list[1] > ensemble_probs_list[
    #     0] else "The audio file is bona-fide."

    result = "The audio file is spoof." if specto_probs_list[1] > specto_probs_list[
         0] else "The audio file is bona-fide."

    # Clean up models
    #os.remove("Models/specto_model.pth")
    #os.remove("Models/voice_model.pth")
    #os.remove("Models/ensemble_model.pth")

    # return result #to change
    return result


def download_model(repo_id, filename, token):
    """
    Download a model from Hugging Face Hub.

    Args:
        repo_id (str): Repository ID.
        filename (str): Filename to download.
        token (str): Hugging Face token.

    Returns:
        str: File path of the downloaded model.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token)


def install_dependencies():
    """
    Install required dependencies from requirements.txt.
    """
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing dependencies.")
        print(e)


def huggingface_login(token):
    """
    Login to Hugging Face using the provided token.

    Args:
        token (str): Hugging Face token.
    """
    try:
        process = subprocess.Popen(
            ['huggingface-cli', 'login'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=token + '\n')

        if process.returncode == 0:
            print("Hugging Face login completed successfully.")
        else:
            print("An error occurred while logging into Hugging Face.")
            print(stderr)
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing the process.")
        print(e)


if __name__ == '__main__':
    main()
