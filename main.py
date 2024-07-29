# Imports
import os
import shutil
import sys
import subprocess
import importlib
import numpy as np
import librosa as lb
from huggingface_hub import hf_hub_download
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
import torch
import torch.nn as nn



class getModelVoice(nn.Module):
    def __init__(self, input_size=551053, hidden_size=128, num_layers=2, num_classes=2):
        super(getModelVoice, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=0.5)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension: (batch_size, 1, input_size)
        out, _ = self.lstm1(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Utility function to compute mel spectrogram
def get_spectrogram(wav_file, sample_rate):
    mel_spectrogram = lb.feature.melspectrogram(y=wav_file, sr=sample_rate)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    return np.expand_dims(mel_spectrogram_db, axis=0)


class AudioClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.models_initialized = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Audio Classifier')
        layout = QVBoxLayout()

        open_button = QPushButton('Upload Audio File', self)
        open_button.clicked.connect(self.show_dialog)
        layout.addWidget(open_button)

        close_button = QPushButton('Close', self)
        close_button.clicked.connect(self.close_app)
        layout.addWidget(close_button)

        self.setLayout(layout)
        self.show()

    def show_dialog(self):
        if not self.models_initialized:
            self.initialize_models()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Please select the audio file you want to recognize",
            "",
            "Audio Files (*.mp3 *.wav)",
            options=options
        )
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        processing_dialog = QMessageBox(self)
        processing_dialog.setWindowTitle("Processing")
        processing_dialog.setText("AI is recognizing the audio now....")
        processing_dialog.setStandardButtons(QMessageBox.NoButton)
        processing_dialog.show()

        try:
            result = query_function(file_path)
            processing_dialog.accept()
            self.show_result(result)
        except Exception as e:
            processing_dialog.accept()
            self.show_error(str(e))

    def show_result(self, result):
        QMessageBox.information(self, "Result", result)
        print("Classification result:", result)

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        print(f"Error during classification: {error_message}")

    def close_app(self):
        clean()
        QApplication.instance().quit()

    def initialize_models(self):
        init_dialog = QMessageBox(self)
        init_dialog.setWindowTitle("Initializing")
        init_dialog.setText("Initializing AI models...")
        init_dialog.setStandardButtons(QMessageBox.NoButton)
        init_dialog.show()

        initialize_models()
        self.models_initialized = True
        init_dialog.accept()

def initialize_models():

    hf_login_token = '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'
    hf_models_token = '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'

    voice_model_repo = 'gbenari2/voice'
    specto_model_repo = 'gbenari2/specto'
    ensemble_model_repo = 'gbenari2/ensemble'
    voice_model_filename = 'voiceModel.pth'
    specto_model_filename = 'spectoModel.pth'
    ensemble_model_filename = 'ensembleModel.pth'
    custom_models_filename = 'Voice_model_loader.py'
    # Define the path to the folder
    folder_path = 'Models'

    # Download the models
    voice_model_path = download_model(voice_model_repo, voice_model_filename, hf_models_token)
    specto_model_path = download_model(specto_model_repo, specto_model_filename, hf_models_token)
    ensemble_model_path = download_model(ensemble_model_repo, ensemble_model_filename, hf_models_token)

    # # Download the Python file
    # voice_class_path = hf_hub_download(repo_id=voice_model_repo, filename=custom_models_filename,
    #                                    use_auth_token=hf_models_token)
    # # Move the Python file to the Models folder
    # shutil.copy(voice_class_path, os.path.join(folder_path, custom_models_filename))
    #
    # custom_model_save_path = os.path.join('Models', custom_models_filename)
    # os.rename(voice_class_path, custom_model_save_path)
    #
    # # Print to verify the path
    # print(f"Voice model class path: {custom_model_save_path}")

    # Load the models and map to CPU
    voice_model = torch.load(voice_model_path, map_location=torch.device('cpu'))
    specto_model = torch.load(specto_model_path, map_location=torch.device('cpu'))
    ensemble_model = torch.load(ensemble_model_path, map_location=torch.device('cpu'))

    # Set models to evaluation mode
    voice_model.eval()
    specto_model.eval()
    ensemble_model.eval()

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        os.makedirs(folder_path)

    # Save models to Models folder
    torch.save(voice_model, 'Models/voice_model.pth')
    torch.save(specto_model, 'Models/specto_model.pth')
    torch.save(ensemble_model, 'Models/ensemble_model.pth')

    print("Initialization complete")


def main():
    install_dependencies()
    getModelVoice = import_voice_model()  # Import the getModelVoice class

    if getModelVoice is None:
        print("Failed to import the getModelVoice class. Exiting.")
    else:
        print("getModelVoice class imported successfully.")
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

    # Run into specto model
    specto_input = mel_t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        specto_output = specto_model(specto_input)
    specto_probs = torch.exp(specto_output)
    specto_probs = specto_probs / torch.sum(specto_probs) * 100
    specto_probs_list = specto_probs.cpu().numpy().flatten().tolist()
    print(specto_probs_list)

    # Run into voice model
    voice_input = wav_and_samp_t.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        voice_output = voice_model(voice_input)
    voice_probs = torch.exp(voice_output)
    voice_probs = voice_probs / torch.sum(voice_probs) * 100
    voice_probs_list = voice_probs.cpu().numpy().flatten().tolist()

    # Combine the probabilities for ensemble model
    combined_probs_list = specto_probs_list + voice_probs_list  # Concatenate lists

    # Run into ensemble model
    ensemble_input = torch.tensor(combined_probs_list, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        ensemble_output = ensemble_model(ensemble_input)
    ensemble_probs = torch.exp(ensemble_output)
    ensemble_probs = ensemble_probs / torch.sum(ensemble_probs) * 100
    ensemble_probs_list = ensemble_probs.cpu().numpy().flatten().tolist()

    spoof_probability = ensemble_probs_list[0]
    result = "The audio file is spoof." if spoof_probability > 50 else "The audio file is bona-fide."

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

def clean():
    folder_path = 'Models'
    files_to_remove = ["specto_model.pth", "voice_model.pth", "ensemble_model.pth", "Voice_model_loader.py"]

    for file_name in files_to_remove:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")

def import_voice_model():
    # Add Models directory to sys.path
    sys.path.append(os.path.abspath('Models'))

    # Import the getModelVoice class from the downloaded file
    try:
        voice_model_loader = importlib.import_module('Voice_model_loader')
        getModelVoice = getattr(voice_model_loader, 'getModelVoice')
        print("Model imported successfully.")
        return getModelVoice
    except ModuleNotFoundError as e:
        print(f"Error importing model: {e}")
        return None


if __name__ == '__main__':
    main()
