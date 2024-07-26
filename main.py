# First open terminal and run: pip install -r requirements.txt

# imports
import subprocess
import numpy as np
import librosa as lb
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
import os


# needed funcs
def getSpecto(wavFile, sampleRate):
    # Compute mel spectrogram
    mel_spectrogram = lb.feature.melspectrogram(y=wavFile, sr=sampleRate)
    mel_spectrogram_db = lb.power_to_db(mel_spectrogram, ref=np.max)

    # Add channel dimension while preserving the actual values
    melTemp = np.expand_dims(mel_spectrogram_db, axis=0)

    return melTemp


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PyQt Example')
        layout = QVBoxLayout()

        open_button = QPushButton('Open Audio File', self)
        open_button.clicked.connect(self.showDialog)
        layout.addWidget(open_button)

        self.setLayout(layout)
        self.show()

    def showDialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.mp3 *.wav)",
                                                   options=options)
        if file_path:
            result = query_function(file_path)
            QMessageBox.information(self, "Result", result)


def initialize():
    # install_dependencies()
    #subprocess.run([sys.executable, '-m', 'huggingface-cli', 'login'], check=True)
    hf_login_token = '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'
    # Set your Hugging Face hf_models_token if needed
    hf_models_token = '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'
    #huggingface_login(hf_login_token)

    # Repositories and filenames
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
    voice_model = torch.load(voice_model_path, map_location=torch.device('cpu'))
    specto_model = torch.load(specto_model_path, map_location=torch.device('cpu'))
    ensemble_model = torch.load(ensemble_model_path, map_location=torch.device('cpu'))
    print("load")


    voice_model.eval()
    specto_model.eval()
    ensemble_model.eval()
    print("eval")


    # save models to Models folder
    torch.save(voice_model, 'Models/voiceModel.pth')
    torch.save(specto_model, 'Models/spectoModel.pth')
    torch.save(ensemble_model, 'Models/ensembleModel.pth')
    print("save")


    print("Initialization complete")


def main():
    initialize()

    app = QApplication(sys.argv)
    ex = App()  # Create an instance of the App class, which initializes and shows the main window
    sys.exit(app.exec_())  # Start the event loop


def query_function(file_path):

    print(file_path)
    # load the audio file
    wavFile, sampleRate = lb.load(file_path)

    max_length = 551052  # longest datapoint, 24 seconds

    # Pad or truncate the audio file to ensure consistent length
    if len(wavFile) < max_length:
        # Pad with zeros if the length is less than the maximum length
        wavFile = np.pad(wavFile, (0, max_length - len(wavFile)), 'constant')

    # run into models
    mel = getSpecto(wavFile, sampleRate)
    wavAndSamp = np.concatenate(wavFile, sampleRate)
    wavAndSampT = torch.tensor(wavAndSamp)
    melT = torch.tensor(mel)

    # Load the models and map to CPU
    voice_model = torch.load("Models/voice_Model.pth", map_location=torch.device('cpu'))
    specto_model = torch.load("Models/specto_Model.pth", map_location=torch.device('cpu'))
    ensemble_model = torch.load("Models/ensemble_Model.pth", map_location=torch.device('cpu'))

    # run into specto
    spectoInput = melT.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        spectoOutput = specto_model(spectoInput)
    print(spectoOutput)
    spectoProbs = torch.exp(spectoOutput)
    spectoProbs = spectoProbs / torch.sum(spectoProbs) * 100
    spectoProbs_list = spectoProbs.cpu().numpy().flatten().tolist()
    formatted_probs = [f"{prob:.2f}" for prob in spectoProbs_list]
    print(formatted_probs)

    #run into voice
    voiceInput = wavAndSampT.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        voiceOutput = voice_model(voiceInput)
    voiceProbs = torch.exp(voiceOutput)
    voiceProbs = voiceProbs / torch.sum(voiceProbs) * 100
    voiceProbs_list = voiceProbs.cpu().numpy().flatten().tolist()
    formatted_voice_probs = [f"{prob:.2f}" for prob in voiceProbs_list]
    print(formatted_voice_probs)

    #run into ensemble
    ensembleInput = torch.tensor([spectoProbs_list, voiceProbs_list], dtype=torch.float32)
    ensembleInput = ensembleInput.unsqueeze(0)  # Add batch dimension and move to device
    with torch.no_grad():
        ensembleOutput = ensemble_model(ensembleInput)
    ensembleProbs = torch.exp(ensembleOutput)
    ensembleProbs = ensembleProbs / torch.sum(ensembleProbs) * 100
    ensembleProbs_list = ensembleProbs.cpu().numpy().flatten().tolist()
    formatted_ensemble_probs = [f"{prob:.2f}" for prob in ensembleProbs_list]
    print(formatted_ensemble_probs)

        #process output and return
    if ensembleProbs[1] > ensembleProbs[0]:
        return "The audio file is spoof."
    else:
        return "The audio file is bona-fide."
    os.remove("Models/specto_Model.pth")
    os.remove("Models/voice_Model.pth")
    os.remove("Models/ensemble_Model.pth")
    return "formatted_probs"


# Function to download a file from Hugging Face Hub
def download_model(repo_id, filename, token):
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token)
    return file_path


# Function to load a model from Hugging Face
def load_model(repo_name, token):
    model = AutoModel.from_pretrained(repo_name, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(repo_name, use_auth_token=token)
    return model, tokenizer


def install_dependencies():
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing dependencies.")
        print(e)


def huggingface_login(token):
    try:
        # Start the huggingface-cli login process
        process = subprocess.Popen(
            ['C:\\Users\\guybe\\PycharmProjects\\AI-Voice-Recognition\\venv\\Scripts\\huggingface-cli.exe', 'login'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Send the token to the process's stdin
        stdout, stderr = process.communicate(input=token + '\n')

        # Check for errors
        if process.returncode == 0:
            print("Hugging Face login completed successfully.")
        else:
            print("An error occurred while logging into Hugging Face.")
            print(stderr)
    except subprocess.CalledProcessError as e:
        print("An error occurred while executing the process.")
        print(e)


# Main.
if __name__ == '__main__':
    main()
