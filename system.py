# Import necessary libraries
from huggingface_hub import hf_hub_download, login, logout
import os
import torch
import subprocess
import sys
import dill

# Global variables
paths_array = []
dummy_contents = ''
hf_login_token = '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'
hf_models_token = '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'
voice_model_repo = 'gbenari2/voice'
specto_model_repo = 'gbenari2/specto'
ensemble_model_repo = 'gbenari2/ensemble'
voice_model_filename = 'voiceModel.pth'
specto_model_filename = 'spectoModel.pth'
ensemble_model_filename = 'ensembleModel.pth'


# Initialize the models
def initialize_models():
    global paths_array
    global hf_login_token
    global  hf_models_token
    global voice_model_repo
    global specto_model_repo
    global ensemble_model_repo
    global voice_model_filename
    global specto_model_filename
    global ensemble_model_filename

    huggingface_login()

    # Define the path to the folder
    folder_path = 'Models'

    # Download the models
    voice_model_path = download_model(voice_model_repo, voice_model_filename, hf_models_token)
    specto_model_path = download_model(specto_model_repo, specto_model_filename, hf_models_token)
    ensemble_model_path = download_model(ensemble_model_repo, ensemble_model_filename, hf_models_token)

    # Append the paths to the array
    paths_array.append(voice_model_path)
    paths_array.append(specto_model_path)
    paths_array.append(ensemble_model_path)

    try:
        voice_model = torch.load(voice_model_path, map_location=torch.device('cpu'))
        specto_model = torch.load(specto_model_path, map_location=torch.device('cpu'))
        ensemble_model = torch.load(ensemble_model_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Failed to load the models. Error: {e}")

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        os.makedirs(folder_path)

    # Save models to Models folder
    torch.save(voice_model, 'Models/voice_model.pth', pickle_module=dill)
    torch.save(specto_model, 'Models/specto_model.pth')
    torch.save(ensemble_model, 'Models/ensemble_model.pth')

    # Append the paths to the array to delete at shutdown
    paths_array.append('Models/voice_model.pth')
    paths_array.append('Models/specto_model.pth')
    paths_array.append('Models/ensemble_model.pth')

    print("Initialization complete")


# Download a model from Hugging Face Hub
def download_model(repo_id, filename, token):
    return hf_hub_download(repo_id=repo_id, filename=filename, use_auth_token=token)


# Install dependencies
def install_dependencies():
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing dependencies.")
        print(e)


# Log in to Hugging Face
def huggingface_login():
    global hf_login_token
    try:
        login(hf_login_token)
    except Exception as e:
        print("An unexpected error occurred while executing the process.")
        print(e)


# Logout from Hugging Face
def logout_huggingface():
    logout()


# Clean up the system
def clean():
    global dummy_contents
    global paths_array

    # Remove the downloaded files
    for path in paths_array:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed {os.path.basename(path)}")
            except OSError as e:
                print(f"Error removing {os.path.basename(path)}: {e}")

    # Logout from huggingface
    logout_huggingface()