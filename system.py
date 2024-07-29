from huggingface_hub import hf_hub_download
import os
import torch
import subprocess
import sys
import importlib
from models import getModelVoice
#import shutil


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

    getModelVoiceClass = import_voice_model()  # Import the getModelVoice class

    if getModelVoiceClass is None:
        print("Failed to import getModelVoice class.")
    else:
        print("getModelVoice class imported successfully.")

    try:
        voice_model = torch.load(voice_model_path, map_location=torch.device('cpu'))
        specto_model = torch.load(specto_model_path, map_location=torch.device('cpu'))
        ensemble_model = torch.load(ensemble_model_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Failed to load the models. Error: {e}")

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
        getModelVoiceClass = getattr(voice_model_loader, 'getModelVoice')
        print("Model imported successfully.")
        return getModelVoiceClass
    except ModuleNotFoundError as e:
        print(f"Error importing model: {e}\nInstead, loading hardcoded archtiecture.")

        return None
