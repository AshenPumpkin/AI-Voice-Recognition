# Import necessary libraries
from huggingface_hub import hf_hub_download, login
import os
import torch
import subprocess
import sys
import dill
import warnings


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
custom_models_filename = 'Voice_model_loader.py'


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
    global custom_models_filename

    # Define the path to the folder
    folder_path = 'AVR/Models'

    # Download the models
    voice_model_path = download_model(voice_model_repo, voice_model_filename, hf_models_token)
    specto_model_path = download_model(specto_model_repo, specto_model_filename, hf_models_token)
    ensemble_model_path = download_model(ensemble_model_repo, ensemble_model_filename, hf_models_token)

    # Append the paths to the array
    paths_array.append(voice_model_path)
    paths_array.append(specto_model_path)
    paths_array.append(ensemble_model_path)

    # Load the models
    voice_model = torch.load(voice_model_path, map_location=torch.device('cpu'), pickle_module=dill)
    specto_model = torch.load(specto_model_path, map_location=torch.device('cpu'), pickle_module=dill)
    ensemble_model = torch.load(ensemble_model_path, map_location=torch.device('cpu'), pickle_module=dill)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # Create the folder if it does not exist
        os.makedirs(folder_path)

    # Save models to Models folder
    torch.save(voice_model, 'AVR/Models/voice_model.pth', pickle_module=dill)
    torch.save(specto_model, 'AVR/Models/specto_model.pth', pickle_module=dill)
    torch.save(ensemble_model, 'AVR/Models/ensemble_model.pth', pickle_module=dill)

    # Append the paths to the array to delete at shutdown
    paths_array.append('Models/voice_model.pth')
    paths_array.append('Models/specto_model.pth')
    paths_array.append('Models/ensemble_model.pth')

    print("AVR: Initialization complete")


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
def huggingface_login(token):
    try:
        # Save the current stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect stdout and stderr to null
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # Perform the login
        login(token, add_to_git_credential=False)

    except Exception as e:
        # Restore stdout and stderr before printing the error
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print("An unexpected error occurred while executing the process.")
        print(e)

    finally:
        # Restore stdout and stderr after the login
        sys.stdout = original_stdout
        sys.stderr = original_stderr

# Logout from Hugging Face
def logout_huggingface():
    subprocess.run(['huggingface-cli', 'logout'])


# Clean up the system
def clean():
    global dummy_contents
    global paths_array

    # Reset the dummy file
    with open('AVR/voiceModel.py', 'w') as f:
        f.write(dummy_contents)

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


# Initialize the system
def initialize_system():
    global dummy_contents
    global paths_array
    global voice_model_repo
    global custom_models_filename
    global hf_models_token

    # Ignore futurewarning from torch.load and huggingface model download warning about symlink
    warnings.filterwarnings("ignore")

    # log in to huggingface
    huggingface_login(hf_login_token)

    # Download the voice model
    path_to_py = download_model(voice_model_repo, custom_models_filename, hf_models_token)


    # Define the path to the dummy file
    dummy_file_path = 'AVR/voiceModel.py'

    # Append the path to the array to delete at shutdown
    paths_array.append(path_to_py)

    # Import the getModelVoice class from the downloaded file
    try:
        # Read the content of the downloaded file
        with open(path_to_py, 'r') as downloaded_file:
            downloaded_content = downloaded_file.read()

        with open(dummy_file_path, 'r') as dummy_read_file:
            dummy_contents = dummy_read_file.read()

        # Replace the content of the dummy file with the downloaded content
        with open(dummy_file_path, 'w') as dummy_file:
            dummy_file.write(downloaded_content)

        print("Module imported successfully.")
    except ModuleNotFoundError as e:
        print(f"Error importing model: {e}\nInstead, loading hardcoded archtiecture.")
