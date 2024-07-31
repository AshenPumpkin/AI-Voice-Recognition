from huggingface_hub import hf_hub_download
import os
import torch
import subprocess
import sys
import importlib
from voiceModel import getModelVoice
import dill

#global variables
paths_array = []
dummy_contents = ''

def initialize_models():
    global paths_array

    hf_login_token = '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'
    hf_models_token = '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'

    print("get to initialize_models")

    huggingface_login(hf_login_token)

    voice_model_repo = 'gbenari2/voice'
    specto_model_repo = 'gbenari2/specto'
    ensemble_model_repo = 'gbenari2/ensemble'
    voice_model_filename = 'voiceModel.pth'
    specto_model_filename = 'spectoModel.pth'
    ensemble_model_filename = 'ensembleModel.pth'
    custom_models_filename = 'Voice_model_loader.py'
    # Define the path to the folder
    folder_path = 'Models'

    print("start download models")

    # Download the models
    voice_model_path = download_model(voice_model_repo, voice_model_filename, hf_models_token)
    specto_model_path = download_model(specto_model_repo, specto_model_filename, hf_models_token)
    ensemble_model_path = download_model(ensemble_model_repo, ensemble_model_filename, hf_models_token)

    # Append the paths to the array
    paths_array.append(voice_model_path)
    paths_array.append(specto_model_path)
    paths_array.append(ensemble_model_path)

    # Import the getModelVoice class
    import_voice_model(voice_model_repo, custom_models_filename, hf_models_token)


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
    Log in to Hugging Face using the provided token.

    Args:
        token (str): Hugging Face token.
    """
    try:
        # Define the command with the --token option
        command = ['huggingface-cli', 'login', '--token', token]

        # Execute the command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Capture the output and errors
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Hugging Face login completed successfully.")
            print(stdout)  # Optional: print the stdout to see the response
        else:
            print("An error occurred while logging into Hugging Face.")
            print(stderr)  # Print stderr to see any error messages

    except Exception as e:
        print("An unexpected error occurred while executing the process.")
        print(e)


def logout_huggingface():
    subprocess.run(['huggingface-cli', 'logout'])


def clean():
    global dummy_contents
    global paths_array

    #reset dummy file
    with open('voiceModel.py', 'w') as f:
        f.write(dummy_contents)

    # Remove the downloaded files
    for path in paths_array:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed {os.path.basename(path)}")
            except OSError as e:
                print(f"Error removing {os.path.basename(path)}: {e}")

    # logout from huggingface
    logout_huggingface()


def import_voice_model(repo, filename, token):
    global dummy_contents
    global paths_array

    path_to_py = download_model(repo, filename, token)
    dummy_file_path = 'voiceModel.py'
    module_name = 'voiceModel'

    paths_array.append(path_to_py)  # Append the path to the array to delete at shutdown

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

        # Reload the module
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)

        print("Model imported successfully.")
    except ModuleNotFoundError as e:
        print(f"Error importing model: {e}\nInstead, loading hardcoded archtiecture.")