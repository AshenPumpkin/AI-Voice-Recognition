# Import necessary libraries
from system import initialize_models, initialize_system
from UI import AudioClassifierApp
import sys
from PyQt5.QtWidgets import QApplication
import warnings
import importlib

# Initialize system configurations
initialize_system()

# Dynamically import and reload the voice model module
module = importlib.import_module("voiceModel")
importlib.reload(module)

# Import the voice model class after reloading the module
from voiceModel import getModelVoice

def main():
    # Suppress future warnings to clean up the console output
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Initialize models
    initialize_models()

    # Create the PyQt5 application instance and execute the event loop
    app = QApplication(sys.argv)
    ex = AudioClassifierApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
