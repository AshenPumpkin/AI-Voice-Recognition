# Import necessary libraries
from system import initialize_models, initialize_system
from UI import AudioClassifierApp
import sys
from PyQt5.QtWidgets import QApplication
import warnings
import importlib

# Initialize the system, and reload the voiceModel module
initialize_system()

module = importlib.import_module("voiceModel")
importlib.reload(module)

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
