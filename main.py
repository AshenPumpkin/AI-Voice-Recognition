# Imports
from system import initialize_models, initialize_system
from UI import AudioClassifierApp
import sys
from PyQt5.QtWidgets import QApplication
import warnings
import importlib
initialize_system()
module = importlib.import_module("voiceModel")
importlib.reload(module)
from voiceModel import getModelVoice


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    initialize_models()

    # Create the application
    app = QApplication(sys.argv)
    ex = AudioClassifierApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
