# Imports
from system import install_dependencies, initialize_models
from UI import AudioClassifierApp
import sys
from PyQt5.QtWidgets import QApplication
import warnings
from voiceModel import getModelVoice

def main():
    # Initialization
    install_dependencies()
    warnings.filterwarnings("ignore", category=FutureWarning)
    initialize_models()


    # Create the application
    app = QApplication(sys.argv)
    ex = AudioClassifierApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
