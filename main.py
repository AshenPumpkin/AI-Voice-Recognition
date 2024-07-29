# Imports
from system import install_dependencies, import_voice_model
from UI import AudioClassifierApp
import sys
from PyQt5.QtWidgets import QApplication
from models import getModelVoice
import warnings


def main():
    install_dependencies()
    warnings.filterwarnings("ignore", category=FutureWarning)

    app = QApplication(sys.argv)
    ex = AudioClassifierApp()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()
