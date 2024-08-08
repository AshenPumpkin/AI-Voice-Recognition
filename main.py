# Import necessary libraries
from system import initialize_models
from UI import AudioClassifierApp
import sys
from PyQt5.QtWidgets import QApplication
import warnings
from voiceModel import getModelVoice

# Main function to initialize the models and run the application
def main():
    # Suppress future warnings to clean up the console output
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Initialize models
    initialize_models()

    # Create the PyQt5 application instance and execute the event loop
    app = QApplication(sys.argv)
    ex = AudioClassifierApp()
    sys.exit(app.exec_())

# Entry point of the application
if __name__ == '__main__':
    main()
