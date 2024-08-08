# Import necessary libraries
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QDesktopServices, QCloseEvent
from PyQt5.QtCore import QUrl
from models import query_function
from system import clean, initialize_models


# Create the main application window
class AudioClassifierApp(QWidget):
    # Initialize the application
    def __init__(self):
        super().__init__()
        self.init_ui()

    # Create the user interface
    def init_ui(self):
        self.setWindowTitle('AI Voice Recognize')
        self.setFixedSize(500, 400)
        layout = QVBoxLayout()

        open_button = QPushButton('Upload Audio File', self)
        open_button.clicked.connect(self.show_dialog)
        layout.addWidget(open_button)

        website_button = QPushButton('Open AVR Website', self)
        website_button.clicked.connect(self.open_website)
        layout.addWidget(website_button)

        close_button = QPushButton('Close', self)
        close_button.clicked.connect(self.close_app)
        layout.addWidget(close_button)

        self.setLayout(layout)
        self.show()

    # Show the file dialog to select the audio file
    def show_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Please select the audio file you want to recognize:",
            "",
            "Audio Files (*.mp3 *.wav)",
            options=options
        )
        if file_path:
            self.process_file(file_path)

    # Process the selected audio file
    def process_file(self, file_path):
        processing_dialog = QMessageBox(self)
        processing_dialog.setWindowTitle("Processing audio")
        processing_dialog.setText("Please wait while the AI is recognizing the audio file...")
        processing_dialog.setStandardButtons(QMessageBox.NoButton)
        processing_dialog.setModal(True)  # Ensure it blocks interaction
        processing_dialog.show()

        QApplication.processEvents()  # Allow UI to update

        try:
            result = query_function(file_path)
            processing_dialog.accept()  # Close dialog after processing
            self.show_result(result)

        except Exception as e:
            processing_dialog.close()  # Close dialog if an error occurs
            self.show_error(str(e))

    # Display the classification result
    def show_result(self, result):
        QMessageBox.information(self, "AVR audio file prediction results:", result)

    # Display an error message
    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        print(f"Error during classification: {error_message}")

    # Close the application
    def close_app(self):
        clean()
        QApplication.instance().quit()

    # Initialize the AI models
    def initialize_models(self):
        init_dialog = QMessageBox(self)
        init_dialog.setWindowTitle("Initializing system")
        init_dialog.setText("Please wait while the AI models are initialized...")
        init_dialog.setStandardButtons(QMessageBox.NoButton)
        init_dialog.show()
        initialize_models()
        init_dialog.accept()

    # Open the AVR website
    def open_website(self):
        url = "https://ashenpumpkin.github.io/AI-Voice-Recognition/"
        QDesktopServices.openUrl(QUrl(url))

    # Handle the close event
    def closeEvent(self, event: QCloseEvent):
        self.close_app()
        event.accept()
