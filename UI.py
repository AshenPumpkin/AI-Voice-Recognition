from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from models import query_function
from system import clean, initialize_models

class AudioClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.models_initialized = False
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Audio Classifier')
        layout = QVBoxLayout()

        open_button = QPushButton('Upload Audio File', self)
        open_button.clicked.connect(self.show_dialog)
        layout.addWidget(open_button)

        close_button = QPushButton('Close', self)
        close_button.clicked.connect(self.close_app)
        layout.addWidget(close_button)

        self.setLayout(layout)
        self.show()

    def show_dialog(self):
        if not self.models_initialized:
            self.initialize_models()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Please select the audio file you want to recognize",
            "",
            "Audio Files (*.mp3 *.wav)",
            options=options
        )
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        processing_dialog = QMessageBox(self)
        processing_dialog.setWindowTitle("Processing")
        processing_dialog.setText("AI is recognizing the audio now....")
        processing_dialog.setStandardButtons(QMessageBox.NoButton)
        processing_dialog.show()

        try:
            result = query_function(file_path)
            processing_dialog.accept()
            self.show_result(result)
        except Exception as e:
            processing_dialog.accept()
            self.show_error(str(e))

    def show_result(self, result):
        QMessageBox.information(self, "Result", result)
        print("Classification result:", result)

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        print(f"Error during classification: {error_message}")

    def close_app(self):
        clean()
        QApplication.instance().quit()

    def initialize_models(self):
        init_dialog = QMessageBox(self)
        init_dialog.setWindowTitle("Initializing")
        init_dialog.setText("Initializing AI models...")
        init_dialog.setStandardButtons(QMessageBox.NoButton)
        init_dialog.show()

        initialize_models()
        self.models_initialized = True
        init_dialog.accept()
