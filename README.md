# AI Voice Recognize (AVR) 🎙️
Usage Instructions

    Download and Install:
        Download the latest release
        Extract the entirety of the release to a directory on your machine
        Simply run the .exe file to launch the application.

    Using the Application:
        Open the executable file.
        Use the graphical interface to upload audio files.
        The system will process the audio and display whether it is genuine or spoofed.

Overview

The AI Voice Recognize (AVR) project, developed by Guy Ben Ari and Ynon Friedman from Afeka College of Engineering, focuses on detecting spoofed audio files and distinguishing them from genuine human voices using advanced machine learning techniques.
Dataset

ASVspoof2019 dataset is used, featuring a diverse range of real and spoofed audio files for model training and evaluation.
Approach

    Architecture: Utilizes LSTM networks for sequential audio data and Conv2D layers for feature extraction from MFCC images.
    Model: Features are concatenated and classified using fully connected layers.

Development Environment

    Prototyping and Training: Conducted using JupyterLab Notebook.
    Version Control: Managed with Git.

Integration

The system is packaged as an executable file for ease of use, providing both real-time and batch processing capabilities.
Testing Breakdown

    Tests Conducted: Includes UI functionality, audio processing accuracy, and model integration.
    Results: Includes test scripts, issues identified, and corrective actions taken.

Limitations and Solutions

    Data Inconsistency: Resolved through normalization techniques.
    Integration Issues: Fixed by integrating with HuggingFace.
    Resource Constraints: Addressed by upgrading VRAM.
    Performance: Enhanced with CUDA.
    UI and Security: Improved based on feedback and added encryption.

Contact

For inquiries or collaboration:

    Guy Ben Ari: gbenari2@gmail.com
    Ynon Friedman: ynonfridman@gmai.com
