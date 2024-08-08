from setuptools import setup, find_packages

setup(
    name='AVR',
    version='0.1.0',
    packages=find_packages(include=['AVR', 'AVR.*']),
    install_requires=[
        'librosa >= 0.10.1',
        'matplotlib >= 3.7.3',
        'numpy >= 1.24.0',
        'pandas >= 2.1.0',
        'torch == 2.4.0',
        'torchvision == 0.19.0',
        'torchaudio == 2.4.0',
        'scipy >= 1.11.2',
        'soundfile >= 0.12.1',
        'setuptools >= 65.5.0',
        'scikit-learn >= 1.3.0',
        'transformers >= 4.33.2',
        'huggingface_hub >= 0.17.2',
        'dill >= 0.3.8',
    ],
    entry_points={
        'console_scripts': [
            # 'command_name = your_project.module:function',
        ],
    },
    author='Ynon Friedman & Guy Ben Ari',
    author_email='ynonfridman@gmail.com, Gbenari2@gmail.com',
    description='AVR is a voice anti-spoofing system that uses deep learning models to detect spoofed audio files.',
    long_description=open('README.md',encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AshenPumpkin/AI-Voice-Recognition',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
