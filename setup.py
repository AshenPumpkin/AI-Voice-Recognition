from setuptools import setup, find_packages

setup(
    name='AVRecognize',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'torch>=1.13.1'
        'librosa>=1.0.0'
    ],
)