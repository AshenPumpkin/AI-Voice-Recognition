import os
import torch

#executes on library import

# Get the directory path of the current script (AVRecognize/__init__.py)
module_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the mock_model.pth file relative to the current script
path_to_model = os.path.join(module_dir, 'Helpers', 'mock_model.pth')

#load premade
premadeModel = torch.load(path_to_model)


