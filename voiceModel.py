import torch
import torch.nn as nn

# Dummy model, will be replaced with the actual model during the live session
# Then returned to the dummy model during shutdown.
# This is to prevent the model from being revealed in the code and to the user.
class getModelVoice(nn.Module):
    def __init__(self):
        pass

 
    def forward(self):
        pass

    def initialize_weights(self):
        pass