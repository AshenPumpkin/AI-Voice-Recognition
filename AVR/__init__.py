# Description: This file is used to initialize the system configurations and models.
# imports
from .system import initialize_models, initialize_system, install_dependencies
import importlib
from .models import query_function
from .system import clean as flush
import __main__


# Initialize system configurations
initialize_system()

module = importlib.import_module("AVR.voiceModel")
importlib.reload(module)

from .voiceModel import getModelVoice

# Set the getModelVoice function to the main module to allow torch.load to work
setattr(__main__, 'getModelVoice', getModelVoice)

initialize_models()