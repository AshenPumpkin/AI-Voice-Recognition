from .system import initialize_models, initialize_system, install_dependencies
import importlib
from .models import query_function
from .system import clean as flush

# Initialize system configurations
initialize_system()

# Dynamically import and reload the voice model module
module = importlib.import_module("AVR.voiceModel")
importlib.reload(module)

from voiceModel import getModelVoice

initialize_models()