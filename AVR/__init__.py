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

setattr(__main__, 'getModelVoice', getModelVoice)

initialize_models()