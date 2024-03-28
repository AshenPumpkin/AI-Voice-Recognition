from . import premadeModel
from featuresWorker import _featureExtractor


def queryModel(file):
    #this function will return a prediction on one audio sample
    featuresExtract = _featureExtractor(file)
    result = _queryModelOneFile(premadeModel, featuresExtract)
    return result

def _queryModelOneFile(model, features):
    result = premadeModel(features)
    # Convert tensor to a Python scalar
    scalar = result.item()
    if scalar == 0:
        res = False
    else:
        res = True
    return res
