# score.py
from inference import load_model as _load_model, predict as _predict

def load_model():
    return _load_model()

def predict(data, model=None):
    if model is None:
        model = load_model()
    return _predict(data, model)