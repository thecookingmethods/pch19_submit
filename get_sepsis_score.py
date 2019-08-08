#!/usr/bin/env python

import numpy as np
from tensorflow import keras

MODEL_FILE_NAME = 'weights00000005-0.4603-0.8046-0.4603.h5'

PROB_MOD = 0.35

def get_sepsis_score(data, model):

    test_example = np.array([data])

    prediction_probabs = model.predict_on_batch(test_example)
    prediction_class = np.round(prediction_probabs + PROB_MOD)

    score = prediction_probabs[0,-1,0]
    label = prediction_class[0,-1,0]

    return score, label

def load_sepsis_model():
    model = keras.models.load_model(MODEL_FILE_NAME)
    return model
