#!/usr/bin/env python

import numpy as np
import os
import pickle

from model import Model
from parse_dataset import PatientData
from transformer import Transformer, ModifiedStandardScaler

MODEL_ROOT_DIR = os.path.dirname(__file__)

MODEL_NAME = 'prototypes/m_20190821072955/best_loss'
DECISION_THERSHOLD = 0.30
PICKLED_DIR = 'pickled_data'
PICKLED_FILE_DATA = 'data.pkl'

def get_sepsis_score(data, model):
    # input defined by physionet challenge
    # data: shape = (t, 40)
    # model: classification model returned from load_sepsis_model()

    dummy_y = np.zeros(data.shape[0])
    patient_data = PatientData(sequence_x=data,
                               sequence_y=dummy_y)

    y_true, y_prob = model.get_predictions([patient_data], 1)

    return y_prob, np.round(y_prob + DECISION_THERSHOLD)


def load_sepsis_model():
    transformer = load_pickled_data('transformer', 'pickled_data')
    model = Model(transformer=transformer,
                  model_dir=os.path.join(MODEL_ROOT_DIR, MODEL_NAME),
                  only_eval=True)
    model.__enter__()
    return model


def load_pickled_data(filename, dirname):
    with open(os.path.join(dirname, '{}_{}'.format(filename, PICKLED_FILE_DATA)), 'rb') as f:
        return pickle.load(f)