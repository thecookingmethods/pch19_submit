import numpy as np
import random
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

from patient_data import PatientData
from transformer import Transformer
from utils import print_progress

DATA_A_DIR = 'training_setA/training'
DATA_B_DIR = 'training_setB'

PICKLED_DIR = 'pickled_data'
PICKLED_FILE_DATA = 'data.pkl'

PROGRESS_ITER = 50

RATIO_TRAIN_TO_EVAL = 0.7
RATIO_VALID_TO_TEST = 0.5

SHUFFLE_SEED = 12345
MAX_NUM_FILES = -1


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # silence sklearn

    data = read_data(DATA_A_DIR)
    data_b = read_data(DATA_B_DIR)
    data.extend(data_b)
    max_seq_len = max(data).seq_len

    train, valid, test = split_data(data)
    display_class_ratios(train, valid, test)

    transformer = Transformer(max_seq_len).fit(train)

    save_pickled_data(train=train,
                      valid=valid,
                      test=test,
                      transformer=transformer,
                      dirname=PICKLED_DIR)


def read_data(data_dir, max_files=MAX_NUM_FILES):
    # prepare
    data = []

    # sort files for reproducibility
    files = os.listdir(data_dir)
    files = sorted(files)
    random.Random(SHUFFLE_SEED).shuffle(files)

    if max_files > 0:
        files = files[:int(max_files)]

    print('Reading files...')
    for index, file_name in enumerate(files):
        if not index % PROGRESS_ITER:
            print_progress(index, files)
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            # read array and parse all elements to float, keep NaNs as nan value
            contents = pd.read_csv(file_path, delimiter='|').values
            patient = PatientData(sequence_x=contents[:, :-1],
                                  sequence_y=contents[:, -1:])
            data.append(patient)

    print_progress(len(files), files)
    return data


def split_by_sepsis(data):
    sepsis_patients = [p for p in data if p.has_sepsis]
    healthy_patients = list(set(data) - set(sepsis_patients))
    return sepsis_patients, healthy_patients


def split_data(data):
    train_set, eval_set = train_test_split(data,
                                           train_size=RATIO_TRAIN_TO_EVAL,
                                           random_state=SHUFFLE_SEED,
                                           stratify=[p.has_sepsis for p in data])
    valid_set, test_set = train_test_split(eval_set,
                                           train_size=RATIO_VALID_TO_TEST,
                                           random_state=SHUFFLE_SEED,
                                           stratify=[p.has_sepsis for p in eval_set])
    return train_set, valid_set, test_set


def display_class_ratios(train=None, valid=None, test=None):

    def _get_counts(set_data, set_name):
        if set_data is not None:
            num_sepsis = np.count_nonzero([p.has_sepsis for p in set_data])
            total = len(set_data)
            print('{} dataset:\t\t{} sepsis patients among {} total ({:0.2f}%)'.format(
                set_name, num_sepsis, total, 100 * num_sepsis / total)
            )
            return num_sepsis / total

    print('\n')
    train_ratio = _get_counts(train, 'Train')
    _get_counts(valid, 'Valid')
    _get_counts(test, 'Test')
    print('\n')
    return train_ratio


def save_pickled_data(train=None, valid=None, test=None, transformer=None, dirname=None):
    if dirname is None:
        dirname = PICKLED_DIR

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    def _dump_data(list_data, list_name):
        with open(os.path.join(dirname, '{}_{}'.format(list_name, PICKLED_FILE_DATA)), 'wb') as f:
            pickle.dump(list_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nSaving to pickle...')
    _dump_data(train, 'train')
    _dump_data(valid, 'valid')
    _dump_data(test, 'test')
    _dump_data(transformer, 'transformer')
    print('Done')


if __name__ == "__main__":
    main()
