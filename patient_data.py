import copy
import numpy as np


class PatientData(object):
    def __init__(self, sequence_x, sequence_y):
        self.x, self.age, self.gender, self.adm_time, self.iculos, self._demo_sequence = \
            self.parse_demographic_data(np.asarray(sequence_x))
        self.y = np.asarray(sequence_y)
        self._original_seq_len, self._original_num_features = sequence_x.shape
        self.has_sepsis = 1. in self.y

    def __ge__(self, other):
        return self.seq_len > other.seq_len

    def __lt__(self, other):
        return self.seq_len < other.seq_len

    @property
    def true_seq_len(self):
        return self._original_seq_len

    @property
    def seq_len(self):
        return self.y.shape[0]

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def demo_features(self):
        return np.concatenate((self.age, [self.gender], self.adm_time), axis=1).squeeze()

    @staticmethod
    def parse_demographic_data(sequence_x):
        demographic_features = sequence_x[0, 34:-1]
        age, gender, _, _, adm_time = demographic_features
        if gender:
            gender = np.asarray([0.0, 1.0])
        else:
            gender = np.asarray([1.0, 0.0])
        iculos = sequence_x[:, -1:]  # intensive care unit length of stay
        return sequence_x[:, :34], age.reshape(-1, 1), gender, adm_time.reshape(-1, 1), iculos, sequence_x[:, 34:]

    def inverse_features_parsing(self):
        return np.array(np.concatenate((self.x, self._demo_sequence), axis=1)), self.y

    def multiply_sequences(self):
        # create multiple fake patients from self by splitting sequences at various lengths
        x, y = self.inverse_features_parsing()
        patients = []
        for row in range(self.true_seq_len):
            patients.append(PatientData(sequence_x=copy.deepcopy(x[:row + 1]),
                                        sequence_y=copy.deepcopy(y[:row + 1])))
        return patients
