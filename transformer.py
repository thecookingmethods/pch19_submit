import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from utils import print_progress

RARE_MEASUREMENTS_THRESHOLD = None
PROGRESS_ITER = 50

FORCE_EARLY_PREDICTION = True  # up to six additional rows will be substituted 0 -> 1, if patient has sepsis
DOUBLE_FEATURES_NAN_MASK = True  # add additional features corresponding to boolean on NaN location


class Transformer(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self._sequences = ModifiedStandardScaler()
        self._age = ModifiedStandardScaler()
        self._adm_time = ModifiedStandardScaler()
        self._is_fit = False

    @property
    def is_fit(self):
        return self._is_fit

    @property
    def rare_features(self):
        if RARE_MEASUREMENTS_THRESHOLD is None:
            return np.zeros_like(self._sequences.n_samples_seen_).astype(np.bool)
        else:
            relative_frequencies = self._sequences.n_samples_seen_ / np.amax(self._sequences.n_samples_seen_)
            return np.less_equal(relative_frequencies, RARE_MEASUREMENTS_THRESHOLD)

    def fit(self, train):
        # NaNs are treated as missing values: disregarded in fit, and maintained in transform
        print('Fitting Transformer...')
        for counter, patient in enumerate(train):
            if not counter % PROGRESS_ITER:
                print_progress(counter, train)
            self._sequences.partial_fit(patient.x)
            self._age.partial_fit(patient.age)
            self._adm_time.partial_fit(patient.adm_time)
        print_progress(len(train), train)
        self._is_fit = True
        return self

    def transform_patients(self, patient_list, mute=False, pad_sequence=True):
        assert self.is_fit
        # NaNs are are not changed in transform() call
        if not mute:
            print('Using Transformer...')
        for index in range(len(patient_list)):
            if not mute and not index % PROGRESS_ITER:
                print_progress(index, patient_list)

            patient_list[index].age = self.transform_age(patient_list[index].age)
            patient_list[index].adm_time = self.transform_adm_time(patient_list[index].adm_time)
            patient_list[index].x = self.transform_sequences(sequences=patient_list[index].x,
                                                             iculos=patient_list[index].iculos,
                                                             pad_sequence=pad_sequence)
            patient_list[index].y = self.transform_targets(patient_list[index].y,
                                                           pad_sequence=pad_sequence)

        if not mute:
            print_progress(len(patient_list), patient_list)
        return patient_list

    def transform_age(self, age):
        age = copy.deepcopy(age)
        if np.isnan(age.squeeze()):
            return np.array([[0.0]])
        else:
            return self._age.transform(age)

    def transform_adm_time(self, adm_time):
        adm_time = copy.deepcopy(adm_time)
        if np.isnan(adm_time.squeeze()):
            return np.array([[0.0]])
        else:
            return self._adm_time.transform(adm_time)

    def transform_demo(self, demo):
        demo = copy.deepcopy(demo)
        demo[0] = self.transform_age(demo[0].reshape(-1, 1)).flatten()
        demo[-1] = self.transform_adm_time(demo[-1].reshape(-1, 1)).flatten()
        return demo

    def transform_sequences(self, sequences, iculos, pad_sequence=True):
        sequences = copy.deepcopy(sequences)
        sequences_tmp = self._sequences.transform(sequences)
        sequences_tmp = self._repair_data(sequences_tmp)
        if pad_sequence:
            sequences_tmp = self.pad_seq_with_zeros(sequences_tmp)

        iculos = self.transform_iculos(iculos, pad_sequence=pad_sequence)
        sequences_tmp = np.concatenate((sequences_tmp, iculos), axis=-1)
        return sequences_tmp

    def transform_targets(self, targets, substitute_early_detection=FORCE_EARLY_PREDICTION, pad_sequence=True):
        targets = copy.deepcopy(targets)
        if substitute_early_detection:
            targets = self._substitute_early_detection_rows(targets)
        if pad_sequence:
            targets = self.pad_seq_with_zeros(targets)
        return targets

    def _substitute_early_detection_rows(self, targets):
        if np.count_nonzero(targets) > 0:
            first_sepsis_row = np.argmax(targets)
            if first_sepsis_row != 0:
                # the first sepsis row is actually (t = t_sepsis - 6h) to promote optimal early detection
                # we add up to additional 6h of sepsis because of scoring
                new_target = np.zeros_like(targets)
                new_target[np.max((0, first_sepsis_row - 6)):, :] = 1.0
                targets = new_target

        return targets

    def transform_iculos(self, iculos, pad_sequence=True):
        iculos = copy.deepcopy(iculos)
        # TODO transform ICULOS into probability of sepsis given time spent in ICU
        if pad_sequence:
            iculos = self.pad_seq_with_zeros(iculos)
        return iculos

    def _repair_data(self, sequences):
        if DOUBLE_FEATURES_NAN_MASK:
            double_features_mask = 1. * np.isnan(sequences)
        else:
            double_features_mask = None

        for col_idx in range(sequences.shape[1]):
            nan_mask = np.isnan(sequences[:, col_idx])

            if np.sum(nan_mask) == 0:
                # column has no NaNs - skip
                pass

            else:
                # otherwise latch last known non-NaN value
                # (in case of header NaNs: backpropagate first known value, when available)
                latch_value = None
                header_latch_last_idx = -1
                for row_idx in range(sequences.shape[0]):
                    if latch_value is None and np.isnan(sequences[row_idx, col_idx]):
                        # NaN in column head => store and update once some value is known
                        header_latch_last_idx = row_idx

                    elif np.isnan(sequences[row_idx, col_idx]):
                        # NaN after know value => apply latch
                        sequences[row_idx, col_idx] = latch_value

                    elif not np.isnan(sequences[row_idx, col_idx]):
                        # known value - store in latch
                        latch_value = sequences[row_idx, col_idx]

                        if header_latch_last_idx >= 0:
                            # backpropagate latch in header
                            sequences[:header_latch_last_idx + 1, col_idx] = latch_value
                            header_latch_last_idx = -1

        # replace all other NaNs with zeros
        sequences[np.isnan(sequences)] = 0.0

        # finally, remove all columns which rarely contain non-NaN values
        sequences = sequences[:, np.logical_not(self.rare_features)]

        if DOUBLE_FEATURES_NAN_MASK:
            sequences = np.concatenate(
                (sequences, double_features_mask[:, np.logical_not(self.rare_features)]),
                axis=-1
            )

        return sequences

    def pad_seq_with_zeros(self, seq, max_len=None):
        if max_len is None:
            max_len = self.max_seq_len
        pad_len = max_len - len(seq)
        padding = np.zeros(shape=[pad_len] + list(seq.shape[1:]))
        return np.concatenate((seq, padding), axis=0)


class ModifiedStandardScaler(StandardScaler):
    def partial_fit(self, *args, **kwargs):
        if hasattr(self, 'var_'):
            var_copy = self.var_.copy()
            scale_copy = self.scale_.copy()
        super().partial_fit(*args, **kwargs)

        # if some column has not been seen, change running estimate from NaN to zero
        mask = self.n_samples_seen_ == 0
        self.mean_[mask] = 0.0
        self.var_[mask] = 0.0
        self.scale_[mask] = 0.0

        # if some column only contains NaNs then computing partial variance is impossible
        # and the above check fails; copy old estimate
        if np.any(np.isnan(self.var_)):
            nan_mask = np.isnan(self.var_)
            self.var_[nan_mask] = var_copy[nan_mask]
            self.scale_[nan_mask] = scale_copy[nan_mask]
