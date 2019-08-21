import imblearn
import json
import numpy as np
import tensorflow as tf
import sklearn
from scipy.special import softmax
from matplotlib import pyplot as plt

from datetime import datetime
from inspect import signature
import os

from utils import timeit, print_progress
from tensor_names import TensorNames

MODEL_ROOT_DIR = os.path.join(os.path.dirname(__file__), 'prototypes')
MODEL_ROOT_BEST_LOSS = 'best_loss'
MODEL_ROOT_BEST_RECALL = 'best_recall'
MODEL_ROOT_BEST_FSCORE = 'best_f_score'

STATS_FILE = 'stats.txt'

RANDOM_SEED = 12345

RNN_OUTPUT_IS_SEQUENCE = False

BATCH_SIZE_TRAINING = 32
BATCH_SIZE_EVAL = 512
LEARNING_RATE = 1e-3

THRESHOLD = 0.5

USE_SOFT_TARGETS = True
SCALE_LOGITS_BY_TEMP = False
SOFT_TARGETS_MIN_TEMP = 10.0  # corresponds to 88:12 % ratio


class Model(object):
    def __init__(self, _sentinel=None, transformer=None,
                 train_patient_example=None, model_dir=None, sepsis_loss_scaling=None, max_num_epochs=None,
                 graph_builder=None, only_eval=False):
        assert _sentinel is None
        assert transformer is not None
        self._transformer = transformer

        self.graph = tf.Graph()
        self.saved_model_dir = model_dir
        self.sess = None
        self.saver = None
        self.saver_dir = None

        # train & eval mode
        self.ph = None
        self.logits = None
        self.prediction = None

        # train ops
        self.loss = None
        self.current_mean_loss = None
        self.update_mean_loss_op = None
        self.train_op = None
        self.local_init_op = None
        self.global_init_op = None

        self.max_num_epochs = max_num_epochs

        if not only_eval:
            assert graph_builder is not None
            assert max_num_epochs is not None
            self._graph_builder = graph_builder

        # create or restore graph
        with self.graph.as_default():
            self.get_session()
            if model_dir is None:
                # training mode
                self.build_new_graph(train_patient_example=train_patient_example,
                                     sepsis_loss_scaling=sepsis_loss_scaling)
                self.get_new_model_saver()

                self.writer_train = tf.summary.FileWriter(
                    logdir=os.path.join(self.saver_dir[MODEL_ROOT_DIR], 'train')
                )
                self.writer_valid = tf.summary.FileWriter(
                    logdir=os.path.join(self.saver_dir[MODEL_ROOT_DIR], 'valid')
                )
            else:
                # eval mode
                self.restore_graph(model_dir)

    def _get_tensor_by_str(self, substr):
        return [op.values()[0] for op in tf.get_default_graph().get_operations() if substr in op.name][0]

    def __enter__(self):
        self.sess.__enter__()
        if hasattr(self, 'writer_train'):
            self.writer_train.__enter__()
            self.writer_valid.__enter__()
        self.restore_variables()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'writer_train'):
            self.writer_train.__exit__(exc_type, exc_val, exc_tb)
            self.writer_valid.__exit__(exc_type, exc_val, exc_tb)
        self.sess.__exit__(exc_type, exc_val, exc_tb)

    def _get_datetime_now(self):
        now = datetime.now()
        return '{}{}{}{}{}{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    def get_session(self):
        self.sess = tf.Session(graph=self.graph)

    def log_scalar(self, writer, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        writer.add_summary(summary, step)

    """
    Save/restore
    """

    def save(self, saved_model_type, saved_model_stat):
        model_name = 'm'
        self.saver.save(sess=self.sess,
                        save_path=os.path.join(self.saver_dir[saved_model_type], model_name))

        with open(os.path.join(self.saver_dir[saved_model_type], STATS_FILE), 'w') as f:
            json.dump({saved_model_type: np.asscalar(saved_model_stat)}, f)

    def get_new_model_saver(self):
        self.saver = tf.train.Saver()
        model_name = 'm_{}'.format(self._get_datetime_now())
        model_root = os.path.normpath(os.path.join(MODEL_ROOT_DIR, model_name))
        self.saver_dir = {
            MODEL_ROOT_DIR: model_root,
            MODEL_ROOT_BEST_LOSS: os.path.normpath(os.path.join(model_root, MODEL_ROOT_BEST_LOSS)),
            MODEL_ROOT_BEST_RECALL: os.path.normpath(os.path.join(model_root, MODEL_ROOT_BEST_RECALL)),
            MODEL_ROOT_BEST_FSCORE: os.path.normpath(os.path.join(model_root, MODEL_ROOT_BEST_FSCORE)),
        }
        for model_path in self.saver_dir.values():
            if not os.path.exists(model_path):
                os.makedirs(model_path)

    def restore_variables(self):
        if self.saved_model_dir is not None:
            self.saver.restore(sess=self.sess,
                               save_path=tf.train.latest_checkpoint(self.saved_model_dir))

    """
    Graph construction
    """

    def restore_graph(self, model_dir):
        meta = [fname for fname in os.listdir(model_dir) if '.meta' in fname][0]
        self.saver = tf.train.import_meta_graph(os.path.join(model_dir, meta))

        self.ph = dict(PH_SEQ_X=self._get_tensor_by_str(TensorNames.PH_SEQ_X),
                       PH_SEQ_Y=self._get_tensor_by_str(TensorNames.PH_SEQ_Y),
                       PH_SEQ_LEN=self._get_tensor_by_str(TensorNames.PH_SEQ_LEN),
                       PH_IS_TRAINING=self._get_tensor_by_str(TensorNames.PH_IS_TRAINING),
                       PH_LR=self._get_tensor_by_str(TensorNames.PH_LR),
                       PH_DEMO=self._get_tensor_by_str(TensorNames.PH_DEMO),
                       PH_EPOCH_IDX=self._get_tensor_by_str(TensorNames.PH_EPOCH_IDX))
        self.logits = self._get_tensor_by_str(TensorNames.OUTPUT_LOGITS)
        self.prediction = self._get_tensor_by_str(TensorNames.OUTPUT_PREDICTIONS)

        # mean loss evaluation ops
        self.update_mean_loss_op = [
            op for op in tf.get_default_graph().get_operations() if 'get_running_mean_metrics' in op.name
        ][-1]

        try:
            self.local_init_op = [
                op for op in tf.get_default_graph().get_operations() if 'get_init_op/local/init' in op.name
            ][0]

        except IndexError:
            self.local_init_op = [
                op for op in tf.get_default_graph().get_operations() if 'get_init_op/init' in op.name
            ][0]

    def build_new_graph(self, train_patient_example, sepsis_loss_scaling=None):
        ph, logits, prediction, loss, current_mean_loss, update_mean_loss_op, \
            train_op, local_init_op, global_init_op = \
                self._graph_builder.build_new_graph(seq_len=train_patient_example.seq_len,
                                                    num_features=train_patient_example.num_features,
                                                    demo_features_len=len(train_patient_example.demo_features),
                                                    sepsis_loss_scaling=sepsis_loss_scaling,
                                                    rnn_output_is_sequence=RNN_OUTPUT_IS_SEQUENCE,
                                                    use_soft_targets=USE_SOFT_TARGETS,
                                                    scale_logits_by_temp=SCALE_LOGITS_BY_TEMP,
                                                    soft_targets_min_temp=SOFT_TARGETS_MIN_TEMP,
                                                    max_num_epochs=self.max_num_epochs)

        self.ph = ph

        # train & eval ops
        self.logits = logits
        self.prediction = prediction

        # train ops
        self.loss = loss
        self.current_mean_loss = current_mean_loss
        self.update_mean_loss_op = update_mean_loss_op
        self.train_op = train_op
        self.local_init_op = local_init_op
        self.global_init_op = global_init_op

    def make_soft_targets(self, y, epoch=None):
        if not USE_SOFT_TARGETS:
            return y
        else:
            if RNN_OUTPUT_IS_SEQUENCE:
                raise NotImplementedError('soft targets for rnn seq output')

            soft_logits = {
                0.0: np.asarray([1e1, -1e1]),
                1.0: np.asarray([-1e1, 1e1]),
            }
            if epoch is None:
                temperature = 1
            else:
                epoch = epoch if epoch > 0 else 1
                temperature = np.max((1, abs(SOFT_TARGETS_MIN_TEMP * self.max_num_epochs // epoch)))
            soft_targets = softmax(np.asarray(
                [soft_logits.get(row) / temperature for row in np.asarray(y).flatten()]
            ), axis=-1)
            return np.expand_dims(soft_targets, axis=1)

    """
    Training
    """

    @staticmethod
    def get_balanced_generator(patient_list, batch_size, seed_increment):
        indices = np.arange(0, len(patient_list))
        labels = np.asarray([1. * p.has_sepsis for p in patient_list])
        generator, num_steps = imblearn.tensorflow.balanced_batch_generator(X=indices.reshape(-1, 1),
                                                                            y=labels,
                                                                            batch_size=batch_size,
                                                                            random_state=RANDOM_SEED + seed_increment)
        if num_steps == 0:
            raise ValueError('Cannot create balanced sample generator, reduce batch_size or increase set.')
        return generator, num_steps

    @timeit
    def train_epoch(self, train, epoch_idx, batch_size=BATCH_SIZE_TRAINING, learning_rate=LEARNING_RATE):
        generator, num_unique_steps = self.get_balanced_generator(patient_list=train,
                                                                  batch_size=batch_size,
                                                                  seed_increment=epoch_idx)
        for _ in range(num_unique_steps):
            subsample_indices, _ = next(generator)
            train_subsample = np.take(train, subsample_indices.squeeze())

            self.run_train_op(train_subsample, learning_rate=learning_rate, epoch_idx=epoch_idx)

    def run_train_op(self, patient_list, learning_rate, epoch_idx):
        ph_x = []
        ph_y = []
        ph_seq_len = []
        ph_demo = []

        if RNN_OUTPUT_IS_SEQUENCE:
            for p in patient_list:
                ph_y.append(self._transformer.transform_targets(p.y))
                ph_seq_len.append(p.true_seq_len)
                ph_demo.append(self._transformer.transform_demo(p.demo_features))

                x_ = p.x[:p.true_seq_len, ...]
                x_ = self._transformer.transform_sequences(x_)
                ph_x.append(x_)
        else:
            # take a random subset of each sequence, classifying only the last relevant row
            for p in patient_list:
                seq_len_ = np.random.randint(1, p.true_seq_len + 1)  # start inclusive, stop exclusive
                y_ = self._transformer.transform_targets(p.y)[seq_len_ - 1].reshape(-1, 1)

                demo_ = p.demo_features
                demo_ = self._transformer.transform_demo(demo_)

                x_ = p.x[:seq_len_, ...]
                iculos_ = p.iculos[:seq_len_, ...]
                x_ = self._transformer.transform_sequences(sequences=x_,
                                                           iculos=iculos_)

                ph_seq_len.append(seq_len_)
                ph_demo.append(demo_)
                ph_y.append(y_)
                ph_x.append(x_)

        y = self.make_soft_targets(np.stack(ph_y, axis=0), epoch=epoch_idx)

        self.sess.run(
            fetches=self.train_op,
            feed_dict={
                self.ph.get('PH_SEQ_X'): np.stack(ph_x, axis=0),
                self.ph.get('PH_SEQ_Y'): y,
                self.ph.get('PH_SEQ_LEN'): np.stack(ph_seq_len, axis=0),
                self.ph.get('PH_DEMO'): np.stack(ph_demo, axis=0),
                self.ph.get('PH_IS_TRAINING'): True,
                self.ph.get('PH_LR'): learning_rate,
                self.ph.get('PH_EPOCH_IDX'): epoch_idx,
            }
        )

    """
    Model evaluation
    """

    def _transform_patient_data_and_append(self, patient, payload):
        x, y, seq_len, demo = payload

        seq_len.append(patient.true_seq_len)
        demo.append(self._transformer.transform_demo(patient.demo_features))

        if RNN_OUTPUT_IS_SEQUENCE:
            y.append(self._transformer.transform_targets(patient.y))
        else:
            y.append(
                self._transformer.transform_targets(patient.y,
                                                    substitute_early_detection=True,
                                                    pad_sequence=False)[-1].reshape(-1, 1)
            )

        x.append(self._transformer.transform_sequences(sequences=patient.x,
                                                       iculos=patient.iculos,
                                                       pad_sequence=False))
        return x, y, seq_len, demo

    @timeit
    def prepare_eval_dataset(self, patient_list, save_dir=None, save_fname=None):
        save = True if save_dir is not None else False

        if save:
            assert save_fname is not None
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            x = []
            y = []
            seq_len = []
            demo = []

        sepsis_patient_counter = -1
        for counter, patient in enumerate(patient_list):
            if not counter % 50:
                print_progress(counter, patient_list)
            # evaluates only on sepsis patients to speed up the process
            if not patient.has_sepsis:
                continue

            else:
                sepsis_patient_counter += 1

                patient_x = []
                patient_y = []
                patient_seq_len = []
                patient_demo = []

                payload = (patient_x, patient_y, patient_seq_len, patient_demo)

                if RNN_OUTPUT_IS_SEQUENCE:
                    payload = self._transform_patient_data_and_append(patient, payload)

                else:
                    for p in patient.multiply_sequences():
                        payload = self._transform_patient_data_and_append(p, payload)

                patient_x, patient_y, patient_seq_len, patient_demo = payload

                if save:
                    filename = save_fname + '_{}'.format(str(sepsis_patient_counter).zfill(7)) + '.npy'
                    with open(os.path.join(save_dir, filename), 'wb') as f:
                        np.save(f, tuple(zip(patient_x, patient_y, patient_seq_len, patient_demo)))
                else:
                    x.extend(patient_x)
                    y.extend(patient_y)
                    seq_len.extend(patient_seq_len)
                    demo.extend(patient_demo)

        print_progress(len(patient_list), patient_list)
        if not save:
            return x, y, seq_len, demo

    def get_evaluation_generator(self, eval_dataset, batch_size=BATCH_SIZE_EVAL):
        x, y, seq_len, demo = eval_dataset

        num_full_batches = len(x) // batch_size
        residual = len(x) % batch_size

        while True:
            batch_idx = -1
            for batch_idx in range(num_full_batches):
                s_ = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)
                if not residual and batch_idx == num_full_batches - 1:
                    last_batch = True
                else:
                    last_batch = False
                if RNN_OUTPUT_IS_SEQUENCE:
                    y_ = [
                        self._transformer.pad_seq_with_zeros(yrow, max_len=self._transformer.max_seq_len)
                        for yrow in y[s_]
                    ]
                else:
                    y_ = y[s_]
                x_ = [
                    self._transformer.pad_seq_with_zeros(xrow, max_len=self._transformer.max_seq_len)
                    for xrow in x[s_]
                ]
                yield x_, y_, seq_len[s_], demo[s_], last_batch

            if residual:
                s_ = slice((batch_idx + 1) * batch_size, len(x))
                if RNN_OUTPUT_IS_SEQUENCE:
                    y_ = [
                        self._transformer.pad_seq_with_zeros(yrow, max_len=self._transformer.max_seq_len)
                        for yrow in y[s_]
                    ]
                else:
                    y_ = y[s_]
                x_ = [
                    self._transformer.pad_seq_with_zeros(xrow, max_len=self._transformer.max_seq_len)
                    for xrow in x[s_]
                ]
                yield x_, y_, seq_len[s_], demo[s_], True

    def _get_predictions_in_loop(self, eval_generator):
        self.sess.run(self.local_init_op)  # reset tf.metrics
        mean_loss = None
        pred_list = []
        labels_list = []
        for x, y, seq_len, demo, is_last_batch in eval_generator:
            mean_loss, preds = self.sess.run(
                fetches=(self.update_mean_loss_op, self.prediction),
                feed_dict={
                    self.ph.get('PH_SEQ_X'): x,
                    self.ph.get('PH_SEQ_Y'): self.make_soft_targets(y),
                    self.ph.get('PH_SEQ_LEN'): seq_len,
                    self.ph.get('PH_DEMO'): demo,
                    self.ph.get('PH_IS_TRAINING'): False,
                    self.ph.get('PH_EPOCH_IDX'): -1,
                }
            )

            sepsis_prob = preds[0, 0, 1]

            if USE_SOFT_TARGETS:
                preds = np.argmax(preds, axis=-1)
            else:
                preds = 1.0 * np.greater_equal(preds, THRESHOLD)

            if RNN_OUTPUT_IS_SEQUENCE:
                flat_true = np.concatenate([y_[:len_] for y_, len_ in zip(y, seq_len)], axis=0)
                flat_pred = np.concatenate([preds_[:len_] for preds_, len_ in zip(preds, seq_len)], axis=0)
            else:
                flat_true = np.asarray(y).flatten()
                flat_pred = np.asarray(preds).flatten()
            labels_list.extend(flat_true)
            pred_list.extend([sepsis_prob])

            if is_last_batch:
                break

        return mean_loss, labels_list, pred_list

    @timeit
    def evaluate_model(self, patient_eval_generator):
        mean_loss, labels_list, pred_list = self._get_predictions_in_loop(patient_eval_generator)

        print(sklearn.metrics.classification_report(y_true=np.asarray(labels_list).flatten(),
                                                    y_pred=np.asarray(pred_list).flatten()))
        _, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(
            y_true=np.asarray(labels_list).flatten(),
            y_pred=np.asarray(pred_list).flatten(),
            average='binary'
        )
        print('Mean loss: {:0.3f}'.format(mean_loss))
        print(sklearn.metrics.confusion_matrix(y_true=np.asarray(labels_list).flatten(),
                                               y_pred=np.asarray(pred_list).flatten()))
        print('\n')
        return mean_loss, recall, fscore

    def precision_recall_curve(self, y_true, y_pred, title_str=''):
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)

        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve {}'.format(title_str))
        plt.show()

        return precision, recall, thresholds

    """
    Prediction
    """

    def get_prediction_generator(self, patient_list, batch_size):
        x = []
        y = []
        seq_len = []
        demo = []

        for patient in patient_list:
            payload = x, y, seq_len, demo
            x, y, seq_len, demo = self._transform_patient_data_and_append(patient, payload)

        dataset = x, y, seq_len, demo
        return self.get_evaluation_generator(dataset, batch_size)

    @timeit
    def get_predictions(self, patient_list, batch_size):
        patient_generator = self.get_prediction_generator(patient_list, batch_size)
        _, labels_list, pred_list = self._get_predictions_in_loop(patient_generator)

        return np.asarray(labels_list).flatten(), np.asarray(pred_list).flatten()
