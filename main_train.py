import numpy as np
import os
import pickle
import copy

import shutil
from parse_dataset import PICKLED_DIR, PICKLED_FILE_DATA
from transformer import Transformer, ModifiedStandardScaler
from patient_data import PatientData

from model import Model, MODEL_ROOT_BEST_LOSS, MODEL_ROOT_BEST_FSCORE, MODEL_ROOT_BEST_RECALL, \
    BATCH_SIZE_TRAINING, BATCH_SIZE_EVAL, LEARNING_RATE

from utils import timeit

# prevent auto-cleanup; needed when loading pickle
PatientData = PatientData
Transformer = Transformer
ModifiedStandardScaler = ModifiedStandardScaler

TAG_MEAN_LOSS = 'mean_loss'
TAG_RECALL = 'recall'
TAG_FSCORE = 'f_score'

EVAL_EVERY_NUM_EPOCHS = 30

NUM_EPOCHS = 1000
PICKLED_DIR_RNN = 'pickled_rnn_data'


def main():
    train = load_pickled_data('train', dirname=PICKLED_DIR)
    transformer = load_pickled_data('transformer', dirname=PICKLED_DIR)

    train_example = copy.deepcopy(train[:1])
    train_example = transformer.transform_patients(train_example,
                                                   mute=True,
                                                   pad_sequence=True)[0]

    model = Model(train_patient_example=train_example,
                  transformer=transformer,
                  graph_builder=GraphBuilder(),
                  max_num_epochs=NUM_EPOCHS)

    try:
        eval_train = load_eval_dataset('train', dirname=PICKLED_DIR_RNN)
        eval_valid = load_eval_dataset('valid', dirname=PICKLED_DIR_RNN)

    except (FileNotFoundError, EOFError):
        if os.path.exists(PICKLED_DIR_RNN):
            shutil.rmtree(PICKLED_DIR_RNN)

        model.prepare_eval_dataset(patient_list=train,
                                   save_dir=PICKLED_DIR_RNN,
                                   save_fname='train')

        valid = load_pickled_data('valid', dirname=PICKLED_DIR)
        model.prepare_eval_dataset(patient_list=valid,
                                   save_dir=PICKLED_DIR_RNN,
                                   save_fname='valid')

        eval_train = load_eval_dataset('train', dirname=PICKLED_DIR_RNN)
        eval_valid = load_eval_dataset('valid', dirname=PICKLED_DIR_RNN)

    print('Train set shape: ({}, {}, {})'.format(
        len(eval_train[0]), transformer.max_seq_len, eval_train[0][0].shape[-1]
    ))
    print('Valid set shape: ({}, {}, {})\n'.format(
        len(eval_valid[0]), transformer.max_seq_len, eval_valid[0][0].shape[-1]
    ))

    with model:
        model.sess.run(model.local_init_op)
        model.sess.run(model.global_init_op)
        model.writer_train.add_graph(graph=model.graph)

        train_eval_generator = model.get_evaluation_generator(eval_train,
                                                              batch_size=BATCH_SIZE_EVAL)
        valid_eval_generator = model.get_evaluation_generator(eval_valid,
                                                              batch_size=BATCH_SIZE_EVAL)

        best_loss = np.Inf
        best_recall = 0.0
        best_fscore = 0.0

        for epoch_idx in range(NUM_EPOCHS):
            print('{} Training epoch #{} {}\n'.format('=' * 10, epoch_idx, '=' * 10))
            model.train_epoch(train, epoch_idx, batch_size=BATCH_SIZE_TRAINING, learning_rate=LEARNING_RATE)

            if not epoch_idx % EVAL_EVERY_NUM_EPOCHS:
                print('Epoch done. Evaluating...')

                print('Training set:')
                eval_model(model=model,
                           eval_generator=train_eval_generator,
                           writer=model.writer_train,
                           epoch_idx=epoch_idx)

                print('Validation set:')
                mean_valid_loss, valid_recall, valid_fscore = eval_model(model=model,
                                                                         eval_generator=valid_eval_generator,
                                                                         writer=model.writer_valid,
                                                                         epoch_idx=epoch_idx)

                if mean_valid_loss < best_loss:
                    print('Loss improved, saving model ({:0.4f} vs {:0.4f}).\n'.format(
                        mean_valid_loss, best_loss
                    ))
                    best_loss = mean_valid_loss
                    model.save(saved_model_type=MODEL_ROOT_BEST_LOSS,
                               saved_model_stat=best_loss)

                if valid_recall > best_recall:
                    print('Recall improved, saving model ({:0.2f} vs {:0.2f}).\n'.format(
                        valid_recall, best_recall
                    ))
                    best_recall = valid_recall
                    model.save(saved_model_type=MODEL_ROOT_BEST_RECALL,
                               saved_model_stat=best_recall)

                if valid_fscore > best_fscore:
                    print('F score improved, saving model ({:0.2f} vs {:0.2f}).\n'.format(
                        valid_fscore, best_fscore
                    ))
                    best_fscore = valid_fscore
                    model.save(saved_model_type=MODEL_ROOT_BEST_FSCORE,
                               saved_model_stat=best_fscore)


@timeit
def load_pickled_data(filename, dirname):
    with open(os.path.join(dirname, '{}_{}'.format(filename, PICKLED_FILE_DATA)), 'rb') as f:
        return pickle.load(f)


@timeit
def load_eval_dataset(filename, dirname):
    matching = sorted([match for match in os.listdir(dirname) if filename in match])
    data = []
    if len(matching) > 0:
        for fname in matching:
            with open(os.path.join(dirname, fname), 'rb') as f:
                data.extend(np.load(f).tolist())
        return tuple(zip(*data))
    else:
        raise FileNotFoundError


def eval_model(model, eval_generator, writer, epoch_idx):
    mean_loss, recall, fscore = model.evaluate_model(eval_generator)
    model.log_scalar(writer=writer,
                     tag=TAG_MEAN_LOSS,
                     value=mean_loss,
                     step=epoch_idx)
    model.log_scalar(writer=writer,
                     tag=TAG_RECALL,
                     value=recall,
                     step=epoch_idx)
    model.log_scalar(writer=writer,
                     tag=TAG_FSCORE,
                     value=fscore,
                     step=epoch_idx)
    return mean_loss, recall, fscore


if __name__ == "__main__":
    main()
