class TensorNames:
    # important: don't change strings (we search by name during restoration)
    PH_SEQ_X = 'ph_seq_x'
    PH_SEQ_Y = 'ph_seq_y'
    PH_SEQ_LEN = 'ph_seq_len'
    PH_IS_TRAINING = 'ph_is_training'
    PH_LR = 'ph_lr'
    PH_DEMO = 'ph_demo'
    PH_EPOCH_IDX = 'ph_epoch'
    OUTPUT_LOGITS = 'out_logits'
    OUTPUT_PREDICTIONS = 'out_predictions'