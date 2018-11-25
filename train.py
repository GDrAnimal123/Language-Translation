# Standard imports
import numpy as np
import pickle
import os
import random

# My own imports
from utils import helper, collect
from Classes.Seq2Seq import Seq2Seq
from Classes.TokenizerWrap import TokenizerWrap

# Keras imports
from keras.optimizers import RMSprop
from keras.callbacks import *

# Get necessary parameters
from Config import *


class TestCallback(Callback):
    def __init__(self, src, dest, num_predictions=5):
        super(TestCallback, self).__init__()
        self.src = src
        self.dest = dest
        self.num_predictions = 5 if num_predictions < len(src) else len(src)

    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        pairs = [[src_seq, dest_seq] for src_seq, dest_seq in zip(self.src, self.dest)]

        random_valid_pairs = random.sample(pairs, k=self.num_predictions)
        for pairs in random_valid_pairs:
            src_seq = pairs[0]
            dest_seq = pairs[1]

            prediction = s2s.predict_greedy(str(src_seq))
            helper.print_prediction(src_seq, prediction, dest_seq)
        print('\n')


if __name__ == "__main__":

    try:
        # Load data for validation
        valid_src = collect.get_data_nmt_dataset(VALIDATION_SRC)
        valid_dest = collect.get_data_nmt_dataset(VALIDATION_DEST)
    except FileNotFoundError:
        print("No validation set in your directory...")

        # If no validation found, we initialized our own text
        valid_src = ["thank you very much", "i like her"]
        valid_dest = ["cảm ơn rất nhiều", "tôi thích cô ấy"]

    # Load our Tokenizer
    tokenizer_src, tokenizer_dest = pickle.load(open(TOKENIZER_SRC, "rb")), pickle.load(open(TOKENIZER_DEST, "rb"))

    encoder_input_data = tokenizer_src.tokens_padded
    decoder_output_data = tokenizer_dest.tokens_padded

    # Trainning data
    x_train = encoder_input_data
    y_train = decoder_output_data

    print("Input shape: {}".format(x_train.shape))
    print("Output shape: {}".format(y_train.shape))

    # Initialize our model
    s2s = Seq2Seq(tokenizer_src, tokenizer_dest, START_WORD, END_WORD)
    s2s.build(NUM_WORDS, EMBEDDING_SIZE, STATE_SIZE, LAYERS, DROPOUT_RATE)
    s2s.compile('rmsprop')
    s2s.model.summary()
    s2s.decoder_model.summary()

    # # Necessary callbacks
    # callback_print = TestCallback(valid_src, valid_dest)
    # callback_early_stopping = EarlyStopping(patience=3, verbose=1)
    # callback_tensorboard = TensorBoard(log_dir=LOGS_PATH)
    # callback_checkpoint = ModelCheckpoint(MODEL_PATH + '-{epoch:02d}-{val_loss:.4f}.h5',
    #                                       monitor='val_loss',
    #                                       save_best_only=True,
    #                                       period=1)
    # callbacks = [callback_print, callback_early_stopping, callback_checkpoint, callback_tensorboard]  # callback_tensorboard, callback_checkpoint

    # s2s.model.fit([x_train, y_train], None, batch_size=BATCH_SIZE, epochs=EPOCHS,
    #               validation_split=0.2, callbacks=callbacks)
