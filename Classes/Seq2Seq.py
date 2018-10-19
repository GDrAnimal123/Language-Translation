import numpy as np

from .model.Encoder import Encoder
from .model.Decoder import Decoder

import tensorflow as tf
from keras.models import *
from keras.layers import *


class Seq2Seq(object):
    def __init__(self, tokenizer_src, tokenizer_dest, start_word="", end_word="", context_length=50):

        self.tokenizer_src = tokenizer_src
        self.tokenizer_dest = tokenizer_dest

        self.start_word = start_word
        self.end_word = end_word

        self.context_length = context_length

    def build(self, num_words=20000, embedding_size=128, state_size=256, layers=2, dropout_rate=0.1):
        encoder_inputs = Input(shape=(None,), dtype='int32')
        decoder_inputs = Input(shape=(None,), dtype='int32')

        encoder = Encoder(num_words, embedding_size, state_size, layers, dropout_rate)
        decoder = Decoder(num_words, embedding_size, state_size, layers, dropout_rate)

        encoder_outputs, encoder_states = encoder(encoder_inputs)

        dinputs = Lambda(lambda x: x[:, :-1])(decoder_inputs)
        dtargets = Lambda(lambda x: x[:, 1:])(decoder_inputs)

        decoder_outputs, decoder_state_h = decoder(dinputs, encoder_states)

        def get_loss(args):
            y_pred, y_true = args
            y_true = tf.cast(y_true, 'int32')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accuracy(args):
            y_pred, y_true = args
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = Lambda(get_loss)([decoder_outputs, dtargets])
        self.loss = Lambda(K.exp)(loss)
        self.accuracy = Lambda(get_accuracy)([decoder_outputs, dtargets])

        self.model = Model([encoder_inputs, decoder_inputs], loss)
        self.model.add_loss([K.mean(loss)])

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_states_inputs = [Input(shape=(state_size,)) for _ in range(layers)]

        decoder_outputs, decoder_states = decoder(decoder_inputs, decoder_states_inputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def compile(self, optimizer):
        self.model.compile(optimizer, None)
        self.model.metrics_names.append('loss')
        self.model.metrics_tensors.append(self.loss)
        self.model.metrics_names.append('accuracy')
        self.model.metrics_tensors.append(self.accuracy)

    def predict_greedy(self, input_text):
        input_mat = self.tokenizer_src.text_to_tokens(text=input_text,
                                                      reverse=True,
                                                      padding=True)

        state_value = self.encoder_model.predict(input_mat)
        target_seq = np.zeros((1, 1))

        token_start = self.tokenizer_dest.word_index[self.start_word.strip()]
        token_end = self.tokenizer_dest.word_index[self.end_word.strip()]
        sampled_token = token_start

        max_tokens = self.tokenizer_dest.max_tokens

        target_seq[0, 0] = token_start

        decoded_tokens = []

        while sampled_token != token_end and len(decoded_tokens) < max_tokens:
            output_tokens_and_h = self.decoder_model.predict([target_seq] + state_value)
            output_tokens, h = output_tokens_and_h[0], output_tokens_and_h[1:]
            sampled_token = np.argmax(output_tokens[0, -1, :])

            sampled_word = self.tokenizer_dest.token_to_word(sampled_token)

            decoded_tokens.append(sampled_word)

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token
            state_value = h

        return ' '.join(decoded_tokens[:-1])
