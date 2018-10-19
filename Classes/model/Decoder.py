from keras.layers import *


class Decoder():
    def __init__(self, num_words=20000, embedding_size=128,
                 state_size=128, layers=2, dropout_rate=0.1):
        self.emb_layer = Embedding(num_words, embedding_size, mask_zero=True)
        cells = [GRUCell(state_size) for _ in range(layers)]
        self.rnn_layer = RNN(cells, return_sequences=True, return_state=True)
        self.out_layer = Dense(num_words)

    def __call__(self, x, initial_state):
        x = self.emb_layer(x)
        xh = self.rnn_layer(x, initial_state=initial_state)
        x, h = xh[0], xh[1:]
        x = TimeDistributed(self.out_layer)(x)
        return x, h
