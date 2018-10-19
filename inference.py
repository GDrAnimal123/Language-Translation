import numpy as np
import os
import pickle

from utils import helper
from Classes.Seq2Seq import Seq2Seq
from Classes.TokenizerWrap import TokenizerWrap

from Config import *

if __name__ == '__main__':

    # Load our Tokenizer
    tokenizer_src, tokenizer_dest = pickle.load(open(TOKENIZER_SRC, "rb")), pickle.load(open(TOKENIZER_DEST, "rb"))

    # For translate
    s2s = Seq2Seq(tokenizer_src, tokenizer_dest, START_WORD, END_WORD)
    s2s.build(NUM_WORDS, EMBEDDING_SIZE, STATE_SIZE, LAYERS, DROPOUT_RATE)

    # Load weight to predict
    # The underline represents our model_train, however we don't need it for evaluation
    _ = helper.load_weight(s2s.model, WEIGHT_PATH)

    print('\n')

    # Set some dummy example
    for input_text, actual_text in zip(["thank you very much", "i like her"], ["cảm ơn rất nhiều", "tôi thích cô ấy"]):
        predict_text = s2s.predict_greedy(str(input_text))
        helper.print_prediction(input_text, predict_text, actual_text)
    print('\n')
