import pickle

from utils import helper, collect
from Classes.TokenizerWrap import TokenizerWrap

from Config import *


if __name__ == "__main__":
    data_src = collect.get_data_nmt_dataset(DATASET_SRC)
    data_dest = collect.get_data_nmt_dataset(DATASET_DEST, start=START_WORD, end=END_WORD)

    tokenizer_src = TokenizerWrap(texts=data_src, padding='pre', reverse=True, num_words=NUM_WORDS, max_tokens=SEQ_LEN)
    tokenizer_dest = TokenizerWrap(texts=data_dest, padding='post', reverse=False, num_words=NUM_WORDS, max_tokens=SEQ_LEN)

    '''
    This dataset is huge so reloading is expensive and time-consuming.

    Thus, pickle tokenizer so we can reload since it's needed for inference and training.
    '''
    pickle.dump(tokenizer_src, open(TOKENIZER_SRC, "wb"))
    pickle.dump(tokenizer_dest, open(TOKENIZER_DEST, "wb"))
