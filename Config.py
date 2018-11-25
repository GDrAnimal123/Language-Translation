import time

# Prerequisites parameters
START_WORD = 'ssss '
END_WORD = ' eeee'
NUM_WORDS = 20000       # Set this higher if your GPU can afford it.
SEQ_LEN = 50            # Set this higher if your GPU can afford it.

# Hyperparams
EPOCHS = 5
BATCH_SIZE = 128        # Set this higher if your GPU can afford it.
EMBEDDING_SIZE = 256
STATE_SIZE = 512
DROPOUT_RATE = 0.1
LAYERS = 2              # Set this higher if your GPU can afford it.
LR = 1e-3

# Location of your training datasets
TRAIN_SRC = "datasets/train/train.en"
TRAIN_DEST = "datasets/train/train.vi"

VALIDATION_SRC = "datasets/valid/valid.en"
VALIDATION_DEST = "datasets/valid/valid.vi"

# Pretrained model path for inference
WEIGHT_PATH = "checkpoint/model/translation_st512-em256-w20000-05-38.9995.h5"

# Location to save your Weight and Tensorboard logs
LOGS_PATH = './checkpoint/logs/'
MODEL_PATH = 'checkpoint/model/translation_st{}-em{}-w{}'.format(STATE_SIZE, EMBEDDING_SIZE, NUM_WORDS)
