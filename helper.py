import numpy as np


def get_data_nmt_dataset(path, start="", end=""):
     # Open and read all the contents of the data-file.
    with open(path, encoding="utf-8") as file:
        return [start + line.lower().strip().split("\n")[0] + end for line in file]


def get_training_data(src, dest):

    encoder_input_data = src
    '''
    Teacher-forcing (Refer this great blog "https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html")
    Basically, you feed the word from the output sequence with encoder state and try to predict the next word

    You could try to take the top prediction (topk from tensorflow) from your model
    and use that to predict your next word, but it makes your code very complicated :|
    '''
    decoder_input_data = dest[:, :-1]
    decoder_output_data = dest[:, 1:]

    return encoder_input_data, decoder_input_data, decoder_output_data


def print_prediction(input_text, output_text, true_output_text=None):
    # Print the input-text.
    print("Input: {}".format(input_text))
    print("Translate: {}".format(output_text))

    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text: {}".format(true_output_text))

    print()


def load_weight(model, path=None):
    if path is not None:
        print("Load model")
        try:
            model.load_weights(path)
        except Exception as error:
            print(error)
            print("Error trying to load checkpoint.")
    return model