import numpy as np


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
