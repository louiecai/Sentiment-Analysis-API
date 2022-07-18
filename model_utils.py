import os.path

import matplotlib.pyplot as plt
from model import RNN


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def plot_result(total_train_loss: list, total_train_accuracy: list, total_valid_loss: list, total_valid_accuracy: list,
                path: str) -> None:
    """
    Plot the training and validation loss and accuracy.
    :param total_train_loss: list of training loss.
    :param total_train_accuracy: list of training accuracy.
    :param total_valid_loss: list of validation loss.
    :param total_valid_accuracy: list of validation accuracy.
    :param path: path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_train_loss, label='train_loss')
    plt.plot(total_valid_loss, label='valid_loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(total_train_accuracy, label='train_accuracy')
    plt.plot(total_valid_accuracy, label='valid_accuracy')
    plt.legend()
    plt.savefig(os.path.join(path, 'result.png'))
