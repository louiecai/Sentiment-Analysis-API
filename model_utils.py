import os.path

import matplotlib.pyplot as plt
from models import LSTM
import torch.nn as nn

MODELS = {'lstm': LSTM, }


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_model(model_name: str) -> nn.Module:
    return MODELS[model_name] if model_name in MODELS else None


def plot_result(total_train_loss: list, total_train_accuracy: list, total_valid_loss: list, total_valid_accuracy: list,
                path: str) -> None:
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
