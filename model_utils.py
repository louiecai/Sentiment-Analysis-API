import os
from typing import Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim
import torchtext.legacy.data as data
from torch.jit import RecursiveScriptModule

import models

MODELS = {'lstm': models.LSTM, }


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class EarlyStopping:
    def __init__(self, tolerance: int = 5, min_delta: float = 0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def get_model(model_name: str) -> nn.Module:
    return MODELS[model_name] if model_name in MODELS else None


def get_accuracy(model: nn.Module, dataloader: data.BucketIterator, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch.text.to(device)
            t = batch.airline_sentiment.to(device)
            y = model.forward(X)
            _, y_pred = torch.max(y, 1)
            correct += (y_pred == t).sum().item()
    return correct / len(dataloader.dataset)


def train_model(train_dataloader: data.BucketIterator, valid_dataloader: data.BucketIterator, model: nn.Module,
                optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, epochs: int = 50,
                early_stop: bool = False):
    if early_stop:
        early_stopping = EarlyStopping(min_delta=0.02)
    total_train_loss, total_valid_loss = [], []
    total_train_accuracy, total_valid_accuracy = [], []

    for epoch in np.arange(1, epochs + 1):
        print(f'Epoch {epoch}:')
        model.train()
        train_losses, valid_loss = [], []
        for batch_idx, batch in enumerate(train_dataloader):
            X = batch.text.to(device)
            t = batch.airline_sentiment.to(device)

            y = model.forward(X)
            loss = criterion(y, t)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_accuracy = get_accuracy(model, train_dataloader, device)

        print(f'train_loss: {np.mean(train_losses)}, train_accuracy: {train_accuracy}.')
        total_train_loss.append(np.mean(train_losses))
        total_train_accuracy.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                X = batch.text.to(device)
                t = batch.airline_sentiment.to(device)
                y = model.forward(X)
                loss = criterion(y, t)
                valid_loss.append(loss.item())

            valid_accuracy = get_accuracy(model, valid_dataloader, device)
            print(f'valid_loss: {np.mean(valid_loss)}, valid_accuracy: {valid_accuracy}.')

            total_valid_loss.append(np.mean(valid_loss))
            total_valid_accuracy.append(valid_accuracy)

        if early_stop:
            early_stopping(np.mean(train_losses), np.mean(valid_loss))
            if early_stopping.early_stop:
                print('Early stopping!')
                break

    return total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy, model


def test_model(test_dataloader: data.BucketIterator, model: nn.Module, device: torch.device):
    pass  # TODO: implement this function


def model_predict(model: nn.Module, text: str, device: torch.device, text_field: data.Field) -> tuple[str, Any]:
    model.eval()
    token_indices = [text_field.vocab.stoi[token] for token in text_field.tokenize(text)]
    token_tensor = torch.LongTensor(token_indices).to(device).unsqueeze(1)
    prediction = torch.nn.functional.softmax(model.forward(token_tensor), dim=1)
    return 'negative' if prediction.argmax().item() == 0 else 'positive', prediction


def save_model(model: nn.Module, text_field: data.Field, path: str) -> None:
    torch.save(model, os.getcwd() + f'/{path}/model.pt')
    torch.save(text_field, os.getcwd() + f'/{path}/text_field.pt')


def load_model(path: str) -> tuple[nn.Module, data.Field]:
    model = torch.load(os.path.join(os.getcwd(), path, 'model.pt'))
    text_field = torch.load(os.path.join(os.getcwd(), path, 'text_field.pt'))
    return model, text_field


def plot_result(total_train_loss: list, total_train_accuracy: list, total_valid_loss: list, total_valid_accuracy: list):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_train_loss, label='train_loss')
    plt.plot(total_valid_loss, label='valid_loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(total_train_accuracy, label='train_accuracy')
    plt.plot(total_valid_accuracy, label='valid_accuracy')
    plt.legend()
    plt.savefig('result.png')
