import os

import numpy as np
import torch
import torch.nn as nn
from torchtext.legacy import data


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, tolerance: int = 5, min_delta: float = 0):
        """
        :param tolerance: How many epochs to wait before early stopping.
        :param min_delta: Minimum change in the monitored value to qualify as an improvement.
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss: float, validation_loss: float) -> None:
        """
        Update the counter and check if early stopping should be used.
        :param train_loss: Training loss
        :param validation_loss: Validation loss
        :return: None
        """
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class RNN(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, output_size: int, text_field: data.Field,
                 label_field: data.LabelField, num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False,
                 is_lstm: bool = True):
        """
        :param input_size: Size of the input vector.
        :param embedding_size: Size of the embedding vector.
        :param hidden_size: Size of the hidden layer.
        :param output_size: Size of the output vector.
        :param text_field: Text field. Saved in the model.
        :param num_layers: Number of layers.
        :param dropout: Dropout probability.
        :param bidirectional: Whether to use a bidirectional LSTM.
        """
        super(RNN, self).__init__()
        self.text_field = text_field  # saved for prediction
        self.label_field = label_field  # saved for prediction
        self.is_lstm = is_lstm
        self.embedding = nn.Embedding(input_size, embedding_size)  # embedding layer
        if is_lstm:
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout,
                               bidirectional=bidirectional)  # LSTM layer
        else:
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout,
                              bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)  # fully connected layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the data through the network.
        :param x: Input data.
        :return: Output data.
        """
        embedded = self.embedding(x)
        if self.is_lstm:
            _, (hidden, _) = self.rnn(embedded)
        else:
            _, hidden = self.rnn(embedded)
        hidden.squeeze_(0)
        if len(hidden.shape) == 3:
            hidden = hidden[-1]
        output = self.fc(hidden)
        return output

    def train_model(self, train_dataloader: data.BucketIterator, valid_dataloader: data.BucketIterator,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, epochs: int = 50,
                    early_stop: bool = False):
        """
        Train the model with dataloaders and optimizer.
        :param train_dataloader: Training dataloader.
        :param valid_dataloader: Validation dataloader.
        :param optimizer: Optimizer.
        :param criterion: Criterion.
        :param device: Device.
        :param epochs: Number of epochs.
        :param early_stop: Whether to use early stopping.
        """
        if early_stop:
            early_stopping = EarlyStopping(min_delta=0.02)
        total_train_loss, total_valid_loss = [], []
        total_train_accuracy, total_valid_accuracy = [], []

        for epoch in np.arange(1, epochs + 1):
            print(f'Epoch {epoch}:')
            self.train()
            train_losses, valid_loss = [], []
            for batch_idx, batch in enumerate(train_dataloader):
                X = batch.text.to(device)
                t = batch.airline_sentiment.to(device)

                y = self.forward(X)
                loss = criterion(y, t)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_accuracy = RNN.get_accuracy(self, train_dataloader, device)

            print(f'train_loss: {np.mean(train_losses)}, train_accuracy: {train_accuracy}.')
            total_train_loss.append(np.mean(train_losses))
            total_train_accuracy.append(train_accuracy)

            self.eval()
            with torch.no_grad():
                for batch in valid_dataloader:
                    X = batch.text.to(device)
                    t = batch.airline_sentiment.to(device)
                    y = self.forward(X)
                    loss = criterion(y, t)
                    valid_loss.append(loss.item())

                valid_accuracy = RNN.get_accuracy(self, valid_dataloader, device)
                print(f'valid_loss: {np.mean(valid_loss)}, valid_accuracy: {valid_accuracy}.')

                total_valid_loss.append(np.mean(valid_loss))
                total_valid_accuracy.append(valid_accuracy)

            if early_stop:
                early_stopping(np.mean(train_losses), np.mean(valid_loss))
                if early_stopping.early_stop:
                    print('Early stopping!')
                    break

        return total_train_loss, total_train_accuracy, total_valid_loss, total_valid_accuracy

    def predict(self, text: str, device: torch.device) -> tuple[str, torch.LongTensor]:
        """
        Predict the sentiment of a text.
        :param text: Text.
        :param device: Device.
        :return: Tuple of the predicted sentiment and the probability of the sentiment.
        """
        self.eval()
        with torch.no_grad():
            token_indices = [self.text_field.vocab.stoi[token] for token in self.text_field.tokenize(text)]
            token_tensor = torch.LongTensor(token_indices).to(device).unsqueeze(1)
            prediction = torch.nn.functional.softmax(self.forward(token_tensor), dim=1)
            return self.label_field.vocab.itos[prediction.argmax().item()], prediction

    def save(self, path: str) -> None:
        """
        Save the model to `path`.
        :param path: Path to save the model.
        """
        torch.save(self, os.getcwd() + f'/{path}/model.pt')

    @staticmethod
    def load(path: str):
        """
        Load a model from `path`.
        :param path: Path to load the model.
        :return: the model.
        """
        model = torch.load(os.path.join(os.getcwd(), path))
        return model

    @staticmethod
    def get_accuracy(model: nn.Module, dataloader: data.BucketIterator, device: torch.device) -> float:
        """
        Returns the accuracy of the model on the data.
        :param model: Model.
        :param dataloader: data used to calculate the accuracy.
        :param device: device of the model.
        :return: Accuracy as a float
        """
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
